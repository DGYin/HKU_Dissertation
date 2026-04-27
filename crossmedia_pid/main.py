"""
CrossMedia-PID CLI Entry Point
跨媒体人物识别系统 - Phase 1 CLI

Usage:
    python main.py process <image_path> [options]
    python main.py batch <image_dir> [options]
    python main.py search <image_path> [options]
    python main.py stats
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.extractor import PersonExtractor, VisualOutput
from core.feature_vlm import FeatureExtractor, FeatureOutput, create_feature_extractor
from core.matcher import IdentityMatcher, MatchOutput
from core.vectorizer import DynamicVectorizer, VectorOutput
from db.chroma_store import ChromaStore

console = Console()


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件，支持环境变量替换"""
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 处理环境变量替换
    def replace_env_vars(obj):
        if isinstance(obj, dict):
            return {k: replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            default_val = ""
            if ':' in env_var:
                env_var, default_val = env_var.split(':', 1)
            return os.getenv(env_var, default_val)
        return obj
    
    return replace_env_vars(config)


class CrossMediaPID:
    """主控制器"""
    
    def __init__(self, config: dict):
        """初始化系统"""
        self.config = config
        
        # 初始化各模块
        console.print("[bold blue]Initializing CrossMedia-PID...[/bold blue]")
        
        # A模块：视觉提取
        yolo_config = config.get('models', {}).get('yolo', {})
        self.extractor = PersonExtractor(
            model_path=yolo_config.get('model_path', 'yolov8n.pt'),
            conf_threshold=yolo_config.get('conf_threshold', 0.5),
            iou_threshold=yolo_config.get('iou_threshold', 0.45),
            min_bbox_size=config.get('features', {}).get('min_bbox_size', 64)
        )
        
        # B模块：特征提取
        vlm_config = config.get('models', {}).get('vlm', {})
        
        # 检查API密钥
        provider = vlm_config.get('provider', 'cloud')
        if provider == 'cloud':
            api_key = vlm_config.get('api_key', '')
            if not api_key:
                api_key = os.getenv('VLM_API_KEY', '')
            if not api_key:
                console.print("[yellow]Warning: VLM_API_KEY not set. Cloud API will fail.[/yellow]")
            vlm_config['api_key'] = api_key
        elif provider == 'aliyun':
            api_key = vlm_config.get('api_key', '')
            if not api_key:
                api_key = os.getenv('DASHSCOPE_API_KEY', '')
            if not api_key:
                console.print("[yellow]Warning: DASHSCOPE_API_KEY not set. Aliyun API will fail.[/yellow]")
            vlm_config['api_key'] = api_key
        
        self.feature_extractor = create_feature_extractor(vlm_config)
        
        # C模块：向量化
        embedding_config = config.get('models', {}).get('embedding', {})
        registry_config = config.get('registry', {})
        self.vectorizer = DynamicVectorizer(
            dense_model_name=embedding_config.get('model_name', 'BAAI/bge-small-zh-v1.5'),
            max_length=embedding_config.get('max_length', 512),
            registry_path=registry_config.get('persist_path', './attribute_registry.json')
        )
        
        # D模块：匹配
        chroma_config = config.get('database', {}).get('chroma', {})
        self.store = ChromaStore(
            persist_directory=chroma_config.get('persist_directory', './chroma_db'),
            collection_name=chroma_config.get('collection_name', 'person_embeddings'),
            distance_fn=chroma_config.get('distance_fn', 'cosine')
        )
        
        matching_config = config.get('matching', {})
        self.matcher = IdentityMatcher(
            store=self.store,
            threshold=matching_config.get('threshold', 0.72),
            top_k=matching_config.get('top_k', 5),
            weights=matching_config.get('weights'),
            enable_face=False  # Phase 1禁用
        )
        
        console.print("[bold green]System initialized successfully![/bold green]")
    
    def process_image(
        self,
        image_path: Path,
        add_to_db: bool = True
    ) -> Optional[dict]:
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            add_to_db: 是否添加到数据库
            
        Returns:
            处理结果字典
        """
        console.print(f"\n[bold]Processing:[/bold] {image_path}")
        
        start_time = time.time()
        
        # Step 1: 视觉提取 (A模块)
        console.print("  [yellow]Step 1/4:[/yellow] Visual extraction...", end=" ")
        visual_output = self.extractor.extract(image_path, return_best_only=True)
        
        if visual_output is None:
            console.print("[red]FAILED - No person detected[/red]")
            return None
        
        console.print(f"[green]OK[/green] (quality={visual_output.quality_score:.2f})")
        
        # Step 2: 特征提取 (B模块)
        console.print("  [yellow]Step 2/4:[/yellow] Feature extraction...", end=" ")
        feature_output = self.feature_extractor.extract(visual_output.crop_image)
        
        if not feature_output.is_valid:
            console.print(f"[red]FAILED - {feature_output.raw_response[:50]}...[/red]")
            return None
        
        console.print(f"[green]OK[/green] ({len(feature_output.attributes)} attributes)")
        
        # Step 3: 向量化 (C模块)
        console.print("  [yellow]Step 3/4:[/yellow] Vectorization...", end=" ")
        vector_output = self.vectorizer.vectorize(
            feature_output.attributes,
            source_meta={
                'source_path': str(image_path),
                'quality_score': visual_output.quality_score
            }
        )
        console.print("[green]OK[/green]")
        
        # Step 4: 身份匹配 (D模块)
        console.print("  [yellow]Step 4/4:[/yellow] Identity matching...", end=" ")
        match_output = self.matcher.match(
            dense_vector=vector_output.dense_vector,
            sparse_vector=vector_output.sparse_vector,
            query_attributes=feature_output.attributes
        )
        
        if match_output.is_new_identity:
            console.print(f"[cyan]NEW IDENTITY[/cyan] ({match_output.person_uuid})")
        else:
            console.print(f"[green]MATCHED[/green] ({match_output.person_uuid}, score={match_output.match_score:.3f})")
        
        # 添加到数据库
        if add_to_db:
            self.matcher.add_identity(
                person_uuid=match_output.person_uuid,
                dense_vector=vector_output.dense_vector,
                sparse_vector=vector_output.sparse_vector,
                attributes=feature_output.attributes,
                source_meta={
                    'source_path': str(image_path),
                    'quality_score': visual_output.quality_score,
                    'detection_conf': visual_output.detection_confidence
                }
            )
        
        elapsed = time.time() - start_time
        console.print(f"  [dim]Total time: {elapsed:.2f}s[/dim]")
        
        return {
            'image_path': str(image_path),
            'person_uuid': match_output.person_uuid,
            'is_new': match_output.is_new_identity,
            'match_score': match_output.match_score,
            'attributes': feature_output.attributes,
            'quality_score': visual_output.quality_score,
            'elapsed_time': elapsed
        }
    
    def search_by_image(self, image_path: Path, top_k: int = 5) -> list:
        """以图搜图"""
        console.print(f"\n[bold]Searching with image:[/bold] {image_path}")
        
        # 提取特征
        visual_output = self.extractor.extract(image_path, return_best_only=True)
        if visual_output is None:
            console.print("[red]No person detected[/red]")
            return []
        
        feature_output = self.feature_extractor.extract(visual_output.crop_image)
        if not feature_output.is_valid:
            console.print("[red]Feature extraction failed[/red]")
            return []
        
        vector_output = self.vectorizer.vectorize(feature_output.attributes)
        
        # 搜索
        results = self.matcher.search_similar(
            dense_vector=vector_output.dense_vector,
            sparse_vector=vector_output.sparse_vector,
            top_k=top_k
        )
        
        return results
    
    def get_stats(self) -> dict:
        """获取系统统计"""
        return {
            'total_records': self.store.count(),
            'unique_persons': len(self.store.get_all_person_uuids()),
            'registry_stats': self.vectorizer.get_registry_stats()
        }


@click.group()
@click.option('--config', '-c', default='configs/config.yaml', help='配置文件路径')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
@click.pass_context
def cli(ctx, config, verbose):
    """CrossMedia-PID 跨媒体人物识别系统"""
    # 加载配置
    cfg = load_config(config)
    
    # 设置日志
    log_level = "DEBUG" if verbose else cfg.get('logging', {}).get('level', 'INFO')
    setup_logging(log_level)
    
    # 初始化系统
    ctx.ensure_object(dict)
    ctx.obj['config'] = cfg
    ctx.obj['pid'] = CrossMediaPID(cfg)


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--no-add', is_flag=True, help='不添加到数据库')
@click.pass_context
def process(ctx, image_path, no_add):
    """处理单张图片"""
    pid = ctx.obj['pid']
    result = pid.process_image(Path(image_path), add_to_db=not no_add)
    
    if result:
        # 显示属性
        table = Table(title="Extracted Attributes")
        table.add_column("Attribute", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in result['attributes'].items():
            table.add_row(key, str(value))
        
        console.print(table)


@cli.command()
@click.argument('image_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--pattern', '-p', default='*.jpg', help='文件匹配模式')
@click.option('--limit', '-l', type=int, help='最大处理数量')
@click.pass_context
def batch(ctx, image_dir, pattern, limit):
    """批量处理图片目录"""
    pid = ctx.obj['pid']
    image_dir = Path(image_dir)
    
    # 查找图片
    images = list(image_dir.glob(pattern))
    images += list(image_dir.glob(pattern.replace('jpg', 'jpeg')))
    images += list(image_dir.glob(pattern.replace('jpg', 'png')))
    images = sorted(set(images))
    
    if limit:
        images = images[:limit]
    
    console.print(f"[bold]Found {len(images)} images to process[/bold]")
    
    # 批量处理
    results = []
    success_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing...", total=len(images))
        
        for img_path in images:
            progress.update(task, description=f"Processing {img_path.name}...")
            
            result = pid.process_image(img_path)
            if result:
                results.append(result)
                success_count += 1
            
            progress.advance(task)
    
    # 统计
    console.print(f"\n[bold green]Batch processing complete![/bold green]")
    console.print(f"  Total: {len(images)}")
    console.print(f"  Success: {success_count}")
    console.print(f"  Failed: {len(images) - success_count}")
    
    if results:
        avg_time = sum(r['elapsed_time'] for r in results) / len(results)
        console.print(f"  Avg time: {avg_time:.2f}s")


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--top-k', '-k', default=5, help='返回结果数量')
@click.pass_context
def search(ctx, image_path, top_k):
    """以图搜图"""
    pid = ctx.obj['pid']
    results = pid.search_by_image(Path(image_path), top_k=top_k)
    
    if not results:
        console.print("[yellow]No similar persons found[/yellow]")
        return
    
    # 显示结果
    table = Table(title="Search Results")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Person UUID", style="green")
    table.add_column("Total Score", style="yellow")
    table.add_column("Dense", style="blue")
    table.add_column("Sparse", style="magenta")
    
    for i, result in enumerate(results, 1):
        table.add_row(
            str(i),
            result['person_uuid'],
            f"{result['total_score']:.3f}",
            f"{result['dense_score']:.3f}",
            f"{result['sparse_score']:.3f}"
        )
    
    console.print(table)


@cli.command()
@click.pass_context
def stats(ctx):
    """显示系统统计"""
    pid = ctx.obj['pid']
    stats = pid.get_stats()
    
    table = Table(title="System Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Records", str(stats['total_records']))
    table.add_row("Unique Persons", str(stats['unique_persons']))
    table.add_row("Registry Attributes", str(stats['registry_stats']['total_attributes']))
    table.add_row("Verified Attributes", str(stats['registry_stats']['verified_attributes']))
    
    console.print(table)


if __name__ == '__main__':
    cli()
