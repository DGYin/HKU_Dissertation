"""Command line interface for CrossMedia-PID."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from crossmedia_pid.app import CrossMediaPID
from crossmedia_pid.config import DEFAULT_CONFIG_PATH, load_config

console = Console()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_CONFIG_PATH,
    show_default=True,
    help="配置文件路径",
)
@click.option("--verbose", "-v", is_flag=True, help="详细输出")
@click.pass_context
def cli(ctx: click.Context, config: Path, verbose: bool) -> None:
    """CrossMedia-PID 跨媒体人物识别系统."""
    cfg = load_config(config)
    log_level = "DEBUG" if verbose else cfg.get("logging", {}).get("level", "INFO")
    setup_logging(log_level)

    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["pid"] = CrossMediaPID(cfg)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--no-add", is_flag=True, help="不添加到数据库")
@click.pass_context
def process(ctx: click.Context, image_path: Path, no_add: bool) -> None:
    """处理单张图片."""
    pid: CrossMediaPID = ctx.obj["pid"]
    result = pid.process_image(image_path, add_to_db=not no_add)

    if result:
        table = Table(title="Extracted Attributes")
        table.add_column("Attribute", style="cyan")
        table.add_column("Value", style="green")

        for key, value in result["attributes"].items():
            table.add_row(key, str(value))

        console.print(table)


@cli.command()
@click.argument("image_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--pattern", "-p", default="*.jpg", help="文件匹配模式")
@click.option("--limit", "-l", type=int, help="最大处理数量")
@click.pass_context
def batch(ctx: click.Context, image_dir: Path, pattern: str, limit: int | None) -> None:
    """批量处理图片目录."""
    pid: CrossMediaPID = ctx.obj["pid"]

    images = list(image_dir.glob(pattern))
    images += list(image_dir.glob(pattern.replace("jpg", "jpeg")))
    images += list(image_dir.glob(pattern.replace("jpg", "png")))
    images = sorted(set(images))

    if limit:
        images = images[:limit]

    console.print(f"[bold]Found {len(images)} images to process[/bold]")

    results = []
    success_count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(images))
        for image_path in images:
            progress.update(task, description=f"Processing {image_path.name}...")
            result = pid.process_image(image_path)
            if result:
                results.append(result)
                success_count += 1
            progress.advance(task)

    console.print("\n[bold green]Batch processing complete![/bold green]")
    console.print(f"  Total: {len(images)}")
    console.print(f"  Success: {success_count}")
    console.print(f"  Failed: {len(images) - success_count}")

    if results:
        avg_time = sum(result["elapsed_time"] for result in results) / len(results)
        console.print(f"  Avg time: {avg_time:.2f}s")


@cli.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--top-k", "-k", default=5, help="返回结果数量")
@click.pass_context
def search(ctx: click.Context, image_path: Path, top_k: int) -> None:
    """以图搜图."""
    pid: CrossMediaPID = ctx.obj["pid"]
    results = pid.search_by_image(image_path, top_k=top_k)

    if not results:
        console.print("[yellow]No similar persons found[/yellow]")
        return

    table = Table(title="Search Results")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Person UUID", style="green")
    table.add_column("Total Score", style="yellow")
    table.add_column("Dense", style="blue")
    table.add_column("Sparse", style="magenta")

    for index, result in enumerate(results, 1):
        table.add_row(
            str(index),
            result["person_uuid"],
            f"{result['total_score']:.3f}",
            f"{result['dense_score']:.3f}",
            f"{result['sparse_score']:.3f}",
        )

    console.print(table)


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """显示系统统计."""
    pid: CrossMediaPID = ctx.obj["pid"]
    stats_result = pid.get_stats()

    table = Table(title="System Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Records", str(stats_result["total_records"]))
    table.add_row("Unique Persons", str(stats_result["unique_persons"]))
    table.add_row("Registry Attributes", str(stats_result["registry_stats"]["total_attributes"]))
    table.add_row("Verified Attributes", str(stats_result["registry_stats"]["verified_attributes"]))

    console.print(table)


if __name__ == "__main__":
    cli()
