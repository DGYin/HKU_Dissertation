"""Application service that wires the CrossMedia-PID pipeline together."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from crossmedia_pid.config import project_path
from crossmedia_pid.core.extractor import PersonExtractor
from crossmedia_pid.core.feature_vlm import create_feature_extractor
from crossmedia_pid.core.matcher import IdentityMatcher
from crossmedia_pid.core.vectorizer import DynamicVectorizer
from crossmedia_pid.db.chroma_store import ChromaStore

console = Console()


class CrossMediaPID:
    """Main application facade for person extraction, feature extraction, and matching."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

        console.print("[bold blue]Initializing CrossMedia-PID...[/bold blue]")

        yolo_config = config.get("models", {}).get("yolo", {})
        self.extractor = PersonExtractor(
            model_path=yolo_config.get("model_path", str(project_path("models", "yolov8n.pt"))),
            conf_threshold=yolo_config.get("conf_threshold", 0.5),
            iou_threshold=yolo_config.get("iou_threshold", 0.45),
            min_bbox_size=config.get("features", {}).get("min_bbox_size", 64),
        )

        vlm_config = self._prepare_vlm_config(config.get("models", {}).get("vlm", {}))
        self.feature_extractor = create_feature_extractor(vlm_config)

        embedding_config = config.get("models", {}).get("embedding", {})
        registry_config = config.get("registry", {})
        self.vectorizer = DynamicVectorizer(
            dense_model_name=embedding_config.get("model_name", "BAAI/bge-small-zh-v1.5"),
            dense_onnx_path=embedding_config.get("onnx_path"),
            max_length=embedding_config.get("max_length", 512),
            registry_path=registry_config.get(
                "persist_path",
                str(project_path("data", "attribute_registry.json")),
            ),
        )

        chroma_config = config.get("database", {}).get("chroma", {})
        self.store = ChromaStore(
            persist_directory=chroma_config.get(
                "persist_directory",
                str(project_path("data", "chroma_db")),
            ),
            collection_name=chroma_config.get("collection_name", "person_embeddings"),
            distance_fn=chroma_config.get("distance_fn", "cosine"),
        )

        matching_config = config.get("matching", {})
        self.matcher = IdentityMatcher(
            store=self.store,
            threshold=matching_config.get("threshold", 0.72),
            top_k=matching_config.get("top_k", 5),
            weights=matching_config.get("weights"),
            enable_face=matching_config.get("enable_face", False),
        )

        console.print("[bold green]System initialized successfully![/bold green]")

    def _prepare_vlm_config(self, vlm_config: dict[str, Any]) -> dict[str, Any]:
        config = dict(vlm_config)
        provider = config.get("provider", "cloud")

        if provider == "cloud":
            config["api_key"] = config.get("api_key") or os.getenv("VLM_API_KEY", "")
            if not config["api_key"]:
                console.print("[yellow]Warning: VLM_API_KEY not set. Cloud API will fail.[/yellow]")
        elif provider == "aliyun":
            config["api_key"] = config.get("api_key") or os.getenv("DASHSCOPE_API_KEY", "")
            if not config["api_key"]:
                console.print("[yellow]Warning: DASHSCOPE_API_KEY not set. Aliyun API will fail.[/yellow]")

        return config

    def process_image(self, image_path: Path, add_to_db: bool = True) -> dict[str, Any] | None:
        """Process a single image and optionally persist the matched identity."""
        console.print(f"\n[bold]Processing:[/bold] {image_path}")
        start_time = time.time()

        console.print("  [yellow]Step 1/4:[/yellow] Visual extraction...", end=" ")
        visual_output = self.extractor.extract(image_path, return_best_only=True)
        if visual_output is None:
            console.print("[red]FAILED - No person detected[/red]")
            return None
        console.print(f"[green]OK[/green] (quality={visual_output.quality_score:.2f})")

        console.print("  [yellow]Step 2/4:[/yellow] Feature extraction...", end=" ")
        feature_output = self.feature_extractor.extract(visual_output.crop_image)
        if not feature_output.is_valid:
            console.print(f"[red]FAILED - {feature_output.raw_response[:50]}...[/red]")
            return None
        console.print(f"[green]OK[/green] ({len(feature_output.attributes)} attributes)")

        console.print("  [yellow]Step 3/4:[/yellow] Vectorization...", end=" ")
        vector_output = self.vectorizer.vectorize(
            feature_output.attributes,
            source_meta={
                "source_path": str(image_path),
                "quality_score": visual_output.quality_score,
            },
        )
        console.print("[green]OK[/green]")

        console.print("  [yellow]Step 4/4:[/yellow] Identity matching...", end=" ")
        match_output = self.matcher.match(
            dense_vector=vector_output.dense_vector,
            sparse_vector=vector_output.sparse_vector,
            query_attributes=feature_output.attributes,
        )

        if match_output.is_new_identity:
            console.print(f"[cyan]NEW IDENTITY[/cyan] ({match_output.person_uuid})")
        else:
            console.print(
                f"[green]MATCHED[/green] ({match_output.person_uuid}, "
                f"score={match_output.match_score:.3f})"
            )

        if add_to_db:
            self.matcher.add_identity(
                person_uuid=match_output.person_uuid,
                dense_vector=vector_output.dense_vector,
                sparse_vector=vector_output.sparse_vector,
                attributes=feature_output.attributes,
                source_meta={
                    "source_path": str(image_path),
                    "quality_score": visual_output.quality_score,
                    "detection_conf": visual_output.detection_confidence,
                },
            )

        elapsed = time.time() - start_time
        console.print(f"  [dim]Total time: {elapsed:.2f}s[/dim]")

        return {
            "image_path": str(image_path),
            "person_uuid": match_output.person_uuid,
            "is_new": match_output.is_new_identity,
            "match_score": match_output.match_score,
            "attributes": feature_output.attributes,
            "quality_score": visual_output.quality_score,
            "elapsed_time": elapsed,
        }

    def search_by_image(self, image_path: Path, top_k: int = 5) -> list[dict[str, Any]]:
        """Search similar identities by image."""
        console.print(f"\n[bold]Searching with image:[/bold] {image_path}")

        visual_output = self.extractor.extract(image_path, return_best_only=True)
        if visual_output is None:
            console.print("[red]No person detected[/red]")
            return []

        feature_output = self.feature_extractor.extract(visual_output.crop_image)
        if not feature_output.is_valid:
            console.print("[red]Feature extraction failed[/red]")
            return []

        vector_output = self.vectorizer.vectorize(feature_output.attributes)
        return self.matcher.search_similar(
            dense_vector=vector_output.dense_vector,
            sparse_vector=vector_output.sparse_vector,
            top_k=top_k,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return system statistics."""
        return {
            "total_records": self.store.count(),
            "unique_persons": len(self.store.get_all_person_uuids()),
            "registry_stats": self.vectorizer.get_registry_stats(),
        }
