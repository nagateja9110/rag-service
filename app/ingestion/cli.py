"""CLI entrypoint for the offline pipeline.

Deliberately a separate process from the API. Ingestion is bursty, CPU- and
network-bound, and occasionally long-running; the serving path must stay
responsive and horizontally scalable. Coupling them (an ``/ingest`` endpoint
that does the work inline) means one large PDF can starve every query worker.

    python -m app.ingestion.cli ingest --data-dir ./data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.vector_store import VectorStoreError, get_vector_store
from app.ingestion.pipeline import IngestionError, run_ingestion

logger = get_logger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Set log level to DEBUG.")
def cli(verbose: bool) -> None:
    """Offline ingestion tooling for the RAG service."""
    settings = get_settings()
    configure_logging(
        level="DEBUG" if verbose else settings.log_level,
        fmt=settings.log_format,
    )


@cli.command()
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Source directory (defaults to DATA_DIR).",
)
@click.option("--collection", default=None, help="Target collection name.")
@click.option(
    "--rebuild",
    is_flag=True,
    help="Drop the collection and re-ingest everything from scratch.",
)
@click.option(
    "--no-prune",
    is_flag=True,
    help="Keep stored chunks that no longer appear in re-ingested files.",
)
@click.option("--dry-run", is_flag=True, help="Report what would change; embed nothing.")
def ingest(
    data_dir: Path | None,
    collection: str | None,
    rebuild: bool,
    no_prune: bool,
    dry_run: bool,
) -> None:
    """Load, chunk, embed and store documents. Idempotent."""
    if rebuild and not dry_run:
        click.confirm(
            "--rebuild deletes the entire collection and re-embeds every chunk "
            "(this costs money). Continue?",
            abort=True,
        )
    try:
        stats = run_ingestion(
            data_dir=data_dir,
            collection_name=collection,
            rebuild=rebuild,
            prune=not no_prune,
            dry_run=dry_run,
        )
    except (IngestionError, VectorStoreError) as exc:
        logger.error("Ingestion failed: %s", exc)
        raise SystemExit(1) from exc

    click.echo(json.dumps(stats.as_dict(), indent=2))


@cli.command()
@click.option("--collection", default=None, help="Collection to inspect.")
def status(collection: str | None) -> None:
    """Show collection size and connectivity."""
    store = get_vector_store(collection)
    settings = get_settings()
    try:
        payload = {
            "mode": settings.chroma_mode,
            "collection": collection or settings.chroma_collection,
            "healthy": store.healthy(),
            "vectors": store.count(),
        }
    except VectorStoreError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    click.echo(json.dumps(payload, indent=2))


def main() -> None:
    try:
        cli()
    except Exception as exc:
        logger.exception("Unhandled error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
