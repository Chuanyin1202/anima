#!/usr/bin/env python3
"""Qdrant Migration Script: AnimaAgent -> Anima

This script migrates the memory collection from the old naming convention
to the new one, and updates metadata fields.

Changes:
1. Copy collection: anima_AnimaAgent -> anima_Anima
2. Update metadata: about="xiao_guang" -> about="self"

WARNING:
- This script modifies production data. BACKUP FIRST!
- The old collection is NOT deleted (kept as backup).
- Use --dry-run to preview changes without modifying data.

Usage:
    python scripts/migrate_qdrant.py                    # Interactive mode
    python scripts/migrate_qdrant.py --dry-run          # Preview only
    python scripts/migrate_qdrant.py --source X --target Y  # Custom collections
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

load_dotenv(project_root / ".env")


def get_client() -> QdrantClient:
    """Create Qdrant client from environment."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError("QDRANT_URL not set")

    return QdrantClient(url=url, api_key=api_key)


def migrate_collection(
    client: QdrantClient,
    source: str,
    target: str,
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict:
    """Migrate all points from source to target collection.

    Args:
        client: Qdrant client
        source: Source collection name
        target: Target collection name
        batch_size: Number of points per batch
        dry_run: If True, only preview changes without modifying data

    Returns:
        Migration statistics
    """
    stats = {
        "total_points": 0,
        "migrated_points": 0,
        "updated_about_fields": 0,
        "errors": [],
    }

    # Check source exists
    collections = [c.name for c in client.get_collections().collections]
    if source not in collections:
        raise ValueError(f"Source collection '{source}' not found. Available: {collections}")

    # Get source collection info
    source_info = client.get_collection(source)
    vector_size = source_info.config.params.vectors.size
    distance = source_info.config.params.vectors.distance

    print(f"Source collection: {source}")
    print(f"  Vector size: {vector_size}")
    print(f"  Distance: {distance}")
    print(f"  Points: {source_info.points_count}")

    stats["total_points"] = source_info.points_count

    # Create target collection if not exists
    if target not in collections:
        if dry_run:
            print(f"\n[DRY-RUN] Would create target collection: {target}")
        else:
            print(f"\nCreating target collection: {target}")
            client.create_collection(
                collection_name=target,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
    else:
        print(f"\nTarget collection '{target}' already exists")
        target_info = client.get_collection(target)
        print(f"  Existing points: {target_info.points_count}")

    # Scroll through all points and migrate
    print(f"\nMigrating points (batch size: {batch_size})...")

    offset = None
    batch_num = 0

    while True:
        # Scroll points from source
        result = client.scroll(
            collection_name=source,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )

        points, next_offset = result

        if not points:
            break

        batch_num += 1
        print(f"  Batch {batch_num}: {len(points)} points")

        # Transform points
        new_points = []
        for point in points:
            # Update metadata
            payload = dict(point.payload) if point.payload else {}

            # Fix 'about' field
            if payload.get("about") == "xiao_guang":
                payload["about"] = "self"
                stats["updated_about_fields"] += 1

            new_points.append(
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=payload,
                )
            )

        # Upsert to target
        if dry_run:
            stats["migrated_points"] += len(new_points)
        else:
            try:
                client.upsert(
                    collection_name=target,
                    points=new_points,
                )
                stats["migrated_points"] += len(new_points)
            except Exception as e:
                stats["errors"].append(f"Batch {batch_num}: {e}")
                print(f"    Error: {e}")

        # Continue or stop
        if next_offset is None:
            break
        offset = next_offset

    return stats


def main():
    """Run migration."""
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant collection and update metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/migrate_qdrant.py --dry-run
  python scripts/migrate_qdrant.py --source anima_Old --target anima_New
  python scripts/migrate_qdrant.py --yes  # Skip confirmation
        """,
    )
    parser.add_argument(
        "--source",
        default="anima_AnimaAgent",
        help="Source collection name (default: anima_AnimaAgent)",
    )
    parser.add_argument(
        "--target",
        default="anima_Anima",
        help="Target collection name (default: anima_Anima)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying data",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    print("=" * 60)
    if args.dry_run:
        print("Qdrant Migration [DRY-RUN MODE]")
    else:
        print("Qdrant Migration")
    print("=" * 60)

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}This will migrate:")
    print(f"  Source: {args.source}")
    print(f"  Target: {args.target}")
    print(f"  QDRANT_URL: {os.getenv('QDRANT_URL', 'NOT SET')}")
    print(f"\nMetadata changes:")
    print(f"  about='xiao_guang' -> about='self'")

    if not args.dry_run:
        print("\n⚠️  WARNING: This modifies production data!")
        print("   - Old collection is kept as backup")
        print("   - Use --dry-run to preview first")

    if not args.yes and not args.dry_run:
        response = input("\nProceed? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return

    print("\n" + "-" * 60)

    client = get_client()

    try:
        stats = migrate_collection(
            client=client,
            source=args.source,
            target=args.target,
            dry_run=args.dry_run,
        )

        print("\n" + "=" * 60)
        if args.dry_run:
            print("Dry-Run Complete! (No changes made)")
        else:
            print("Migration Complete!")
        print("=" * 60)
        print(f"Total points in source: {stats['total_points']}")
        print(f"{'Would migrate' if args.dry_run else 'Migrated'} points: {stats['migrated_points']}")
        print(f"{'Would update' if args.dry_run else 'Updated'} 'about' fields: {stats['updated_about_fields']}")

        if stats["errors"]:
            print(f"\nErrors ({len(stats['errors'])}):")
            for err in stats["errors"]:
                print(f"  - {err}")

        # Verify (only if not dry-run)
        if not args.dry_run:
            target_info = client.get_collection(args.target)
            print(f"\nTarget collection now has: {target_info.points_count} points")

    except Exception as e:
        print(f"\nMigration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
