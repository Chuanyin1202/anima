#!/usr/bin/env python3
"""Qdrant Migration Script: AnimaAgent -> Anima

This script migrates the memory collection from the old naming convention
to the new one, and updates metadata fields.

Changes:
1. Copy collection: anima_AnimaAgent -> anima_Anima
2. Update metadata: about="xiao_guang" -> about="self"
"""

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
) -> dict:
    """Migrate all points from source to target collection.

    Args:
        client: Qdrant client
        source: Source collection name
        target: Target collection name
        batch_size: Number of points per batch

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
    print("=" * 60)
    print("Qdrant Migration: AnimaAgent -> Anima")
    print("=" * 60)

    source_collection = "anima_AnimaAgent"
    target_collection = "anima_Anima"

    # Confirm
    print(f"\nThis will migrate:")
    print(f"  Source: {source_collection}")
    print(f"  Target: {target_collection}")
    print(f"\nMetadata changes:")
    print(f"  about='xiao_guang' -> about='self'")

    response = input("\nProceed? (y/N): ")
    if response.lower() != "y":
        print("Aborted.")
        return

    print("\n" + "-" * 60)

    client = get_client()

    try:
        stats = migrate_collection(
            client=client,
            source=source_collection,
            target=target_collection,
        )

        print("\n" + "=" * 60)
        print("Migration Complete!")
        print("=" * 60)
        print(f"Total points in source: {stats['total_points']}")
        print(f"Migrated points: {stats['migrated_points']}")
        print(f"Updated 'about' fields: {stats['updated_about_fields']}")

        if stats["errors"]:
            print(f"\nErrors ({len(stats['errors'])}):")
            for err in stats["errors"]:
                print(f"  - {err}")

        # Verify
        target_info = client.get_collection(target_collection)
        print(f"\nTarget collection now has: {target_info.points_count} points")

    except Exception as e:
        print(f"\nMigration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
