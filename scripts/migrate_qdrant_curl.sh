#!/bin/bash
# Qdrant Migration Script (curl-based)
# Migrates: anima_AnimaAgent -> anima_Anima
# Updates: about="xiao_guang" -> about="self"

set -e

# Load env
source "$(dirname "$0")/../.env" 2>/dev/null || true

QDRANT_URL="${QDRANT_URL:-https://anima-qdrant.zeabur.app}"
API_KEY="${QDRANT_API_KEY:-C85JM493wRbz01optZHTix7aseBv2V6g}"

SOURCE="anima_AnimaAgent"
TARGET="anima_Anima"
BATCH_SIZE=100

echo "=============================================="
echo "Qdrant Migration: $SOURCE -> $TARGET"
echo "=============================================="

# Get source collection info
echo -e "\n[1/4] Getting source collection info..."
SOURCE_INFO=$(curl -s -X GET "$QDRANT_URL/collections/$SOURCE" \
    -H "api-key: $API_KEY" \
    -H "Content-Type: application/json")

VECTOR_SIZE=$(echo "$SOURCE_INFO" | jq -r '.result.config.params.vectors.size')
DISTANCE=$(echo "$SOURCE_INFO" | jq -r '.result.config.params.vectors.distance')
POINTS_COUNT=$(echo "$SOURCE_INFO" | jq -r '.result.points_count')

echo "  Vector size: $VECTOR_SIZE"
echo "  Distance: $DISTANCE"
echo "  Points count: $POINTS_COUNT"

# Check if target exists
echo -e "\n[2/4] Checking target collection..."
TARGET_CHECK=$(curl -s -X GET "$QDRANT_URL/collections/$TARGET" -H "api-key: $API_KEY")
TARGET_EXISTS=$(echo "$TARGET_CHECK" | jq -r '.status')

if [ "$TARGET_EXISTS" != "ok" ]; then
    echo "  Creating target collection: $TARGET"
    curl -s -X PUT "$QDRANT_URL/collections/$TARGET" \
        -H "api-key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"vectors\": {
                \"size\": $VECTOR_SIZE,
                \"distance\": \"$DISTANCE\"
            }
        }" | jq -r '.status'
else
    EXISTING_POINTS=$(echo "$TARGET_CHECK" | jq -r '.result.points_count')
    echo "  Target exists with $EXISTING_POINTS points"
fi

# Migrate points in batches
echo -e "\n[3/4] Migrating points..."

OFFSET=""
BATCH_NUM=0
MIGRATED=0
UPDATED_ABOUT=0

while true; do
    BATCH_NUM=$((BATCH_NUM + 1))

    # Build scroll request
    if [ -z "$OFFSET" ]; then
        SCROLL_BODY="{\"limit\": $BATCH_SIZE, \"with_vector\": true, \"with_payload\": true}"
    else
        SCROLL_BODY="{\"limit\": $BATCH_SIZE, \"offset\": $OFFSET, \"with_vector\": true, \"with_payload\": true}"
    fi

    # Scroll points from source
    RESULT=$(curl -s -X POST "$QDRANT_URL/collections/$SOURCE/points/scroll" \
        -H "api-key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$SCROLL_BODY")

    POINTS=$(echo "$RESULT" | jq -r '.result.points')
    NEXT_OFFSET=$(echo "$RESULT" | jq -r '.result.next_page_offset // empty')
    POINTS_LEN=$(echo "$POINTS" | jq 'length')

    if [ "$POINTS_LEN" = "0" ] || [ "$POINTS_LEN" = "null" ]; then
        echo "  No more points to migrate"
        break
    fi

    echo "  Batch $BATCH_NUM: $POINTS_LEN points"

    # Transform points (update about field)
    TRANSFORMED=$(echo "$POINTS" | jq '
        map(
            if .payload.about == "xiao_guang" then
                .payload.about = "self"
            else
                .
            end
        )
    ')

    # Count updated about fields in this batch
    BATCH_UPDATED=$(echo "$POINTS" | jq '[.[] | select(.payload.about == "xiao_guang")] | length')
    UPDATED_ABOUT=$((UPDATED_ABOUT + BATCH_UPDATED))

    # Upsert to target
    UPSERT_BODY=$(echo "$TRANSFORMED" | jq '{points: .}')
    UPSERT_RESULT=$(curl -s -X PUT "$QDRANT_URL/collections/$TARGET/points" \
        -H "api-key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$UPSERT_BODY")

    UPSERT_STATUS=$(echo "$UPSERT_RESULT" | jq -r '.status')
    if [ "$UPSERT_STATUS" != "ok" ]; then
        echo "    Error: $UPSERT_RESULT"
    else
        MIGRATED=$((MIGRATED + POINTS_LEN))
    fi

    # Check for next page
    if [ -z "$NEXT_OFFSET" ]; then
        break
    fi
    OFFSET="$NEXT_OFFSET"
done

# Verify
echo -e "\n[4/4] Verifying migration..."
FINAL_INFO=$(curl -s -X GET "$QDRANT_URL/collections/$TARGET" -H "api-key: $API_KEY")
FINAL_COUNT=$(echo "$FINAL_INFO" | jq -r '.result.points_count')

echo ""
echo "=============================================="
echo "Migration Complete!"
echo "=============================================="
echo "Source points: $POINTS_COUNT"
echo "Migrated points: $MIGRATED"
echo "Updated 'about' fields: $UPDATED_ABOUT"
echo "Target collection now has: $FINAL_COUNT points"
