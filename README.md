# Anima 🤖

English | [繁體中文](README.zh-TW.md)

A social AI agent with persistent memory, persona consistency, and identity recognition, supporting Threads/MCP local conversations and simulation observation mode.

## Features

- **Persistent Memory**: Mem0 three-layer memory (episodic/semantic/reflective) with agent/user scope separation to prevent memory contamination
- **Persona Consistency**: Inspired by Microsoft TinyTroupe's persona schema, generates and validates responses with Adherence scoring and reason tracking
- **Reflection Mechanism**: Inspired by Stanford Generative Agents, periodically generates high-level insights
- **Identity Recognition**: MCP mode supports "I am/My name is" or `anima_set_user` to specify identity, with memory tagged as participant_xxx
- **Platform Agnostic**: CLI/daemon/MCP/Webhook modes work in parallel, easy to integrate with other platforms
- **Customizable Persona**: Define characters through JSON, supports AI signatures, emoji control, and more
- **Observation & Reporting**: Both simulation and real runs have logs, with one-page reports tracking quality and health

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Anima Agent                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │
│  │  Scheduler  │───│   Brain     │───│  Platform   │            │
│  │  (Cron)     │   │  (Decision) │   │  Adapter    │            │
│  └─────────────┘   └──────┬──────┘   └─────────────┘            │
│         ┌─────────────────┼─────────────────┐                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │
│  │   Persona   │   │   Memory    │   │  Reflection │            │
│  │   Engine    │   │   (Mem0)    │   │   Engine    │            │
│  └─────────────┘   └─────────────┘   └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
# Using pip
pip install -e .

# Or using poetry
poetry install
```

### 2. Configure Environment Variables

Required (can use `.env`):
- `OPENAI_API_KEY`
- `THREADS_ACCESS_TOKEN` / `THREADS_USER_ID` (when using Threads)
- `PERSONA_FILE` (default: `personas/default.json`)
- `QDRANT_URL` / `QDRANT_API_KEY` (cloud Qdrant) or local `http://localhost:6333`

Optional:
- `OPENAI_MODEL` (default: `gpt-5-mini`, for decision/validation)
- `OPENAI_MODEL_ADVANCED` (default: `gpt-5.1`, for response generation)
- `REASONING_EFFORT` (default: `low`, reasoning intensity for gpt-5 series)
- `MAX_COMPLETION_TOKENS` (default: `500`)
- `MAX_DAILY_POSTS` / `MAX_DAILY_REPLIES` (rate limits)
- `LOG_LEVEL` (default: `INFO`)

### 3. Start Local Services (Optional, for Development)

```bash
docker-compose up -d qdrant postgres
```

### 4. Run CLI/Daemon

```bash
# Execute one interaction cycle
anima cycle

# Post a message
anima post --topic "AI"

# Generate reflection
anima reflect

# View statistics
anima stats

# Run in daemon mode (scheduled execution)
anima daemon

# Observation mode (simulate without posting)
anima observe --cycles 3

# Observation mode + Mock (for testing, no real API needed)
anima observe --mock --cycles 3

# Annotation and analysis
anima review             # Interactive annotation
anima review --stats     # View statistics
anima analyze            # Generate analysis report

# One-page report
anima report                              # Generate from observation mode data
anima report --source data/real_logs      # Generate from real run data
anima report --html                       # Generate HTML version too
anima report --days 14                    # Specify time range
```

### 5. MCP (Claude Desktop, etc.)

1. Create `mcp-config.json` with absolute project path (API keys are read from project `.env`):
   ```json
   {
     "mcpServers": {
       "anima": {
         "command": "/path/to/anima/.venv/bin/python",
         "args": ["-m", "src.mcp"],
         "env": {
           "PYTHONPATH": "/path/to/anima"
         }
       }
     }
   }
   ```
2. Load in MCP client. During conversation, say "I am Alex" or call `anima_set_user("Alex")` to tag memory as `participant_Alex`.

**Available MCP Tools:**

| Tool                     | Description                                                   |
|--------------------------|---------------------------------------------------------------|
| `anima_chat`             | Chat with Anima (auto identity recognition, memory recording) |
| `anima_set_user`         | Set current conversation participant name                     |
| `anima_search_memory`    | Search Anima's memory                                         |
| `anima_add_memory`       | Add a memory entry                                            |
| `anima_get_recent_memories` | Get recent memories                                        |
| `anima_reflect`          | Trigger Anima reflection                                      |
| `anima_get_persona`      | Get persona information                                       |
| `anima_memory_stats`     | Get memory statistics                                         |

## Threads API Guide

### Permission Requirements

| Permission               | Purpose             | Requires Review                |
|--------------------------|---------------------|--------------------------------|
| `threads_basic`          | Basic read/post     | No                             |
| `threads_keyword_search` | Search public posts | **Yes (Meta review required)** |

### Interaction Modes

Anima supports two interaction modes:

1. **Reply Mode (Default)**
   - Uses `get_replies_to_my_posts()` to fetch replies to own posts
   - No special permissions needed
   - Suitable for most use cases

2. **Search Mode (Fallback)**
   - Uses `search_posts()` to search public posts
   - **Requires `threads_keyword_search` permission** (typically 1-2 weeks review)
   - Automatically attempts when reply mode has no content

### Rate Limits

- Post creation: 250 posts/24 hours
- Post replies: 1,000 replies/24 hours
- Search queries: 2,200 queries/24 hours (requires special permission)

## Mock Mode (For Testing & Development)

Mock mode allows testing the entire system **without real API tokens**:

```bash
# Observation mode + Mock
anima observe --mock --cycles 3

# Or enable via environment variable
USE_MOCK_THREADS=true anima cycle
```

**Features**:
- Pre-built mock post database (various topics)
- Full decision engine testing
- Response generation with persona consistency validation
- Memory system testing

**⚠️ Important**: Mock mode uses a separate test memory collection (`anima_{agent}_test`) to avoid polluting production memory.

**Use Cases**:
- Local development and debugging
- CI/CD automated testing
- Quick prototyping without API permissions

## Logging & Reporting

### Data Storage

| Mode             | Data Directory       | Description             |
|------------------|----------------------|-------------------------|
| Observation mode | `data/simulations/`  | Simulation run records  |
| Real mode        | `data/real_logs/`    | Actual posting records  |
| Report output    | `data/reports/`      | One-page reports        |

### Report Content

One-page reports include:
- **Persona Summary**: Character basic information
- **Memory Overview**: Memory statistics
- **Decisions & Interactions**: Interaction rate, skip reason analysis
- **Interaction Health**: Send success/failure statistics
- **Quality Annotation**: Adherence scores and annotation distribution
- **Problem Diagnosis**: Low-score cases and scoring reasons
- **Actionable Suggestions**: Auto-generated improvement recommendations

## Customize Persona

Edit `personas/default.json` or create a new persona file:

```json
{
  "identity": {
    "name": "Your Character Name",
    "age": 25,
    "occupation": "Occupation",
    "background": "Background story...",
    "signature": "— AI Assistant"
  },
  "personality": {
    "traits": ["curious", "humorous", "thoughtful"],
    "values": ["authenticity", "creativity"],
    "communication_style": "casual and natural"
  },
  "speech_patterns": {
    "emoji_usage": "never",
    "typical_phrases": ["that's interesting", "pretty cool"]
  },
  "interaction_rules": {
    "max_response_length": 280
  }
}
```

**Key Fields:**
- `signature`: Reply ending signature (e.g., "— AI Assistant")
- `emoji_usage`: Set to `"never"` for more human-like responses
- `max_response_length`: Limit response length

For detailed schema, see `src/agent/persona.py`.

## Memory & Identity Design (Key Concept)

- Interactions are split into two entries: `participant_*` (conversation partner content, user scope) and `agent_id` (Agent response, agent scope), with a summary copy to agent scope to ensure reflection/statistics can see conversation partner information.
- `search/get_recent/stats` merge agent/user memories; MCP identity is determined by "I am/My name is" or `anima_set_user`.

## Content Pool for Active Sharing (Quick Content)

- Built-in lightweight harvesting script: `python -m src.utils.harvest_ideas --limit 8`
  - Default sources: OpenAI Blog, Hugging Face Blog, Papers with Code, Hacker News AI (official/RSS, no Anthropic official RSS)
  - Uses OpenAI to convert materials into colloquial Chinese drafts, outputs Markdown + `data/ideas/index.jsonl` (with pending/posted/expired status and Threads post ID)
  - Requires `OPENAI_API_KEY`, can persist output in deployment environment's `/app/data` volume
  - Custom feeds: `--feeds https://example.com/rss ...`

## External Content Sources (Threads Toolkit)

Anima uses the Threads Toolkit Apify Actor to fetch external posts for interaction.

### Configuration

```bash
# Configure environment variables
WEBHOOK_ENABLED=true
WEBHOOK_HOST=0.0.0.0
WEBHOOK_PORT=8080
WEBHOOK_SECRET=your_secret_token

APIFY_ENABLED=true
APIFY_API_TOKEN=your_apify_token

# Start webhook server
anima webhook
```

### How It Works

1. Configure a Threads Toolkit Actor on Apify with your search keywords
2. Set up a webhook in the Actor to notify your Anima instance when runs complete
3. When the Actor finishes, it pushes results to `POST /webhooks/apify`
4. Anima processes the posts and triggers interaction cycles

### Webhook Endpoint

- `POST /webhooks/apify` - Receive Apify webhook notifications
- Configure webhook URL in Apify Actor settings: `http://your-server:8080/webhooks/apify`
- If `WEBHOOK_SECRET` is set, add to request header: `Authorization: Bearer your_secret_token`

### Webhook Payload Example

```json
{
  "eventType": "ACTOR.RUN.SUCCEEDED",
  "resource": {
    "id": "run_id",
    "actId": "actor_id",
    "defaultDatasetId": "dataset_id"
  }
}
```

### Advantages

- Real-time response: Triggers interaction immediately after Actor execution
- Resource-efficient: No periodic polling needed
- Accurate: Direct dataset retrieval from Apify

## Scheduling (Built-in)
- Interaction cycle: Every 4 hours by default
- Content harvesting: Every 4 hours
- Active posting: Daily at 10:00, auto-posts one item from pending ideas (checks for duplicates, marks as posted)
- Content expiration: Daily at 03:00, marks pending ideas older than 7 days as expired
- Reflection: Daily at 23:00

## License

Apache License 2.0

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Mem0](https://github.com/mem0ai/mem0) - Memory system
- [Microsoft TinyTroupe](https://github.com/microsoft/TinyTroupe) - Persona framework concept
- [Stanford Generative Agents](https://github.com/joonspk-research/generative_agents) - Reflection mechanism concept
