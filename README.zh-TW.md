# Anima 🤖

[English](README.md) | 繁體中文

具備持久記憶、人格一致性與身份識別的社群 AI Agent，支援 Threads/MCP 本地對話與模擬觀察模式。

## 特色

- **持久記憶**：Mem0 三層記憶（情節/語義/反思），並分離 agent/user scope，確保對話者與小光的內容不混淆
- **人格一致性**：借鏡 TinyTroupe 的 persona schema，產生/校驗回覆，支援 Adherence 評分與原因追蹤
- **反思機制**：借鏡 Generative Agents，定期生成高層次洞見
- **身份識別**：MCP 模式可「我是/我叫」或 `anima_set_user` 指定身份，記憶會標記 participant_xxx
- **平台無關**：CLI/daemon/MCP 模式並行，易於接到其他平台
- **可自訂人格**：透過 JSON 定義角色，支援 AI 簽名、emoji 控制等細節
- **觀察與報表**：模擬/真實運行皆有日誌，一頁報表追蹤品質與健康度

## 架構

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

## 快速開始

### 1. 安裝依賴

```bash
# 使用 pip
pip install -e .

# 或使用 poetry
poetry install
```

### 2. 設定環境變數

必填（可用 `.env`）：
- `OPENAI_API_KEY`
- `THREADS_ACCESS_TOKEN` / `THREADS_USER_ID`（使用 Threads 時）
- `PERSONA_FILE`（預設 `personas/default.json`）
- `QDRANT_URL` / `QDRANT_API_KEY`（雲端 Qdrant）或本地 `http://localhost:6333`

可選：
- `OPENAI_MODEL`（預設 `gpt-5-mini`，用於決策/驗證）
- `OPENAI_MODEL_ADVANCED`（預設 `gpt-5.1`，用於生成回覆）
- `REASONING_EFFORT`（預設 `low`，gpt-5 系列推理強度）
- `MAX_COMPLETION_TOKENS`（預設 `500`）
- `MAX_DAILY_POSTS` / `MAX_DAILY_REPLIES`（速率限制）
- `LOG_LEVEL`（預設 `INFO`）

### 3. 啟動本地服務（開發用，可選）

```bash
docker-compose up -d qdrant postgres
```

### 4. 運行 CLI/Daemon

```bash
# 執行一次互動循環
anima cycle

# 發一則貼文
anima post --topic "AI"

# 產生反思
anima reflect

# 查看統計
anima stats

# 以 daemon 模式運行（定時執行）
anima daemon

# 觀察模式（模擬但不發文）
anima observe --cycles 3

# 觀察模式 + Mock（測試開發用，不需要真實 API）
anima observe --mock --cycles 3

# 標註與分析
anima review             # 互動式標註
anima review --stats     # 查看統計
anima analyze            # 產生分析報告

# 一頁報表
anima report                              # 從觀察模式資料產生
anima report --source data/real_logs      # 從真實運行資料產生
anima report --html                       # 同時產生 HTML
anima report --days 14                    # 指定時間範圍
```

### 5. MCP（Claude Desktop 等）

1. 建立 `mcp-config.json`，套用專案絕對路徑（API key 等從專案 `.env` 讀取，不用在這裡塞）：
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
2. 在 MCP 客戶端載入。對話時可說「我是 Alex」或呼叫 `anima_set_user("Alex")` 讓記憶標記為 `participant_Alex`。

**可用 MCP 工具：**

| 工具 | 說明 |
|------|------|
| `anima_chat` | 與 Anima 對話（自動識別身份、記錄記憶） |
| `anima_set_user` | 設定當前對話者的名字 |
| `anima_search_memory` | 搜尋 Anima 的記憶 |
| `anima_add_memory` | 新增一則記憶 |
| `anima_get_recent_memories` | 取得最近的記憶 |
| `anima_reflect` | 讓 Anima 進行反思 |
| `anima_get_persona` | 取得人格資訊 |
| `anima_memory_stats` | 取得記憶統計 |

## Threads API 說明

### 權限需求

| 權限 | 用途 | 需要審核 |
|-----|------|---------|
| `threads_basic` | 基本讀取/發文 | 否 |
| `threads_keyword_search` | 搜尋公開貼文 | **是（需 Meta 審核）** |

### 互動模式

Anima 支援兩種互動模式：

1. **回覆模式（Reply Mode）** - 預設
   - 使用 `get_replies_to_my_posts()` 獲取自己貼文的回覆
   - 不需要特殊權限
   - 適合大多數使用場景

2. **搜尋模式（Search Mode）** - 備用
   - 使用 `search_posts()` 搜尋公開貼文
   - **需要 `threads_keyword_search` 權限**（通常需要 1-2 週審核）
   - 當回覆模式沒有內容時自動嘗試

### 速率限制

- 發佈貼文：250 篇/24 小時
- 回覆貼文：1,000 條/24 小時
- 搜尋查詢：2,200 次/24 小時（需要特殊權限）

## Mock 模式（測試開發用）

Mock 模式允許在**沒有真實 API Token** 的情況下測試整個系統：

```bash
# 觀察模式 + Mock
anima observe --mock --cycles 3

# 或通過環境變數啟用
USE_MOCK_THREADS=true anima cycle
```

**功能包含**：
- 預設模擬貼文資料庫（各種話題）
- 完整決策引擎測試
- 回應生成與 Persona 一致性驗證
- 記憶系統測試

**⚠️ 重要**：Mock 模式使用獨立的測試記憶庫（`anima_{agent}_test`），不會污染正式記憶。

**適用場景**：
- 本地開發和調試
- CI/CD 自動化測試
- 沒有 API 權限時的快速原型

## 日誌與報表

### 資料儲存

| 模式 | 資料目錄 | 說明 |
|-----|---------|------|
| 觀察模式 | `data/simulations/` | 模擬運行記錄 |
| 真實模式 | `data/real_logs/` | 實際發文記錄 |
| 報表輸出 | `data/reports/` | 一頁報表 |

### 報表內容

一頁報表包含：
- **Persona 摘要**：角色基本資訊
- **記憶庫概況**：記憶統計
- **決策與互動**：互動率、skip 理由分析
- **互動健康度**：發送成功/失敗統計
- **品質標註**：Adherence 分數與標註分布
- **問題診斷**：低分案例與評分原因
- **可操作建議**：自動生成的改進建議

## 自訂人格

編輯 `personas/default.json` 或建立新的人格檔案：

```json
{
  "identity": {
    "name": "你的角色名",
    "age": 25,
    "occupation": "職業",
    "background": "背景故事...",
    "signature": "— AI 代班"
  },
  "personality": {
    "traits": ["好奇", "幽默", "思考型"],
    "values": ["真實", "創意"],
    "communication_style": "輕鬆自然"
  },
  "speech_patterns": {
    "emoji_usage": "never",
    "typical_phrases": ["蠻有趣", "這個厲害"]
  },
  "interaction_rules": {
    "max_response_length": 280
  }
}
```

**重點欄位：**
- `signature`：回覆結尾簽名（如「— AI 代班」）
- `emoji_usage`：設為 `"never"` 可讓回覆更像真人
- `max_response_length`：限制回覆長度

詳細 schema 請參考 `src/agent/persona.py`。

## 記憶與身份設計（重點）

- 互動會拆成兩筆：`participant_*`（對話者內容，user scope）與 `agent_id`（小光回覆，agent scope），並複寫一份摘要到 agent scope，確保反思/統計可見對話者資訊。
- `search/get_recent/stats` 會合併 agent/user 記憶；MCP 身份由「我是/我叫」或 `anima_set_user` 決定。

## 主動分享素材池（快速堆料）

- 內建輕量抓取腳本：`python -m src.utils.harvest_ideas --limit 8`
  - 預設來源：OpenAI Blog、Hugging Face Blog、Papers with Code、Hacker News AI（官方/RSS，無 Anthropic 官方 RSS）
  - 會用 OpenAI 將素材轉成口語化中文短稿，輸出 Markdown + `data/ideas/index.jsonl`（含 pending/posted/expired 狀態與 Threads 貼文 ID）
  - 需要設定 `OPENAI_API_KEY`，可在 Zeabur 的 `/app/data` volume 中持久化輸出
  - 可自訂 feed：`--feeds https://example.com/rss ...`

## 海巡來源（Threads Toolkit）

Anima 使用 Threads Toolkit Apify Actor 來獲取外部貼文進行互動。

### 配置方式

```bash
# 配置環境變數
WEBHOOK_ENABLED=true
WEBHOOK_HOST=0.0.0.0
WEBHOOK_PORT=8080
WEBHOOK_SECRET=your_secret_token

APIFY_ENABLED=true
APIFY_API_TOKEN=your_apify_token

# 啟動 webhook 伺服器
anima webhook
```

### 運作流程

1. 在 Apify 上配置 Threads Toolkit Actor，設定搜尋關鍵字
2. 在 Actor 中設定 webhook，當執行完成時通知 Anima
3. Actor 執行完成後，推送結果到 `POST /webhooks/apify`
4. Anima 處理貼文並觸發互動循環

### Webhook 端點

- `POST /webhooks/apify` - 接收 Apify webhook 通知
- 在 Apify Actor 設定中配置 webhook URL：`http://your-server:8080/webhooks/apify`
- 如果設定了 `WEBHOOK_SECRET`，需在請求 header 加入：`Authorization: Bearer your_secret_token`

### Webhook Payload 範例

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

### 優勢

- 即時反應：Actor 執行完立即觸發互動
- 節省資源：不需要定期輪詢
- 準確：直接從 Apify dataset 抓取資料

## 排程（已內建）
- 互動循環：預設每 4 小時
- 素材抓取：每 4 小時
- 主動發文：每日 10:00 從 pending ideas 自動發一則（發佈前會做重複檢查，發後標記 posted）
- 素材過期：每日 03:00 將超過 7 天的 pending 標記 expired
- 反思：每日 23:00

## 授權

Apache License 2.0

詳見 [LICENSE](LICENSE) 文件。

## 致謝

- [Mem0](https://github.com/mem0ai/mem0) - 記憶系統
- [TinyTroupe](https://github.com/microsoft/TinyTroupe) - 人格框架概念
- [Generative Agents](https://github.com/joonspk-research/generative_agents) - 反思機制概念
