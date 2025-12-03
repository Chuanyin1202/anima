# Anima 專案說明（給 AI Agent）

## 專案概述

Anima 是一個具備持久記憶和人格一致性的社群 AI Agent，主要運行於 Threads 平台。

## 重要注意事項

### Mock 模式警告

**Mock 模式僅用於開發測試，不要在正式操作中使用！**

```bash
# 開發測試用（使用測試記憶庫）
anima observe --mock --cycles 3

# 正式操作（使用真實 API 和正式記憶庫）
anima observe --cycles 1
anima cycle
```

Mock 模式的特性：
- 使用獨立的測試記憶庫（`anima_{agent}_test`）
- 不會實際發送任何內容到 Threads
- 適用於本地開發、CI/CD 測試、功能驗證

### Threads API 權限

| 功能 | 方法 | 權限需求 |
|-----|------|---------|
| 獲取回覆 | `get_replies_to_my_posts()` | `threads_basic`（預設） |
| 搜尋貼文 | `search_posts()` | `threads_keyword_search`（需審核） |

**建議**：優先使用 `get_replies_to_my_posts()`，不需要額外權限。

### 記憶系統

- 正式記憶庫：`anima_{agent_name}`
- 測試記憶庫：`anima_{agent_name}_test`

**查詢記憶統計**：
```bash
anima stats
```

## 常用命令

```bash
# 執行一次互動循環（真實 API）
anima cycle

# 觀察模式（不實際發文，但使用真實 API）
anima observe --cycles 1

# 觀察模式 + Mock（測試用）
anima observe --mock --cycles 3

# 查看記憶統計
anima stats

# 產生反思
anima reflect

# 發一則貼文
anima post --topic "AI"
```

## 檔案結構

```
src/
├── agent/              # Agent 核心邏輯
│   ├── brain.py        # 決策引擎（互動流程控制）
│   ├── persona.py      # 人格系統（回應生成、一致性驗證）
│   └── scheduler.py    # 排程器（定時任務）
├── memory/             # 記憶系統
│   ├── mem0_adapter.py # Mem0 整合
│   └── reflection.py   # 反思引擎
├── threads/            # Threads API
│   ├── client.py       # 真實 API 客戶端
│   ├── mock_client.py  # 測試用模擬客戶端
│   └── models.py       # 資料模型
├── mcp/                # MCP 服務
└── main.py             # CLI 入口
```

## 關鍵類別

### AgentBrain（`src/agent/brain.py`）
決策引擎，控制整個互動流程：
- `run_cycle()` - 執行一次互動循環
- `_fetch_interesting_posts()` - 獲取要互動的貼文（優先回覆模式）
- `_interact_with_post()` - 處理單個互動

### ThreadsClient（`src/threads/client.py`）
Threads API 客戶端：
- `get_replies_to_my_posts()` - 獲取自己貼文的回覆（不需特殊權限）
- `search_posts()` - 搜尋公開貼文（需要 `threads_keyword_search` 權限）
- `reply_to_post()` - 回覆貼文
- `create_post()` - 發布新貼文

### AgentMemory（`src/memory/mem0_adapter.py`）
記憶系統：
- `observe()` - 記錄觀察
- `record_interaction()` - 記錄互動
- `search()` - 搜尋記憶
- `get_context_for_response()` - 獲取回應上下文
