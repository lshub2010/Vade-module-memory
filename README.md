# memory module

Vade 的**过程记忆**：存得住（两层、去重、冲突不覆盖、超限淘汰、重启不丢）与查得回（语义召回）。
不是知识库（D37）——人工整理的沉淀知识走外置 MCP 知识库，这里只存模型与 evolver 产生的东西。

- 工具：`remember` / `recall` / `forget`
- 存储：SQLite WAL（`data/memory.db`）+ 内存向量索引
- Embedding：OpenAI 兼容 HTTP，或内置确定性 hashed bag-of-words（离线/自测）
- 私有 RPC：`memory/read_session` + 六个 `memory/admin_*`
- 契约：[`schema/memory.schema.json`](schema/memory.schema.json)（C-12）、[`schema/memory-admin.json`](schema/memory-admin.json)（C-16）

## 工具

| 工具 | 参数 | risk | timeout | 说明 |
|---|---|---|---|---|
| `remember` | `text, tags=[], scope="global", namespace="user:default"` | low | 15s | 内容哈希去重；显式写入 `importance=5`、`confidence=1` |
| `recall` | `query, k=5, namespace="", include_session=false, include_superseded=false` | low | 15s | 余弦相似度；默认只查 global；单次受 `recall_token_limit` 限制 |
| `forget` | `id` | **high** | 10s | 软删 + `archive(reason='forget')`；不存在返回 `NOT_FOUND` |

**`forget` 为什么是 high（与 design §15.2 的「memory 整体 low」偏离）**：删除不可逆，
宁可多一次人工审批。manifest 里的 risk 就是 high；这是本模块唯一一处比设计更保守的偏离。

约束：

- 单条 `text` ≤ 8 KiB（超过 `PAYLOAD_TOO_LARGE`）；`tags` ≤ 16 个、每个 ≤ 32 字；
  `k ∈ [1, 20]`（上限还受 `memory.recall_k_max` 约束）。
- `scope ∈ {global, session}`。`namespace` 只在 `global` 下有效；
  正则 `^[a-z0-9][a-z0-9:_-]{0,63}$`，默认 `user:default`。
- **工具参数里没有 `session_id`**。`scope="session"` 时由 core 通过 `ctx.session_id`
  注入；模型直接传 `session_id` 会被 manifest 的 `additionalProperties:false` 拒绝。

## 两层记忆（D40）

| | `session` | `global` |
|---|---|---|
| 分区 | `session_id`（core 注入，模型不可伪造） | `namespace` |
| 写入 | evolver（M4b）+ 显式 `remember(scope="session")` | `remember` + evolver 晋升 |
| 读取 | core 直接注入 prompt（`memory/read_session`），不走 `recall` | `recall` 检索 |
| 生命周期 | `session.closed` 后 +7 天 TTL | 永久，受 5000 条配额 |
| 上限 | ≤200 条 **且** ≤ 字符预算（双闸） | 5000 条，按 score 淘汰 |

**注入是按预算的，不是全量**（AMEND-19/D79）：`memory/read_session` 按
`importance × recency` 降序取条目，累计字符到 `memory.session_inject_char_limit`
即停。被裁掉的条目仍可用 `recall(include_session=true)` 捞到。

**写侧字符闸**：`memory.session_inject_char_limit` 同时是显式 `remember` 的写预算；
C-18 把 evolver 的写预算单独定义为 `evolve.session_char_limit`，若部署时提供了
`EVOLVE_SESSION_CHAR_LIMIT`，写侧用后者（0/未设 = 跟随注入预算）。超过字符闸或
200 条上限返回 **`MEMORY_FULL`**（可重试；evolver 当轮应合并/淘汰后重试），
与「真的满了」的 `OVERLOADED` 区分。

## 去重、冲突、淘汰

- `content_hash = sha256(layer + "\0" + session_id + "\0" + namespace + "\0" + normalized_text)`；
  `normalized_text` 只做 `strip()` + 连续空白折叠。v1 **不做语义去重**（语义合并是
  M4b consolidate 的事，且只合并不删原文）。
- 唯一键 `UNIQUE(layer, session_id, namespace, content_hash)`。**必须带 `session_id`**：
  否则 A/B 两个会话写同一句话会命中同一行，B 读到 A 的记忆，两个会话的配额也会合并。
  `session_id` 用 `NOT NULL DEFAULT ''` 而不是 NULL——SQLite 的 UNIQUE 认为 NULL 互不相等，
  global 层的去重会整个失效。
- 冲突不覆盖：`supersede(old, new)` 给旧条写 `superseded_by`、新条写 `supersedes`；
  默认 `recall` 不返回旧条，`include_superseded=true` 才返回完整链。
- 评分（D44）：

  ```
  score = 0.45*importance + 0.30*exp(-(now-created_at)/30d)
        + 0.20*log1p(hit_count) - 1.00*(superseded_by IS NOT NULL)
  ```

- 超 `global_capacity` 时先按 score 升序淘汰到 90% 水位，淘汰行写
  `archive(reason='quota')`，**不物理删除**。`source='user'` 没有豁免——用户记忆
  `importance=5`，实际很难被淘汰，但库不能只进不出。
- 物理删除只有两个来源：`forget`（`deleted_at` 后 7 天清理）与 session TTL 到期。
  TTL 删除**不写 archive**。

## 私有 RPC（不进工具表）

| 方法 | 入参 | 出参 |
|---|---|---|
| `memory/read_session` | `{session_id, limit}` | `{items:[{id,text,tags,created_at}]}` |
| `memory/admin_list` | `{layer?, namespace?, status?, from?, to?, q?, limit=50, cursor?}` | `{items:[...], next_cursor}` |
| `memory/admin_get` | `{id}` | `{item, chain:[{id,status,created_at}]}` |
| `memory/admin_promote` | `{id, by}` | `{ok, id}`，写 `promoted_by="admin:<name>"` |
| `memory/admin_archive` | `{id, reason, by}` | `{ok}`，软归档 |
| `memory/admin_restore` | `{id, by}` | `{ok}`，`archived → active` |
| `memory/admin_stats` | `{}` | `{global_count, global_limit, staging_count, archived_count, oldest_at}` |

三条硬约束：

1. **不做权限判断**：只有 core 能连模块（FR-5.7），角色在 core 侧判。
2. **不返回原文对话**：只有记忆正文与 `session_id` 指针，没有 `user_text` /
   `assistant_text`（§7.9 隐私边界）。
3. **没有编辑 RPC**：改内容 = `archive` 旧条 + `remember` 新条，`supersedes` 链保留溯源。

`admin_list` 的 `q` 是 SQLite `LIKE`（`text` 与 `tags`），**不走向量**：管理面要的是
“我大概记得有这句”，向量会让翻页不可预测且每页重算 embedding。

## 配置

权威定义在 design §7.9.1 / C-18。模块只读下列环境变量（名字映射到 `core.toml` 键；
默认值与 C-18 一致）：

| 环境变量 | `core.toml` 键 | 默认 |
|---|---|---|
| `MEMORY_GLOBAL_CAPACITY` | `memory.global_capacity` | 5000 |
| `MEMORY_SESSION_CAPACITY` | `memory.session_capacity` | 200 |
| `MEMORY_SESSION_INJECT_CHAR_LIMIT` | `memory.session_inject_char_limit` | 4200 |
| `EVOLVE_SESSION_CHAR_LIMIT` | `evolve.session_char_limit`（写侧；0=跟随注入） | 0 |
| `MEMORY_RECALL_K_MAX` | `memory.recall_k_max` | 20 |
| `MEMORY_RECALL_TOKEN_LIMIT` | `memory.recall_token_limit` | 800 |
| `MEMORY_TTL_DAYS` | `memory.ttl_days` | 7 |
| `MEMORY_EMBEDDING_MODEL` | `memory.embedding_model` | `""`（内置 `local-hash-v1`） |
| `MEMORY_EMBEDDING_DIM` | `memory.embedding_dim` | 1536 |
| `MEMORY_DATA_DIR` / `VADE_DATA_DIR` | 部署路径（非 C-18 键） | `data` |

Embedding 额外变量（模块自治，core 不代管）：

| 变量 | 说明 |
|---|---|
| `MEMORY_EMBEDDING_BASE_URL` | 设了就走 OpenAI 兼容 `POST {base_url}/embeddings`；不设走本地 hash |
| `MEMORY_API_KEY_ENV` | **环境变量名**，默认 `MEMORY_API_KEY`；模块不接收字面量密钥 |
| `MEMORY_EMBEDDING_TIMEOUT_S` | 默认 10s，仅 429/5xx 重试 1 次 |

维度/模型写进 `meta` 表。启动时与配置不一致 → **拒绝启动**，报错包含库里的
`dim/model` 与配置值；不会静默重算向量（那会写坏库）。换模型/维度必须重建数据库。

## 运行

```bash
cd Vade-module-memory
PYTHONPATH=src python -m memory_module --self-test          # 无 core、无网络、无密钥
PYTHONPATH=src python -m memory_module --check              # manifest 自检
PYTHONPATH=src python -m memory_module --print-manifest

PYTHONPATH=src python -m memory_module --standalone remember \
  --args '{"text":"用户偏好深色主题","tags":["pref"]}'
PYTHONPATH=src python -m memory_module --standalone recall \
  --args '{"query":"界面主题","k":3}'
PYTHONPATH=src python -m memory_module --standalone forget \
  --args '{"id":"<上一步的 id>"}'

# 私有 RPC 也可独立调试（core 不在时）
PYTHONPATH=src python -m memory_module --rpc memory/admin_stats --args '{}'
PYTHONPATH=src python -m memory_module --event session.closed \
  --args '{"session_id":"01..."}'
```

连 core：

```bash
PYTHONPATH=src python -m memory_module --core wss://core:8443
```

`--standalone` 的 `remember(scope="session")` 不会成功：没有 core 注入 `session_id`
时返回 `PERMISSION_DENIED`。这是有意的 fail-closed。

## 测试

```bash
cd Vade-module-memory
python -m pytest tests -q
```

覆盖：去重、两层隔离（两个会话同句两条）、重启后顺序/分数一致、supersedes、
session TTL、`session.closed` 从关闭时刻起算 7 天、`MEMORY_FULL` 双闸、配额淘汰到
archive、`read_session` 预算与 `recall` 捞回被裁条目、lesson evidence、管理面分页/
统计/晋升/归档/恢复、无原文对话字段、维度/模型不一致拒绝启动、embedding 失败无残留。

## Docker

```bash
docker build -t vade-memory .
docker run --rm -v vade-memory-data:/data \
  -e MEMORY_EMBEDDING_BASE_URL=https://llm.example/v1 \
  -e MEMORY_API_KEY_ENV=MEMORY_API_KEY \
  --env-file .env vade-memory
```

**`/data` 必须挂持久卷**：`data/memory.db`（含 `-wal` / `-shm`）是唯一状态；
不挂卷，容器重启后记忆全丢。镜像默认只装运行时依赖；core 模式需要把
`vade-sdk` 挂到 `/opt/vade-sdk`（与 echo 模块同一约定），不挂时 `--self-test`
与 `--standalone` 仍可运行。

## 存储接口与 M4b

`store.MemoryBackend` 是唯一接口，`store.SqliteBackend` 是 v1 唯一实现。
`tools.py` 与 `evolve/` 都不接触 `sqlite3.Connection`；换真向量库或远程共享服务时
只换 backend，不改工具层。

M4a 已建好/提供（M4b 只调用，不改 `store.py`）：

- `staging_put / staging_list / staging_promote / staging_drop`
- `supersede` / `archive_row`
- `get_last_consolidate()` / `record_consolidate(kind, stats)`（`consolidate_run` 水位）
- `MemoryRecord` / `StagingRecord` / `ArchiveRecord` 的字段见 C-12

`evolve/` 空包已预置，manifest 也已声明 `role="evolve"`、
`subscriptions=["session.turn","session.closed"]`、`injects=["session_id"]`。

### M4b 的模型连接（模块私有环境变量，不改 C-18）

四个模型触发器（`turn_review` / `session_close` / `consolidate` / `skill_evolve`）
需要一个真实模型。连接信息**不进** `[evolve]` 表——C-18 的 `[evolve]` 共 14 个键，
没有上游地址，往 `EvolveSettings` 加键会让 core 以「未知键」拒绝启动。这里照抄
embedder 的既有先例（`MEMORY_EMBEDDING_BASE_URL` 同样不在 `[memory]` schema 里）：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_EVOLVE_ENABLED` | `0` | 总开关；false ⇒ 不订阅、不写入、不调模型 |
| `MEMORY_EVOLVE_LLM_BASE_URL` | 空 | OpenAI 兼容地址，写到 `/v1` 这一级。空 ⇒ 纯规则 |
| `MEMORY_EVOLVE_LLM_API_KEY_ENV` | `MEMORY_API_KEY` | **变量名**，不是密钥本身；每次调用重读 |
| `MEMORY_EVOLVE_LLM_MODEL` | 空 | provider 的模型 id；空则退回 `evolve.model`（`"cheap"` 标签），**并打警告** |
| `MEMORY_EVOLVE_LLM_TIMEOUT_S` | `30` | 单次请求 socket 超时（0.1–120） |

只配 `MEMORY_EVOLVE_ENABLED=1` 而不配 `MEMORY_EVOLVE_LLM_BASE_URL` 时，模块正常启动、
打一行 `evolve enabled but MEMORY_EVOLVE_LLM_BASE_URL is unset` 警告，然后按纯规则跑。
超时/重试/降级语义全部在 `evolve/llm.py` 的 `EvolveLLM` 里，`evolve/openai_compat.py`
只负责「发一次非流式 POST，拿文本或抛错」。

## 已知限制

1. **SDK 0.1 的私有 RPC/ctx 钩子缺口**：`vade_sdk.transport` 只分发
   `tools/call`、`tools/cancel`、`module/drain`、`prompts/get`、`events/deliver`，
   不会把 `memory/*` 方法交给模块，也不把 `params.ctx.session_id` 传给工具函数。
   本模块已完整实现 `RpcDispatcher`、`handle_event`、`handle_tools_call`
   （读 `ctx.session_id`），并全部有测试；等 SDK 增加一个 `on_rpc`/`ctx` 钩子即可接线。
   在那之前，core 的 `read_session`/管理面无法端到端触达本模块——这是 SDK 缺口，
   不是 dispatcher 缺口。
2. **单实例设计**：`concurrency="exclusive"`，全量向量读进内存（20,000 条上限，
   1536 维约 120 MB 的 float32 `array`；有 numpy 时矩阵化加速）。多 core 共享不要
   直接复用，需要远程 backend。
3. **本地 hash embedding 是词法的，不是语义的**：生产应配
   `MEMORY_EMBEDDING_BASE_URL`；本地模式只保证离线可跑、确定性、可测。
4. **`OVERLOADED` 在 v1 很难触发**：淘汰总能腾出一个槽；只有归档失败/库不可写时
   才会返回。正常配额路径是“淘汰 + 写入”，不会静默丢新数据。
5. **hit_count 同步批量更新**：当前在一次 SQL `executemany` 内完成，不阻塞网络；
   没有做异步队列。语义正确，只是少了“后台合并”的延迟收益。
6. **没有 `admin_update`**：这是契约要求，不是遗漏；修改 = archive + remember。

## 安全边界

- 密钥只以**环境变量名**出现（`MEMORY_API_KEY_ENV`），代码/配置/仓库里没有字面量。
- 日志只允许 `text_digest` 与长度；`tools.py`、`store.py`、`embed.py` 均不打印正文。
- 错误信息最多 300 字，不回显正文、不回显 API key、不回显绝对路径之外的密钥材料。
- `forget` 是 high risk；`admin_*` 不做权限判断，但只有 core 能连（FR-5.7）。
