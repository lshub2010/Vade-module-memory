# evolve/ — M4b 的目录

M4a 只预置这个包，**不写实现**。M4b 在 W6 填：

- 订阅 `session.turn` / `session.closed`（manifest 已声明）
- 规则抽取 → `session` 层
- 每 N 轮复盘 / 会话结束总结 → `staging`
- 每日 consolidate（`consolidate_run` 幂等水位）

## 可用接口（只能走这个，不能碰 SQLite）

`memory_module.store.MemoryBackend`：

| 方法 | 用途 |
|---|---|
| `staging_put(text, namespace, session_id, embedding, confidence, kind=..., evidence=...)` | 写候选；同一 `(kind, namespace, content_hash)` 合并 `seen_sessions` |
| `staging_list(namespace=None)` | 读候选（含 embedding，consolidate 聚类用） |
| `staging_promote(staging_id, by="evolve")` | 晋升到 global；admin 通道传 `by="admin:<name>"` |
| `staging_drop(staging_id)` | 丢弃候选 |
| `put(rec)` | 直接写记忆（`rec.embedding` 必填） |
| `supersede(old_id, new_id)` | 建立替代链，两条都保留 |
| `archive_row(mem_id, reason)` | 软归档，可恢复 |
| `get_last_consolidate()` / `record_consolidate(kind, stats)` | 幂等水位 |

`MemoryBackend` 之外不要 import `store.py` 的私有函数；`_conn` 永远不外借。
