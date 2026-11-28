# M4b 最后一公里：真实 LLM 调用接线

分支 `tmp/m4b-llm`，基线 `main` @ `56e74e3`。

## 1. 改了什么

| 文件 | 变更 |
|---|---|
| `src/memory_module/evolve/openai_compat.py` | **新增**。`OpenAICompatLLM`：`async def __call__(prompt) -> str`，一次非流式 `POST {base_url}/chat/completions`，错误一律上抛。 |
| `src/memory_module/settings.py` | 加 4 个模块私有字段 `evolve_llm_base_url` / `evolve_llm_api_key_env` / `evolve_llm_timeout_s` / `evolve_llm_model`，`from_env` 读 `MEMORY_EVOLVE_LLM_*`，`__post_init__` 校验；加 `effective_evolve_model()`。 |
| `src/memory_module/server.py` | 新增 `build_evolve_llm(settings, evolve)`；`build_app()` 改为 `build_service(settings, evolve=..., llm=build_evolve_llm(...))`。加模块级 `_log`。 |
| `src/memory_module/evolve/config.py` | 只加 docstring：说明连接配置为何不在这里。**`_KNOWN_KEYS` 一个没动。** |
| `README.md` | 新增「M4b 的模型连接」小节 + 环境变量表。 |
| `tests/test_evolve_llm_wire.py` | **新增**，13 个测试。 |
| `tests/test_evolve_config.py` | 加 4 个测试（键集守卫、env 默认值、env 校验）。 |
| `scripts/upstream_stub.py` | **新增**，可独立运行的 stdlib 假上游，用于手工端到端验证。 |

未触碰（`git diff main` 为空）：`evolve/llm.py`、`store.py`、`manifest.json`、`schema/`。

## 2. 配置项设计理由

### 为什么**不**往 `EvolveSettings` 加 `base_url` / `api_key_env`

任务书最初要求在 `EvolveSettings` 加这两个键，**这会违反项目契约，已改为环境变量**。依据是代码而非文档措辞：

1. `EvolveSettings.from_mapping` 对未知键**直接抛错**（`config.py:205-209`），而 core 拿 `schema/config.schema.json` 校验 `[evolve]`。实测该表共 **14 个键**，无 `base_url`、无 `api_key_env`。加键 ⇒ core 以「未知键」拒绝启动。
2. `docs/agents/46-M4b-evolve.md` §九 原话：「加键先改 C-18，不要在 config.py 里自己加」。
3. 测试 `test_evolve_settings_still_rejects_connection_keys` 与 `test_evolve_settings_key_set_matches_c18_exactly` 把这条约束钉死，防止以后有人「顺手」加回去。

### memory 模块现有的 embedder 连接配置是什么机制？（报告必须回答）

**是模块私有环境变量，完全不走 C-18。** 证据：

- `MemorySettings.embedding_base_url`（`settings.py:73`）、`api_key_env`（:74）、`embedding_timeout_s`（:75）；
- `from_env` 里读 `MEMORY_EMBEDDING_BASE_URL`（:150）、`MEMORY_API_KEY_ENV`（:132）、`MEMORY_EMBEDDING_TIMEOUT_S`（:152）；
- 消费点 `build_embedder()`（`server.py:76-85`）：配了 base_url ⇒ `OpenAIEmbeddingClient`，没配 ⇒ `LocalHashEmbedding`。

关键佐证：我实测 `schema/config.schema.json` 的 `[memory]` 段**只有 8 个键**，`embedding_base_url` / `api_key_env` **都不在其中**。也就是说 embedder 从第一天起就是「模块私有环境变量」模式，C-18 只管 core 下发给模块的策略，不管模块自己往哪发 HTTP。

**结论：选择跟随 embedder 先例，不另起一套。** 命名也照抄（`<域>_base_url` / `api_key_env` / `timeout_s`），加 `evolve_` 前缀避免与 embedding 的 `MEMORY_API_KEY_ENV` 撞车——两件事共用一个 key 是常见的部署错误，分开才能让 evolve 用便宜的 provider、embedding 用另一个。

### 为什么密钥只存环境变量名

`api_key_env` 存的是**变量名**，值每次调用时从 `os.environ` 重读（`openai_compat.py:_key()`）：

- 密钥不落 `MemorySettings`、不落日志、不落 manifest、不落数据库 —— 测试 `test_session_close_reaches_a_real_upstream` 断言 `SECRET not in json.dumps(memory.__dict__)`；
- 轮换密钥不需要重启；
- 与 embedder 的既有约定一致，运维只需学一次。

### 四个变量

| 变量 | 默认 | 理由 |
|---|---|---|
| `MEMORY_EVOLVE_LLM_BASE_URL` | `""` | 空 ⇒ `llm=None` ⇒ 纯规则，与今天完全一致。写到 `/v1`，代码拼 `+ "/chat/completions"`（对齐 core `core/vade/llm/openai_compat.py:147`，但**不 import core**）。 |
| `MEMORY_EVOLVE_LLM_API_KEY_ENV` | `MEMORY_API_KEY` | 只存名字。 |
| `MEMORY_EVOLVE_LLM_TIMEOUT_S` | `30` | 与 `EvolveLLM.DEFAULT_TIMEOUT_S` 同值。内层 socket 超时必须 ≤ 外层 `asyncio.wait_for`，否则外层先触发、内层线程还挂着。 |
| `MEMORY_EVOLVE_LLM_MODEL` | `""` | provider 的模型 id。空则退回 `evolve.model` 标签。 |

### 为什么用标准库而不是新增依赖

`pyproject.toml` 的依赖只有 `vade-sdk`，**没有 httpx / aiohttp**。所以用 `urllib.request` + `asyncio.to_thread`：零新增依赖，不把模块的安装体积和供应链为一个 POST 拖大。`urllib` 的 socket 超时在线程里生效，外层 `asyncio.wait_for` 仍能按时放弃——`test_upstream_timeout_degrades_without_raising` 验证了这条路径真的返回 `reason=timeout`。

### 职责边界（没动任何现有语义）

- `evolve/llm.py` **零改动**：严格 JSON、失败重试一次、30s 上限、`evolve_degraded` 全部原样。
- `openai_compat.py` 只做「发请求拿文本」，**不做重试、不做降级**，错误上抛给 `EvolveLLM` 决定。它故意不把 `TimeoutError` 包成自己的异常类型，因为 `EvolveLLM` 有专门的 `reason=timeout` 分支，包一层会退化成 `reason=LlmRequestError`。

## 3. 测试原始输出

### 3.1 新增的 13 个接线测试

```
$ PYTHONPATH=D:/HermesWorkspace/Vade/Vade/sdk/python \
  D:/HermesWorkspace/Vade/Vade/.venv/Scripts/python.exe -m pytest tests/test_evolve_llm_wire.py -v
```
```
platform win32 -- Python 3.12.14, pytest-9.2.0, pluggy-1.7.0
collected 13 items

test_session_close_reaches_a_real_upstream PASSED                     [  7%]
test_default_base_url_is_empty_so_nothing_is_called PASSED            [ 15%]
test_enabled_without_base_url_warns_once_and_degrades PASSED          [ 23%]
test_disabled_evolve_builds_no_client_even_with_a_base_url PASSED     [ 30%]
test_upstream_500_degrades_without_reaching_the_session PASSED        [ 38%]
test_upstream_timeout_degrades_without_raising PASSED                 [ 46%]
test_missing_api_key_env_degrades_instead_of_sending PASSED           [ 53%]
test_endpoint_joins_base_url_and_chat_completions PASSED              [ 61%]
test_client_raises_on_a_non_json_body PASSED                          [ 69%]
test_client_rejects_a_bad_base_url_at_construction PASSED             [ 76%]
test_bad_base_url_at_startup_degrades_instead_of_failing_to_boot PASSED [ 84%]
test_build_app_hands_the_real_client_to_the_service PASSED            [ 92%]
test_build_app_without_a_base_url_keeps_llm_none PASSED               [100%]

============================= 13 passed in 2.70s ==============================
EXIT=0
```

三条必写测试对应关系：

1. **真调通的 e2e** = `test_session_close_reaches_a_real_upstream`。stdlib `ThreadingHTTPServer` 假上游；断言 `path == "/v1/chat/completions"`、`Authorization == "Bearer <env 值>"`、`model/stream/temperature`、`outcome["status"] == "ok"`，以及 summary 真的落进了 session 层。
2. **没配 base_url 不崩** = `test_enabled_without_base_url_warns_once_and_degrades`（+ `test_build_app_without_a_base_url_keeps_llm_none`）。断言 `client is None`、恰好一条含 `MEMORY_EVOLVE_LLM_BASE_URL` 的 warning、pipeline 实际触发时返回 `error == "no_llm"` 且不抛。
3. **上游 500 / 超时** = `test_upstream_500_degrades_without_reaching_the_session`、`test_upstream_timeout_degrades_without_raising`。断言 `status == "degraded"`、`error` 分别为 `exception` / `timeout`、日志有 `evolve_degraded`、`staging_list() == []`（没有半截写入）。

### 3.2 全量

```
$ ... -m pytest tests -q
```
```
..................F..................................................... [ 37%]
........................................................................ [ 75%]
................................................                         [100%]
================================== FAILURES ===================================
__________ test_c12_validates_a_session_record_with_empty_namespace ___________
    def test_c12_validates_a_session_record_with_empty_namespace():
>       import jsonschema
E       ModuleNotFoundError: No module named 'jsonschema'
tests\test_contracts.py:65: ModuleNotFoundError
=========================== short test summary info ============================
FAILED tests/test_contracts.py::test_c12_validates_a_session_record_with_empty_namespace
1 failed, 191 passed in 5.57s
EXIT=1
```

**这 1 个失败是环境缺失，不是本次改动引入的。** 在干净的 `main`（未打任何补丁）上同样失败：

```
$ cd D:/HermesWorkspace/Vade/Vade-module-memory && ... -m pytest tests/test_contracts.py -q
FAILED tests/test_contracts.py::test_c12_validates_a_session_record_with_empty_namespace
1 failed, 4 passed in 0.14s
```

该 venv 里没有 `pip`（`No module named pip`），无法就地安装；按纪律不删测试、不 skip、不弱化断言，故保留失败原样记录。排除这一项后：

```
$ ... -m pytest tests -q --deselect tests/test_contracts.py::test_c12_validates_a_session_record_with_empty_namespace
191 passed, 1 deselected in 4.97s
EXIT=0
```

### 3.3 真实 `python -m memory_module` 路径（不是测试里的桩）

用 `scripts/upstream_stub.py` 起独立进程，`MEMORY_EVOLVE_LLM_BASE_URL=http://127.0.0.1:18777/v1`，走 `build_app()`：

```
INFO memory_module evolve llm configured endpoint=http://127.0.0.1:18777/v1/chat/completions model=e2e-cheap key_env=E2E_KEY
INFO memory_module.evolve evolve_skill_skipped reason=too_few_tool_calls
--- handle_event_async session.closed ---
{
 "ok": true,
 "handled": true,
 "session_id": "01M8GZQZR56MA6KMER64K9Z8QM",
 "updated": 0,
 "evolve": {
  "status": "ok",
  "turns": 4,
  "summary_chars": 18,
  "facts": 1,
  "lessons": 0,
  "dropped": 0,
  "retried": false,
  "error": null,
  ...
 }
}
--- session layer ---
 * 端到端检查：会话确认部署走本地回环。
```

假上游实际收到的请求：

```
requests captured: 1
path          : /v1/chat/completions
authorization : Bearer sk-e2e-<redacted>-env
content-type  : application/json
model/stream  : e2e-cheap False 0
roles         : ['user']
prompt head   : 你是记忆系统的会话总结器。根据 rolling summary、最近若干轮（文本已裁剪）和 skill 目录快照，产出
```

`--check` 与 `--self-test` 均 `EXIT=0`，manifest 未变。

## 4. `enabled=false` 行为完全未变的论证

按「代码路径」逐条论证，不是靠印象：

**4.1 不构造客户端。** `build_evolve_llm` 第一件事就是 `if not enabled: return None`，**在读取 `base_url` 之前**。所以即使运维配了 `MEMORY_EVOLVE_LLM_BASE_URL` 而没开 `enabled`，也不会建出 client、不会打日志、不会读密钥。`test_disabled_evolve_builds_no_client_even_with_a_base_url` 断言了这一点，并且断言 `caplog.records == []`（一条日志都没有）。

**4.2 `MemoryService` 的 if 分支没动。** `server.py` 里仍是 `if evolve is not None and evolve.enabled:` 才建 `EvolvePipeline`/`EvolveWorker`。`enabled=false` ⇒ `worker=None` ⇒ `handle_event_async` 里 `if worker is None or not worker.enabled: return result`，返回的就是纯 M4a 的 `dispatch_event` 结果。真实运行验证：

```
enabled       : False
llm client    : None
worker        : None
event result  : {"ok": true, "handled": true, "session_id": "01M8GZRP1S2S03F433V0QRSTGN", "updated": 0}
session rows  : 0
staging rows  : 0
--- requests seen by upstream (should still be 1) ---
1        ← 没有新增请求（前面 enabled=true 那次留下的就是这 1 条）
```

**4.3 `[evolve]` 配置面零变化。** `_KNOWN_KEYS` 仍是原 14 个键，`from_mapping` / `from_env` / `__post_init__` 一行没改（`config.py` 的 diff 只有 docstring）。`test_evolve_settings_key_set_matches_c18_exactly` 断言键集与 C-18 逐字相等。

**4.4 新增字段默认值都是「关」。** `evolve_llm_base_url=""`、`evolve_llm_timeout_s=30.0`、`evolve_llm_model=""`，`test_evolve_connection_env_vars_default_to_off` 断言 `MemorySettings.from_env({})` 全部取默认。空 base_url 是唯一的启用条件。

**4.5 降级逻辑未改。** `evolve/llm.py` 的 `git diff main` 为空。`EvolveLLM` 的严格 JSON、单次重试、30s 上限、`evolve_degraded` 全部原样；`enabled=false` 时 pipeline 根本不会被构造，所以连 `EvolveLLM` 都不会走到。

**4.6 既有测试未改。** `tests/` 下原有 8 个 evolve 测试文件与 `test_persistence.py` / `test_admin_rpc.py` 等全部通过（191 passed），没有任何断言被放宽。

## 5. 明确回答：`evolve.model`（那个 `"cheap"` 标签）现在是什么语义？

**它是给日志/溯源用的成本标签，不是发给模型的 model id —— 但在没配 `MEMORY_EVOLVE_LLM_MODEL` 时，它会被当作 model id 发出去（fallback）。**

代码依据：

- `config.py:157` `model: str = "cheap"`，默认值；C-18 只约束它是非空字符串。
- 全仓只有**一处**消费：`server.py:124` `SkillGenerator(..., model=str(evolve.model or ""))`。
- `SkillGenerator` 里只有一处用它：`skillgen.py:401-402`
  ```python
  if self.model:
      payload["derived_from"]["model"] = self.model[:128]
  ```
  即写进技能提案的 `derived_from.model` 字段做溯源。**没有任何代码把它塞进 HTTP 请求。**

也就是说原设计里 `"cheap"` 纯属溯源标签（设计文档 §九 注释「走 M4a 的 LLM 配置」——它描述的是成本档位，不是 provider 的模型名）。

但真实请求必须带一个 provider 认识的 `model`。所以 `MemorySettings.effective_evolve_model(label)` 的优先级是：

1. `MEMORY_EVOLVE_LLM_MODEL`（运维显式配的 provider 模型 id）；
2. 否则退回传入的 `evolve.model` 标签（`"cheap"`）——只对「路由器自己认识 `cheap` 这个别名」的部署成立。

**所以准确说法是：`evolve.model` 的语义是「成本档位标签，用于 skill 提案溯源」；它从未被设计成 provider 模型名。** 真实部署应当配 `MEMORY_EVOLVE_LLM_MODEL`，否则发出去的 `model: "cheap"` 大概率被上游 400。这一点写进了 README 的变量表。

## 6. 遗留 / 未做

- `jsonschema` 缺失导致 `test_contracts.py::test_c12_...` 失败（既有环境问题，非本次引入）。
- `MEMORY_EVOLVE_LLM_MODEL` 未配时不会报错，只在发请求时由上游拒绝。若希望启动期就拦住，可在 `build_evolve_llm` 里对「配了 base_url 但没配 model 且 `evolve.model` 是默认 `"cheap"`」加一条 warning——本次没加，避免对现有 `enabled=true` 部署产生新的启动噪音。
- 未改 C-18 / `schema/config.schema.json`（按约束）。
