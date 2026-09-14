# GAIA 主实验数据构建与实验协议

## 1. 范围冻结

这份协议只对应当前 PPT 的 GAIA 补充实验，不把旧 HumanEval、HotpotQA、历史 AIME 结果或 MASBench 数据混入 GAIA 表。

主实验比较三个方法：

| 方法 | 实验要求 |
|---|---|
| RPAS | 使用仓库中的原生搜索、反思、协作流程；搜索结果只允许来自 `search`，配置选择只允许来自 `select` |
| AFlow | 使用原生 MCTS/workflow 搜索与执行器；不得删掉搜索过程后当作单次调用 |
| MaAS | 使用原生 meta-agent/programmer/执行流程；代码执行、重试和失败必须计入 trace |

主模型固定为 Qwen3.5-9B。27B 只能作为另列的 heterogeneous/reflection ablation，不能进入三个方法的同质主表。

## 2. 数据构建

GAIA 只从用户已授权的 Hugging Face gated snapshot 中读取 `validation`。官方数据页说明了 `metadata.parquet`/`metadata.jsonl` 字段、附件相对路径以及不要把 validation/test 重新放进可爬取仓库的要求；因此公开仓库只放本构建器、协议、schema 和 hash manifest，不放问题、答案、metadata 或附件。

构建命令：

```bash
python scripts/build_gaia_ppt_dataset.py \
  --snapshot-root "$GAIA_SNAPSHOT" \
  --output-root "$GAIA_PRIVATE_PACKAGE" \
  --source-split validation \
  --source-revision "$GAIA_SOURCE_REVISION" \
  --seed 2026 \
  --search-size 60 \
  --select-size 30 \
  --test-size 60

python scripts/audit_gaia_ppt_dataset.py --root "$GAIA_PRIVATE_PACKAGE"
```

构建器的 fail-closed 规则：

1. 只接收明确位于 validation 路径或带有 validation/dev/public split 字段的 metadata；无法判断 split 时拒绝。
2. 每条记录必须有唯一 `task_id`、问题、gold answer 和合法 level（1/2/3）。冲突重复 id 直接失败。
3. 附件必须能在本地 snapshot 中解析；正式实验不允许用“只有路径字符串、读不到附件”的伪成功模式。
4. 每条记录保存 question/answer/attachment 的 SHA256；`manifest.json` 保存 source revision、metadata 文件 hash、seed、计数和隐私标记。

## 3. 划分

这里的 `test` 是从官方 validation 内部划出的本地 holdout，报告中必须写作 `gaia_local_test`，不能冒充 GAIA 官方 private test。

| 分区 | 数量 | 用途 | 是否可用于调参 |
|---|---:|---|---|
| search | 60 | 方法原生搜索、prompt/workflow 候选和失败模式探索 | 是 |
| select | 30 | 每个方法在固定候选中选出一个最终配置 | 仅可用于选择 |
| gaia_local_test | 60 | 一次性最终比较 | 否 |

三个分区按 `Level × attachment_type` 分层，`data_seed=2026`，task id 互斥。模型输入不暴露 `Level`，level 只用于分层和分组报告，避免把难度标签泄漏给模型。

## 4. 统一输入与工具边界

三个方法收到同一份问题文本、同一附件可见性和同一答案协议。附件通过统一的受限工具 adapter 暴露，而不是把宿主机任意路径直接塞进 prompt。adapter 至少记录：

- `list_files`：仅允许访问当前任务附件目录；
- `read_text`：记录编码、字节数和截断；
- `extract_pdf_text`：记录页数、提取器版本和失败原因；
- `extract_table`：记录 sheet/row/column、解析器版本和失败原因；
- `inspect_image` / `transcribe_audio`：记录输入 hash、模型/工具版本和返回状态。

工具失败、附件不存在、解析失败和模型输出格式错误不能被当成普通错误答案吞掉，必须分别计数。

固定解码与资源条件：

```text
temperature=0
top_p=1
enable_thinking=false
max_tokens=6144
CUDA_VISIBLE_DEVICES=0,1
global_max_inflight=32
per_replica_max_inflight=16
request_timeout=300s
```

两张 A100 各运行一个 9B replica，只允许使用逻辑 GPU 0/1。正式吞吐档为 `gpu_memory_utilization=0.95`、开启 prefix cache、每个 replica 最多 16 个 sequence、每个 runner 的 eval concurrency=8、每卡 2 个 runner；理论上限是每卡 16 个在途请求、双卡 32 个在途请求。若 smoke test 出现 OOM/排队抖动，只降低 `RPAS_NATIVE_CONCURRENCY` 或 `RPAS_GPU_MEMORY_UTILIZATION`，不改变数据、搜索预算或解码口径，并在 manifest 中记录实际值。并发只影响排队和 wall time，不得改变 split、停止条件、搜索候选数或选择规则。

## 5. 原生搜索与公平性

公平性控制在外层实验条件，方法能力保留在内层原生实现：

1. 同一模型、同一数据分区、同一附件工具、同一解码参数、同一超时和重试上限。
2. RPAS/AFlow/MaAS 各自完整运行其原生 search/controller；不把 RPAS 候选预算硬塞给 AFlow/MaAS，也不把 AFlow/MaAS 的搜索调用记为零成本。
3. 每个方法只在 `search` 生成候选，在 `select` 选一个冻结配置，之后只在 `gaia_local_test` 运行一次；禁止用 local test 反馈回改配置。
4. 三个方法的主表都报告搜索成本、选择成本和最终测试成本，避免只比较最后一轮答案准确率。
5. 所有输出保持原生，只在 score-only adapter 中统一抽取 `FINAL ANSWER: ...`；adapter 不修改推理内容，也不补写缺失答案。

## 6. 指标与结果表

主指标为 `gaia_local_test` exact accuracy；答案规范化只做大小写、空白、外层标点和明确的列表逗号规范化，不使用 LLM judge。必须同时报告：

- overall accuracy；Level 1/2/3 accuracy；有附件/无附件 accuracy；attachment_type accuracy；
- answer protocol valid rate、execution valid rate、tool success rate、truncation rate；
- 每题 calls、prompt/completion/total tokens、wall latency、p50/p95、GPU-hours 和推理成本；
- search/select/test 的独立调用量与 token 量；失败分类（模型、工具、解析、超时、资源、协议）；
- 三次不同 `run_seed` 的均值和标准差只作为稳定性附表，主 split 和主 test ids 不变。

输出最少包含：`run_manifest.json`、`test_outputs.jsonl`、`calls.jsonl`、`tool_trace.jsonl`、`result.json`。公开仓库只接受去答案、去问题、去附件的 summary 和 hash manifest。

## 7. 启动前验收

- [ ] `audit_gaia_ppt_dataset.py` 返回 `PASS`；
- [ ] search/select/gaia_local_test 数量和 hash 固定，三者无 task id 交集；
- [ ] 真实附件 smoke test 通过（PDF、表格、图片/音频按实际出现类型覆盖）；
- [ ] 三个方法都完成原生 search → select → frozen test 链路；
- [ ] 日志能区分模型调用、工具调用、重试、超时和解析失败；
- [ ] 公开仓库 secret scan、raw-data scan 和 split-leakage scan 通过；
- [ ] 报告不把本地 `gaia_local_test` 写成官方 GAIA test。
