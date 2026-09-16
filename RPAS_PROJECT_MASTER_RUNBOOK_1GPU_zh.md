# RPAS 项目总文档：单张 A100 完整运行方案

> 版本：2026-09-16  
> 目标：用一张 A100 完成当前 PPT 对应的 AIME、MASBench、GAIA 实验，并保持三方法公平、可复现、可审计。

## 0. 结论和硬门槛

单张 A100 的正确拓扑是：一个 Qwen3.5-9B vLLM replica、`TP=1`、只使用逻辑 GPU 0，三个方法通过共享服务队列串行或受控并发运行。现有双卡 launcher 不能直接调用。

当前 AIME 已有终态结果，不需要重跑。MASBench 和 GAIA 在正式统计前有两个硬门槛：

1. 当前 MASBench 的 AFlow/MaAS runner 仍存在 `search_select_not_executed=true` 的 test-only 路径，必须先接通各自原生 search→select→test；
2. GAIA 不能只把附件路径字符串传给模型，必须先完成受限附件读取/解析 adapter 并通过实际附件 smoke。

没有通过这两个门槛，只能报告 blocker 或 smoke，不得生成“完整主实验”分数。

## 1. 实验范围

主表比较仓库交接文档冻结的三个方法：

| 方法 | 正式要求 |
|---|---|
| RPAS | 原生 WAN-GEPA Pareto search、reflection、select、frozen test |
| AFlow | 原生 MCTS/workflow search、select、执行和 test |
| MaAS | 原生 controller/meta-agent/graph search、代码执行、select 和 test |

主模型固定为 Qwen3.5-9B：temperature=0、top_p=1、thinking off、`max_tokens=6144`。27B 只做独立 heterogeneous/reflection ablation，不进入 9B homogeneous 主表。

当前 PPT 截图出现过 `AFlow / ADAS / MaAS`，而仓库 runbook 冻结的是 `RPAS / AFlow / MaAS`。本方案以仓库交接为准；如果 ADAS 是必须的第三个对比，运行前必须补充其代码仓、commit、原生搜索预算和评分协议。

## 2. 数据和划分

### 2.1 MASBench

- 来源：`Salesforce/MASBench`；固定 revision：`e85fc3e60107c87b690f9cdc478bad375a243ed0`；
- 五个轴：`breadth`、`depth`、`horizon`、`parallel`、`robustness`；
- 每轴：official train 内 `search=24`、`select=24`，official test 内 `test=60`；
- `data_seed=2026`，按真实 realized axis value 分层；
- search/select 绝不读取 test，三个方法使用同一 test manifest；
- 主指标：完整序列 exact accuracy；component accuracy 仅作诊断。

### 2.2 GAIA

- 来源：`gaia-benchmark/GAIA` gated `validation`，不使用 official private test；
- 当前私有包 source revision：`682dd723ee1e1697e00360edccf2366dc8418dd9`；
- 内部划分：`search=60`、`select=30`、`gaia_local_test=60`；
- 按 `Level × attachment_type` 分层，模型输入不暴露 Level；
- 当前本地私有包共 150 条，审计为 `PASS`；
- 报告中必须写 `gaia_local_test`，不能冒充官方 GAIA test。

### 2.3 已有 AIME 结果

| 方法 | D_select | AIME 2025 | AIME 2026 |
|---|---:|---:|---:|
| RPAS | 20/30 | 12/30 | 13/30 |
| AFlow | 13/30 | 7/30 | 9/30 |
| MaAS | 1/30 | 1/30 | 1/30 |

## 3. 单卡基础设施

```text
CUDA_VISIBLE_DEVICES=0
        └── Qwen3.5-9B vLLM replica :8004
                └── RPAS / AFlow / MaAS 共享请求队列
```

资源边界：

- 只访问作业内逻辑 GPU `0`；不访问 GPU 1 或其它设备；
- 只启动一个 9B 服务，`tensor_parallel_size=1`；
- 不为每个方法复制一份模型；
- 并发只能改变排队速度，不能改变 search/select/test 数量、停止条件、test ids 或答案解析。

建议档位（实际值写入 run manifest）：

| 档位 | A100 40G | A100 80G | 用途 |
|---|---|---|---|
| S0 | `max_num_seqs=2`、client=2 | `max_num_seqs=4`、client=4 | smoke |
| S1 | `max_num_seqs=4`、client=4 | `max_num_seqs=8`、client=8 | 默认正式档 |
| S2 | `max_num_seqs=8`、client=6–8 | `max_num_seqs=12–16`、client=8–12 | S1 稳定后再试 |

固定 `max_model_len=16384`、`max_tokens=6144`。遇到 OOM、排队抖动或 timeout，只降低并发、batch token 或显存利用率，并记录实际值；不要删题或跳过搜索。

## 4. 节点初始化

在实际 A100 节点执行，不要在当前 Mac 工作区跑正式推理：

```bash
export RPAS_REPO_ROOT=/mnt/cpfs/epic-user/niutianle-20260612/RPAS
cd "$RPAS_REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0
hostname
nvidia-smi -L
test "$CUDA_VISIBLE_DEVICES" = 0
```

显式映射运行环境，不要假设旧 home 路径存在：

```bash
export RPAS_QWEN_ENV=/实际路径/qwen35_vllm
export RPAS_GEPA_ENV=/实际路径/gepa_exp
export RPAS_MODEL_PATH=/实际路径/Qwen3.5-9B

test -x "$RPAS_QWEN_ENV/bin/python"
test -x "$RPAS_QWEN_ENV/bin/vllm"
"$RPAS_QWEN_ENV/bin/python" -c 'import torch, vllm, transformers; print(torch.__version__)'
"$RPAS_GEPA_ENV/bin/python" -c 'import litellm, pyarrow; print("experiment deps OK")'
```

如果一个环境包含全部依赖，可以让 `RPAS_QWEN_ENV` 和 `RPAS_GEPA_ENV` 指向同一路径。缺依赖时只在节点环境安装匹配 CUDA/driver/PyTorch 的版本，不要在无 GPU 的本机盲装 vLLM。

## 5. 数据下载与冻结

### 5.1 MASBench

```bash
export DATA_ROOT="$RPAS_REPO_ROOT/data/ppt_20260916"
mkdir -p "$DATA_ROOT"
DATA_ROOT="$DATA_ROOT" bash scripts/fetch_ppt_datasets.sh

python -m experiments.prepare_masbench_aime_split \
  --data-dir "$DATA_ROOT/masbench" \
  --output-dir "$DATA_ROOT/masbench_aime_v1" \
  --seed 2026 --search-size 24 --select-size 24 --test-size 60
```

五个轴都必须有 `search.jsonl`、`select.jsonl`、`test.jsonl`；保存官方 parquet SHA256 和新 manifest SHA256。旧 `60/30/30` 目录不能替代当前正式切分。

### 5.2 GAIA

GAIA 需要用户已授权的 Hugging Face gated snapshot。token 只通过 secret store/环境变量注入，不写入命令、日志或 Git：

```bash
export GAIA_SNAPSHOT=/受限路径/gaia_snapshot
export GAIA_PRIVATE_PACKAGE=/受限路径/gaia_private_20260916
export GAIA_SOURCE_REVISION=<snapshot 实际 commit>
GAIA_SNAPSHOT="$GAIA_SNAPSHOT" \
GAIA_PRIVATE_PACKAGE="$GAIA_PRIVATE_PACKAGE" \
GAIA_SOURCE_REVISION="$GAIA_SOURCE_REVISION" \
bash scripts/prepare_gaia_private_package.sh
```

必须确认数量为 60/30/60、task id 互斥、hash 一致、附件存在且审计 `PASS`。records、答案、metadata 和附件只留在受限节点。

## 6. 单卡 vLLM 启动

交付版 `main_2gpu_max_throughput.sh` 是双卡入口，单卡不要调用。单卡只启动下面一个服务：

```bash
mkdir -p "$RPAS_REPO_ROOT/logs"
export MAX_MODEL_LEN=16384
export SERVER_MAX_NUM_SEQS=8
export MAX_NUM_BATCHED_TOKENS=16384
export GPU_MEMORY_UTILIZATION=0.90

CUDA_VISIBLE_DEVICES=0 "$RPAS_QWEN_ENV/bin/vllm" serve "$RPAS_MODEL_PATH" \
  --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port 8004 \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$SERVER_MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --language-model-only \
  > "$RPAS_REPO_ROOT/logs/vllm-9b-gpu0.log" 2>&1 &
export VLLM_PID=$!

for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8004/health >/dev/null && break
  kill -0 "$VLLM_PID" 2>/dev/null || { tail -n 240 "$RPAS_REPO_ROOT/logs/vllm-9b-gpu0.log"; exit 1; }
  sleep 5
done
curl -sf http://127.0.0.1:8004/health >/dev/null
```

记录 GPU 型号/显存、CUDA/driver、vLLM、模型 revision、服务参数和启动时间。

## 7. 正式运行顺序

### 7.1 先做硬门槛 smoke

每个方法至少跑一条 MASBench 轴；GAIA 覆盖无附件、PDF、表格以及当前包实际出现的图像/音频类型。检查：

- native search、select、test trace 确实发生；
- 输出能被统一 score-only adapter 解析；
- calls、prompt/completion/total tokens、latency、finish reason、error、truncation 都落盘；
- GAIA 工具读取受限于当前任务附件目录；
- 断点续跑不会重复污染结果。

### 7.2 MASBench 队列

共 15 个逻辑任务：3 methods × 5 axes。每个任务严格执行：

```text
official train/search 24 → official train/select 24
→ freeze selected config → official test 60
```

输出隔离为：

```text
outputs/main_1gpu_9b/masbench/<method>/<axis>/seed_0/
```

AFlow/MaAS 如果仍然只跑 frozen test，立即停止正式统计。

### 7.3 GAIA 队列

三个方法分别执行：

```text
validation/search 60 → validation/select 30
→ freeze selected config → gaia_local_test 60
```

输出隔离为：

```text
outputs/main_1gpu_9b/gaia/<method>/seed_0/
```

至少保存 `run_manifest.json`、`test_outputs.jsonl`、`calls.jsonl`、`tool_trace.jsonl` 和 `result.json`。

### 7.4 并发原则

S0 稳定后用 S1；连续 100 个请求无 OOM、异常 preemption、资源错误后才能试 S2。单卡推荐一个共享服务、一个或两个逻辑 worker；不要同时启动多个 vLLM 模型副本。先完成 MASBench 再 GAIA 最容易审计，也可以按 method×axis 串行完成，但必须保持目录和 manifest 隔离。

## 8. GAIA 附件 adapter

禁止只传 `Attachment path`。至少记录以下字段：

| 类型 | 必须记录 |
|---|---|
| text/structured | 编码、字节数、截断 |
| PDF | 页数、提取器版本、失败原因 |
| spreadsheet | sheet、行列范围、解析器版本 |
| image/audio | 输入 hash、工具/模型版本、返回状态 |
| archive/other | 明确支持或预先定义排除规则，不能静默删题 |

工具失败、文件不存在、解析失败、模型错误、答案抽取错误和超时必须分开统计；默认禁止联网搜索，避免额外变量。

## 9. 统一指标和成本

主指标：MASBench 完整序列 accuracy；GAIA `gaia_local_test` normalized exact match。

辅助指标：component accuracy、Level/轴/附件类型分层 accuracy、valid answer rate、execution/tool success、protocol error、truncation、calls、prompt/completion/total tokens、p50/p95 latency、wall time、GPU-hours、tokens/s 和失败分类。

search、select、test 三部分的成本都要分别记录；不能把 AFlow/MaAS 搜索调用记为零。未配置 GPU 单价时成本写 `null`，不要把 `$0` 当真实成本。主结果可先用 `seed=0`；稳定性重复应在冻结配置上做 test-only repeat，不得用 test 反馈调参。

## 10. 断点续跑和故障处理

- 每个 method×dataset×axis 独立 output/cache；
- 搜索候选、选择记录和 checkpoint 必须落盘；
- 中断后从 checkpoint 继续，重复请求标记为 retry/replay；
- OOM：降低并发、batch token 或显存利用率，保持数据和 max tokens 不变；
- vLLM 路径不存在：设置实际 `RPAS_QWEN_ENV`，不要假设旧 home 路径；
- MASBench 目录不存在：固定 revision 重新下载并生成 24/24/60；
- GAIA 401：完成 gated 授权，不能把 token 写进仓库；
- AFlow/MaAS 无 search/select trace：停止正式汇总；
- Tinker 仅作独立 provider，不能与本地 A100 结果混列。

## 11. 终态验收清单

### 数据

- [ ] MASBench revision、五轴 parquet hash、五轴 manifest hash 已记录；
- [ ] 每轴 search/select/test=24/24/60；
- [ ] GAIA source revision、metadata hash、私有 manifest hash 已记录；
- [ ] GAIA=60/30/60，id 互斥，audit=`PASS`；
- [ ] 无 test→search/select 泄漏。

### 方法和资源

- [ ] RPAS/AFlow/MaAS 都有原生 search→select→frozen test trace；
- [ ] 不存在 `search_select_not_executed` 正式结果；
- [ ] 三方法模型、解码、答案协议和 test manifest 一致；
- [ ] 只使用 `CUDA_VISIBLE_DEVICES=0`，只有一个 9B replica；
- [ ] 实际并发和回退参数写入 manifest。

### 公开交付

- [ ] 公开仓库不含 GAIA records、答案、metadata、附件或 secret；
- [ ] secret/raw-data/split-leakage scan 通过；
- [ ] 明确 `gaia_local_test` 不是 official private test；
- [ ] AIME、MASBench、GAIA 的结果和成本口径分开；
- [ ] JSONL/manifest 可以重建最终表格。

## 12. 当前状态

| 项目 | 状态 | 单卡动作 |
|---|---|---|
| AIME | 已完成 | 直接汇总，不重跑搜索 |
| GAIA 私有数据 | 已构建、audit PASS | 节点确认 revision 和 adapter |
| GAIA 正式推理 | 未完成 | 完成真实附件 smoke |
| MASBench 协议 | 已冻结 | 远端下载并生成 24/24/60 |
| MASBench RPAS | 有 runner/协议 | 节点 smoke 后运行 |
| MASBench AFlow/MaAS | 当前存在 test-only 风险 | 补齐原生 search/select |
| 双卡 launcher | 已有 | 单卡不要调用 |
| 本机 GPU | 不可用 | 只在 A100 节点执行 |

## 13. 相关文件

- `delivery-upload/README.md`
- `delivery-upload/notes/DELIVERY_RUNBOOK_zh.md`
- `delivery-upload/notes/FULL_MAIN_EXPERIMENT_2XA100_RUNBOOK_zh.md`
- `delivery-upload/notes/ENVIRONMENT_MAPPING_zh.md`
- `delivery-upload/data/GAIA_CONSTRUCTION_PROTOCOL_zh.md`
- `delivery-upload/scripts/preflight_main_2gpu.sh`
- `delivery-upload/scripts/main_2gpu_max_throughput.sh`
- `delivery-upload/experiments/prepare_masbench_aime_split.py`
- `delivery-upload/experiments/masbench_native.py`
- `private-gaia-package-20260915/manifest.json`
- `private-gaia-package-20260915/audit_result.json`
- `work/MAIN_EXPERIMENT_AUDIT.md`
