# RPAS PPT 主实验交付包（2026-09-14）

本包只对应当前 PPT 的主实验范围，不包含旧的 HumanEval、HotpotQA 或其它历史数据集。

## 交付范围

- MASBench：breadth、depth、horizon、parallel、robustness 五个轴；RPAS、AFlow、MaAS 三个原生方法。
- GAIA：validation 的本地分层实验设计与附件可见性 adapter 入口。GAIA 原始题目和附件不放入公开仓库，运行前由有权限的机器从官方 gated 数据集下载。
- 主模型：Qwen3.5-9B；两张 A100 时只允许作业内的 `gpu0` 和 `gpu1`，每张卡一个 vLLM replica。
- 27B 只作为单独的 heterogeneous/reflection ablation，不混入同质主表。

## 关键冻结口径

| 项目 | 正式设置 |
|---|---|
| MASBench split | official train 内 search=24、select=24；official test 内 test=60 |
| MASBench seed | `data_seed=2026` |
| 解码 | temperature=0、top_p=1、thinking off、max_tokens=6144 |
| 资源 | 两卡 `CUDA_VISIBLE_DEVICES=0,1`；每卡一个 9B 服务 |
| RPAS search | 9 个 seed candidates + 24 个新候选；shortlist 上限 8 |
| GAIA split | validation 内 search=60、select=30、test=60；按 level/附件类型分层 |

## 数据说明

MASBench 的官方快照、校验和与下载命令见 `data/MASBENCH_SOURCE_AND_GAIA_ACCESS.md`。GAIA 的官方条款禁止把 validation/test 数据重新放入可爬取的公开仓库，因此本包只提供下载、校验和本地路径约定，不复制原始 GAIA 文件。

## 启动

1. 在计算节点执行 `scripts/fetch_ppt_datasets.sh`，完成 MASBench 下载；GAIA 需要先在 Hugging Face 账号中接受 gated 条款。
2. 在节点上确认 `nvidia-smi -L` 只暴露目标两张卡，并确认 `CUDA_VISIBLE_DEVICES=0,1`。
3. 设置 `RPAS_REPO_ROOT`、模型路径和 endpoint 后运行 `scripts/main_2gpu_max_throughput.sh`。
4. GAIA 只有在附件读取 smoke 通过后才可进入正式实验；旧的“只传附件路径字符串”实现不算完整结果。

详细的变量控制、失败分类、telemetry、断点续跑和验收清单见 `notes/DELIVERY_RUNBOOK_zh.md`。
