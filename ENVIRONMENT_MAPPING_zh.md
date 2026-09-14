# SCIR 环境映射与启动前检查

## 现象

原始启动器把 `qwen35_vllm` 和 `gepa_exp` 写成了固定的 home-directory 路径。本机没有 conda/vLLM 时会在 vLLM 启动前立即失败，这不代表 SCIR 计算节点也没有运行环境。

新入口 `scripts/preflight_main_2gpu.sh` 已改为按以下顺序寻找环境：显式 `RPAS_QWEN_ENV`/`RPAS_GEPA_ENV` → 常见 conda/mamba 前缀 → 当前 `CONDA_PREFIX`/conda 环境列表 → PATH 中的 vLLM。找到后会检查 `vllm`、`torch`、`transformers`、`litellm`、`pyarrow`、`nvidia-smi` 和 `curl`，并打印最终解析路径。

## SCIR 节点上的推荐用法

先在已分配的节点上确认模块或 conda 环境，而不是在本地登录工作站安装 CUDA 版本依赖：

```bash
cd /path/to/RPAS-PPT-Main-Experiments-20260914

# 方案 A：节点已经有两个环境
export RPAS_QWEN_ENV=/实际路径/qwen35_vllm
export RPAS_GEPA_ENV=/实际路径/gepa_exp

# 方案 B：一个环境同时包含 vLLM + litellm + 实验依赖
# export RPAS_QWEN_ENV=/实际路径/one_env
# export RPAS_GEPA_ENV=/实际路径/one_env

export RPAS_REPO_ROOT="$PWD"
export RPAS_MODEL_PATH=/实际路径/Qwen3.5-9B
export RPAS_MASBENCH_DATA_ROOT=/实际路径/masbench_aime_v1
export CUDA_VISIBLE_DEVICES=0,1

bash scripts/preflight_main_2gpu.sh
bash scripts/main_2gpu_max_throughput.sh
```

如果节点只有 Python 环境但没有 `litellm`/`pyarrow`，先在该环境中安装仓库实验依赖；vLLM 必须选择与节点 CUDA、PyTorch 和驱动匹配的 wheel/container，不能直接在无 GPU 的本地 Python 3.13 环境盲装最新版：

```bash
"$RPAS_QWEN_ENV/bin/python" -m pip install -U litellm pyarrow
"$RPAS_QWEN_ENV/bin/python" -c 'import torch, vllm, transformers; print(torch.__version__)'
```

如果 AFlow/MaAS 使用独立代码目录，还需在节点设置其路径并确保其依赖在 `RPAS_GEPA_ENV` 中可导入。依赖安装完成前不要进入正式队列；preflight 返回 `runtime checks: PASS` 后才启动两张 A100。

## 吞吐档与回退

启动器当前默认使用 `gpu_memory_utilization=0.95`、prefix cache、`max_num_seqs=16`、`max_num_batched_tokens=32768`、每卡 2 个 runner、每个 runner 8 并发。若节点实际是 32G 卡或模型/工具栈额外占显存，按顺序降低 `RPAS_NATIVE_CONCURRENCY`、`RPAS_SERVER_MAX_NUM_SEQS`、`RPAS_GPU_MEMORY_UTILIZATION`；每次回退都保留日志中的实际参数。不要通过降低 test 数量或跳过原生搜索来“提速”。
