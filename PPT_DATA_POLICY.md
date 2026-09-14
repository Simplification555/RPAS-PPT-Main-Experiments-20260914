# 当前 PPT 数据包说明

## MASBench

- 来源：`Salesforce/MASBench`
- 固定 revision：`e85fc3e60107c87b690f9cdc478bad375a243ed0`
- 配置：`breadth`、`depth`、`horizon`、`parallel`、`robustness`
- 官方数据集声明许可证：Apache-2.0；使用时仍需保留原数据集说明和引用。
- 正式切分：每个轴从 official train 生成 search=24、select=24，从 official test 生成 test=60；真实轴值分层，`data_seed=2026`。

下载脚本不会把历史 HumanEval/HotpotQA 数据混入本包。

## GAIA

- 来源：`gaia-benchmark/GAIA`
- 只使用公开 validation，不接触 private test answer。
- 公开仓库不复制 GAIA 的 metadata、答案或附件。官方数据页要求 gated access，并明确要求不要把 validation/test 重新放入可爬取仓库。
- 计算节点需由用户账号完成授权后执行 `snapshot_download`；脚本只记录本地 snapshot 路径和 SHA256，不把数据推回 GitHub。

## 校验要求

下载后必须保存：dataset revision、每个文件 SHA256、下载时间、split manifest SHA256。任何重新下载或 revision 变化都要生成新 manifest，不覆盖旧结果。
