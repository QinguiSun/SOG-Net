# DPA4C 与周期 SOG：ELEMENTA 研究工作流（草稿）

本目录提供 `sog_candidates` 数据转换、训练配置、流式验证、配对测试和图表工具。当前只通过小型 CPU/模拟模型测试；尚未在真实 DeePMD、SCNet A800 或完整 ELEMENTA 数据上验证。不要将本草稿理解为模型已经训练完成。

指定的 [DPA4C-OMol 模型](https://huggingface.co/deepmodelingcommunity/DPA4C-OMol) 是面向分子的预训练模型；这里将其用于周期固体属于迁移实验。示例固定 Neo、版本 `v20260820`，训练和评估使用相同 checkpoint 家族及电子态策略。该模型卡标注 CC BY-NC 4.0，使用时需遵守其许可。

原始筛选导出的 `sog_candidates` 为 train 3,543,786、val 443,734、test 442,489 帧；实际以**完整训练索引**的统计为准。只选择这个通道，保留所有已索引帧和既有划分。DeePMD 随机抽取训练 batch，固定步数不会保证每帧恰好出现一次。

模型条件 `[0, 1]` 表示实验中统一输入的中性、单重度条件，不是从磁矩推导出的真实 DFT 状态；出现明确冲突元数据时工具停止。SOG 中的零和潜在电荷约束也是模型约束，不证明真实材料的电荷状态。

## 重要的当前阻碍

上游 [DeePMD-kit PR #6031](https://github.com/deepmodeling/deepmd-kit/pull/6031) 的精确提交 `ceefa8f806fa6e8607d79d10d8316b4ab325ad11` 包含周期 SOG，但其[官方测试](https://github.com/deepmodeling/deepmd-kit/actions/runs/40072526249/job/120808847130)仍有周期 PME compiled/eager virial 不一致和导出失败。部分非周期/其他路径通过不代表该组合通过。**先修复或更新并验证上游代码，再在目标 A800 上检查 E/F、力有限差分、参数梯度、周期平移、网格收敛、保存/重载及显存；通过前不要把 SOG 配置作为正式大规模训练入口。** 本仓库不声称已实现或修复上游周期 SOG。

该上游版本的内置 full validation 会缓存完整系统数据，可能随整个验证集增长。本流程关闭它，使用 `dpa4c_sog.validate` 逐个 checkpoint 流式遍历完整 val，按力分量 MAE 选择；训练显示中的一批 validation 不是完整验证。每个候选要完整读取一次 val，耗时与候选数量成正比。默认保留最近五个 checkpoint；选优范围只是保留下来的候选。

训练数据加载也有待解决的规模限制：当前 DeepMD NPY loader 为每个已访问的 system 保留一个 set 的缓存，内存可能随遇到的组成数增长；系统列表、目录扫描和按系统抽样也随 system 数增加。微调启动时的统计还会汇集各个 system 的样本，可能在第一个优化步骤之前占用大量内存。转换器的 `--set-frames` 只限制单个 set，**不能保证训练进程总内存有上限**。完整 354 万训练帧前必须适配或更换训练加载器和统计路径，再在 A800 主机上测试；仅把全数据的训练步数设为 200 不能成为安全的小样本试跑。

## 1. Windows 转换完整数据

完成原先的训练索引后，在本仓库 `CACE-SOG` 目录激活具有 NumPy 和 ASE 的 CPU 环境：

```powershell
python -m dpa4c_sog.convert --database 'C:\Users\qingu\Documents\elementa_cace_sog_windows\training\elementa.sqlite3' --output 'C:\Users\qingu\Documents\elementa_cace_sog_windows\dpa4c_converted' --assume-charge-spin 0 1
python -m dpa4c_sog.integrity seal --dataset 'C:\Users\qingu\Documents\elementa_cace_sog_windows\dpa4c_converted' --manifest 'C:\Users\qingu\Documents\elementa_cace_sog_windows\dpa4c_manifest.jsonl'
```

转换逐帧验证 source SHA，并按真实组成分成标准 DeePMD system，坐标和力采用相同原子排列。它不修改原始 extxyz，不重新划分或去重。输出目录必须是新目录；如果中断，状态是 partial，需要使用新目录重试，当前版本不支持续转。记录 seal 输出的 `manifest_sha256`，并独立保存；校验清单本身不是数字签名。

转换期间会有磁盘 SQLite 中间数据及最终数组的并存，转换之后还有完整的 frame 清单；预留充足空间。数据按组成产生大量小文件和目录，还需检查 Windows/SCNet 文件数、inode 和目录扫描性能。无法仅以原 extxyz 大小准确预测全部占用。

## 2. 上传与目标端核验

用二进制传输，按原有目录结构上传转换目录和其独立 SHA 清单。还需要上传原始 export 与 seal 的训练索引包并完成 bind，以供逐帧 val/test 评估：如果原训练索引覆盖三个通道，它的 bind 需要所有被索引的文件，即使本实验只训练 `sog_candidates`。原始 NAS 两个大文件不需要上传。

以下路径是示例，按目标实际目录调整；在有资源分配的作业节点上运行大型校验，不要在登录节点扫描数百万帧。SCNet 无独立 CPU 队列时，校验可在 `qdagnormal` GPU 作业配套的 CPU 上执行，会占用 GPU 作业资源。

```bash
export DPA_ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
export DPA_DATA="$DPA_ROOT/converted"
export DPA_INDEX=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3
mkdir -p "$DPA_ROOT/reports"
read -r -p '输入 Windows seal 输出的 manifest_sha256: ' DPA_MANIFEST_SHA
python -m dpa4c_sog.integrity verify --dataset "$DPA_DATA" --manifest "$DPA_ROOT/dpa4c_manifest.jsonl" --expected-manifest-sha256 "$DPA_MANIFEST_SHA" --report "$DPA_ROOT/reports/integrity.json"
python -m dpa4c_sog.alignment --database "$DPA_INDEX" --converted "$DPA_DATA" --output "$DPA_ROOT/reports/alignment.json"
```

前者核对目标目录每个文件和目录的完整性，后者核对每条转换记录与绑定索引中的来源、split、SHA 和排列信息。核验后保持转换目录不变；变更后需要重新核验，并使用新报告路径。索引 metadata 身份在 Windows→SCNet 重绑定后可以不同，不以此单独判定不同数据。

## 3. 固定环境、准备输入并试跑短程模型

模型卡说明 DPA4C 使用 DeePMD-kit 3.2 的 PyTorch Exportable 后端 `dp --pt-expt`，训练配置记录了 PyTorch 2.11/CUDA 12.8。先在独立环境中核实目标 A800、DeePMD 和 PyTorch 的组合；**不要直接覆盖已验证的 CACE 环境**。具体安装依平台条件参考[官方安装指南](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install.html)。

准备配对的发布配置及权重：

- [Neo 配置](https://huggingface.co/deepmodelingcommunity/DPA4C-OMol/resolve/2b2d3296e5401e5daac9ca9515fff0d790daf1c7/DPA4C-Neo-OMol25-100M-v20260820.json)
- [Neo 权重](https://huggingface.co/deepmodelingcommunity/DPA4C-OMol/resolve/2b2d3296e5401e5daac9ca9515fff0d790daf1c7/DPA4C-Neo-OMol25-100M-v20260820.pt)

将两文件传到 SCNet 的 `$DPA_ROOT`。先准备指向全量数据的配置，以下 200 步只是生成研究配置时使用的占位值；**不要直接用该全量配置启动训练**：

```bash
python -m dpa4c_sog.prepare \
  --release-config "$DPA_ROOT/DPA4C-Neo-OMol25-100M-v20260820.json" \
  --converted "$DPA_DATA" --integrity-report "$DPA_ROOT/reports/integrity.json" \
  --alignment-report "$DPA_ROOT/reports/alignment.json" --output "$DPA_ROOT/inputs_smoke" \
  --train-steps 200 --sog-steps 200 --batch-size 1
```

短程 `model` 部分保留发布配置；不会把 OMol 的统计文件或 `mix:100000` 批大小沿用到本数据。输出的 `input_sog_experimental.json` 仅供修复上游和完成数值验证后的研究使用。

在目标机生成单独的 smoke 副本：只复制 train 和 val 各最多两个组成的第一个 set，并重新校验复制文件；不读取 test，不改变全量数据。这只是接口测试，不能说明全量训练可以运行。

```bash
python -m dpa4c_sog.smoke --converted "$DPA_DATA" \
  --baseline-config "$DPA_ROOT/inputs_smoke/input_dpa4c.json" \
  --integrity-report "$DPA_ROOT/reports/integrity.json" \
  --output "$DPA_ROOT/smoke_subset" --systems-per-split 2 --steps 200
```

提交短程 A800 试跑时，下面的环境前缀要改为**实际安装并验证 DeePMD 的环境**；当前终端应位于 SCNet 的 `SOG-Net/CACE-SOG`：

```bash
export DPA_ENV="$HOME/envs/dpa4c"
export DPA_INPUT="$DPA_ROOT/smoke_subset/input_dpa4c_smoke.json"
export DPA_PRETRAINED="$DPA_ROOT/DPA4C-Neo-OMol25-100M-v20260820.pt"
export DPA_RUN="$DPA_ROOT/runs/baseline_smoke"
sbatch --export=ALL examples/dpa4c/baseline_a800.sbatch
```

如果非交互作业找不到 Conda，先将 `DPA_CONDA_SH` 设为该机器 `conda info --base` 下的 `etc/profile.d/conda.sh`。作业目录必须是新目录；中断后需按实际 DeepMD checkpoint 和官方 `--restart` 语义恢复，不能对已有目录盲目再次启动。固定的上游实验提交会恢复模型、优化器和训练步数；尚未确认它保存数据加载器的随机状态和样本位置，因此不能承诺断点续训后的样本顺序与不中断时完全相同。该模板尚未在 SCNet 上执行。

## 4. 数值试验、流式验证和最终测试

从短程 smoke 结果取得实际 checkpoint 路径，设置 `DPA_CKPT_1` 后，可先对原索引中的少量 **train** 帧检查能量—力有限差分与晶格平移不变性：

```bash
python -m dpa4c_sog.preflight --database "$DPA_INDEX" --model "$DPA_CKPT_1" --kind dpa \
  --output "$DPA_ROOT/reports/dpa_preflight.json" --charge-spin 0 1
```

工具写明位移和容差，超出容差时报告失败。得到 SOG 的实验 checkpoint 后可将 `--kind` 改为 `sog`、使用新的报告路径重复检查；该小样本检查不验证训练中 SOG 参数的梯度、网格收敛或显著长程响应，也不能代替 upstream 失败测试的修复。

每次完整读取原始索引中的 val 帧，逐个评估保留下来的候选 checkpoint。将下例环境变量设为实际 checkpoint 路径；可重复 `--checkpoint` 添加候选：

```bash
python -m dpa4c_sog.validate --database "$DPA_INDEX" --kind dpa \
  --checkpoint "$DPA_CKPT_1" --checkpoint "$DPA_CKPT_2" \
  --output "$DPA_ROOT/val_dpa" --charge-spin 0 1
```

报告 `validation.json` 中的 `selected_checkpoint` 按完整 val 力分量 MAE 选择。通过 SOG 数值检查并训练成功后，使用相同方法、`--kind sog` 及其候选 checkpoint 选择 SOG 模型。固定两者后才运行 test：

```bash
python -m dpa4c_sog.predict --database "$DPA_INDEX" --dpa "$SELECTED_DPA" --sog "$SELECTED_SOG" \
  --dpa-validation "$DPA_ROOT/val_dpa/validation.json" --sog-validation "$DPA_ROOT/val_sog/validation.json" \
  --output "$DPA_ROOT/paired_test.jsonl" --charge-spin 0 1
python -m dpa4c_sog.metrics --input "$DPA_ROOT/paired_test.jsonl" --output "$DPA_ROOT/test_report" --bins 200
```

预测前必须提供两份完整 val 报告，工具核对各自选中的 checkpoint 哈希、相同的索引与验证帧、电子态条件和最小 val 力 MAE。输出 `metrics.json`、`dpa_parity.png` 和 `sog_parity.png`；每个图含能量/原子与力分量两个面板、1:1 线和全部测试数据的密度。Mean difference 定义为预测减 DFT；MAE、RMSE、R² 也分别统计能量/原子和全部力分量，R² 在参考值方差为零时标为未定义。预测状态文件与 SHA 用于阻止使用未完成的结果。

目标是让 SOG 模型从已训练短程 checkpoint 继续微调；需要另行保存训练命令、起始 checkpoint 哈希与日志来证明继承关系，当前预测工具不会自行证明。SOG 模型经过额外优化，即使 test 指标改善，也不能仅凭这次比较断言全部改善来自长程项。要判断增益，建议额外比较同等继续训练步数和数据的短程模型，并设计独立长程响应测试。
