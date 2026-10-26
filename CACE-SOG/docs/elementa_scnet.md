# ELEMENTA → SCNet：CACE-SOG 大规模训练

本入口在现有 CACE-SOG 模型上增加磁盘位置索引、按需构图、流式指标、可靠 checkpoint 和命令行训练。保留筛选阶段的通道与 `train/val/test`，不重新随机切分，不设置组成或轨迹帧数配额。

代码适配的起点为 `QinguiSun/SOG-Net` 的提交 `b249fc9bf1e92f5a0d4a1432acd26d7f4f0db89f`。训练报告应保存实际运行的适配提交和环境；用户已验证的 SCNet 环境为 `$HOME/envs/mlip`，Python 3.10.9。请继续使用该环境，下面不重新安装 CUDA 或重建 Conda 环境。

## 1. 文件怎样从 Windows 保存到 SCNet

路径对应关系：

| 用途 | 路径 |
|---|---|
| Windows 已完成导出 | `C:\Users\qingu\Documents\elementa_cace_sog_windows\export` |
| SCNet 项目 | `/work/home/einlovestan/projects/SOG-Net/CACE-SOG` |
| SCNet 数据（建议） | `/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/export` |
| 新建训练索引（建议） | `/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3` |
| 本次模型与日志 | `/work/home/einlovestan/projects/SOG-Net/CACE-SOG/runs/elementa-sog` |

将整个 `export` 目录的**内容与子目录结构**保存到 SCNet 的目标 `export` 中，不要多套一层 `export/export`。应保留 `manifest.jsonl`、`summary.json`、`run_status.json` 和各通道的 extxyz 文件。前三通道训练至少需要 `sog_candidates`、`lattice_auxiliary`、`pbe_lowdim_auxiliary` 各自的 `train.extxyz`、`val.extxyz`、`test.extxyz`。如果以后要直接启用 Spin，建议一并保存 `spin_review`；其余审核文件也可完整保留。

此前汇总中的原始导出帧约 56.49 GB；manifest、索引、模型、日志和文件系统缓存另计。传输前确认 SCNet 目标存储的实际可用空间与账户配额，不将原始帧字节数当作整个项目的空间上限。

训练索引在 SCNet 上重新建立，不需要上传原 NAS 的两个大文件，也不需要 Windows 筛选阶段的 `index.sqlite3`。manifest 中原始 Windows/NAS 来源路径仅作追溯；读取训练帧使用 `export_path` 等相对路径。

### 取得真实连接信息

在 SCNet 控制台取得该计算中心的 SSH/SFTP 域名、端口、账号与密钥。`einlovestan` 是已知 home 路径中的用户名，不能据此推导公网主机名或端口。SCNet 官方说明中介绍了从 E-Shell 的 SSH 连接入口取得连接信息；请使用您当前实例显示的值，不复制文档中的示例地址。[SCNet 文件管理进阶](https://www2.scnet.cn/help/docs/mainsite/hpc/advanced-guide/advanced-efile/index.html)

先在 SCNet 创建目标目录：

```bash
mkdir -p /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/export
mkdir -p /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices
```

Windows 可以使用 SCNet 当前提供的大文件传输入口；若该连接支持 SFTP，也可以在 WinSCP 中保存一个已验证的 SFTP 站点，上传整个目录。使用 **binary（二进制）**传输，保持原始字节和换行。WinSCP 的 SFTP 临时文件与续传功能适合大文件，断线后使用同一版本源文件继续传输；普通 SCP 不提供相同的续传行为。[WinSCP 续传说明](https://winscp.net/eng/docs/resume)

包内 PowerShell 辅助脚本可以先生成校验文件，再通过您已经保存的 WinSCP 站点续传。它不保存密码，也不自动接受未知主机密钥。在 WinSCP GUI 中完成首次登录与主机身份核对，将站点保存为例如 `SCNet` 后，在 Windows 的仓库 `CACE-SOG` 目录运行：

```powershell
.\examples\elementa\transfer_export.ps1 -StoredSession 'SCNet'
```

如 WinSCP.com 不在默认安装位置，通过 `-WinScpCom '实际路径\WinSCP.com'` 指定。也可以只生成校验清单，随后使用 GUI 传输：

```powershell
.\examples\elementa\transfer_export.ps1 -ChecksumOnly
```

生成 SHA256SUMS 会顺序读取一遍导出文件，因此需要时间；不将大文件加载到内存。该脚本上传目录内容，不删除远端多余文件。上传后，在 SCNet 检查内容：

```bash
cd /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/export
sha256sum -c SHA256SUMS
```

所有文件通过后开始建索引。保留源数据不变，不对 extxyz 做文本换行转换。若只上传前三通道，应生成对应数据副本的完整校验清单；不要让 SHA256SUMS 引用未上传文件。

如果校验失败，应覆盖重传相应文件后再次校验；不能因为文件大小相同就判定内容正确。辅助脚本的同步按大小比较，以便续传未改变的导出目录，最终内容一致性以 SHA 校验为准。

## 2. 更新代码并固定环境

在现有项目中先检查 `git status --short`。如有您在 SCNet 测试时做的本地修改，先保存为自己的提交或补丁，再切换分支，避免覆盖尚未提交的测试修复。

在 GitHub 审核适配分支后，可以检出该分支进行测试；合并后也可使用包含该提交的 main。下面分支名与本次 PR 对应：

```bash
cd /work/home/einlovestan/projects/SOG-Net
git fetch origin
git switch --track -c codex/elementa-large-scale-training origin/codex/elementa-large-scale-training
cd /work/home/einlovestan/projects/SOG-Net/CACE-SOG
conda activate "$HOME/envs/mlip"
python -m cace_sog_training --help
```

若本地已经存在同名分支，使用 `git switch codex/elementa-large-scale-training`。当前目录就是 Python 新入口所在目录；这些命令不需要更换原来的 PyTorch/CUDA 安装。若您习惯从其他目录启动，可先设置 `export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`。不要为本入口直接运行旧版 `pip install -e .` 强制变更 ASE/PyTorch；已验证的环境应先保持原样，缺失依赖只针对实际错误处理。

记录环境与实际代码：

```bash
mkdir -p runs/provenance
git rev-parse HEAD > runs/provenance/git-commit.txt
git status --short > runs/provenance/git-status.txt
conda list --explicit > runs/provenance/conda-explicit.txt
python -m pip freeze > runs/provenance/pip-freeze.txt
python -m cace_sog_training environment --output runs/provenance/environment.json
```

本适配不调用旧的整文件读取/整轮预测张量累积入口。原示例和模型基础代码保留，已有 Molten-NaCl 测试可继续独立运行。

## 3. 通道与既有划分

默认使用 `sog_candidates`、`lattice_auxiliary`、`pbe_lowdim_auxiliary`。如果全部数据与您提供的导出汇总一致，这三个通道合计：train 7,178,258 帧，val 893,408 帧，test 897,313 帧。实际以索引报告为准。

所有 train 参与参数优化；val 用于选择 checkpoint 和超参数；test 用于模型确定后的分通道最终评估。元素能量基线只从选定的 train 拟合，再原样用于 val/test。保持所有审核后加入的数据的原 split。

`pending_review` 不进入训练清单。`spin_review` 可以通过参数启用，但标准模型依然没有自旋输入：您需要先确认使用的数据符合一致的电子态策略，并同时提供 `--spin-policy consistent-state` 与 `--spin-policy-note '实际审核采用的状态策略说明'`。这两个参数记录用户对已整理数据的使用声明，不会自动验证磁态或把潜变量 q 变成自旋，也不能让同几何的多值 E/F 标签成为可区分输入。

本代训练入口的物理范围是**三维周期、按来源协议整体中性的体系**。索引要求三个 PBC 方向均为真、晶胞非退化且行列式为正；显式非零、无效或互相冲突的总净电荷字段拒绝入库，不会通过改通道或补零绕过。没有净电荷字段时，仅记录来源协议的中性假设，不声称已逐帧测得 Q=0。训练不会从名义价态或 magmom 推断总电荷。

这些检查以原始帧的晶胞、PBC 与显式元数据为依据，不能只看通道名称：例如 Spin 的高优先级可能使原筛选中的电荷/边界风险记录也归入 spin_review，启用该通道仍须重新通过相同物理范围检查。需要真实带电体系或非三维边界时，应另外适配模型与长程边界处理。

## 4. 索引与标签校验

建索引主要是 CPU/磁盘工作，不需要占用 A800。使用已分配的 CPU 计算任务，或下述 CPU sbatch 脚本；不要在登录节点上处理数百万帧。数据库与日志使用计算中心支持的工作文件系统，避免多任务同时写同一个索引。

已确认 CPU 数据处理分区为 `qdhcnormal`，A800 分区为 `qdagnormal`，并需要 `--account`。请在当前终端将环境变量 `SCNET_ACCOUNT` 设置为您实际获准使用的计费 account；实际值不写入公开仓库。下面会在变量未设置时停止提交：

```bash
cd /work/home/einlovestan/projects/SOG-Net/CACE-SOG
conda activate "$HOME/envs/mlip"
mkdir -p logs
: "${SCNET_ACCOUNT:?请先 export SCNET_ACCOUNT 为实际可用的 account}"
sbatch --partition=qdhcnormal --account="$SCNET_ACCOUNT" --time=1-00:00:00 examples/elementa/index_cpu.sbatch
```

索引作业中断后，可在保持该变量已设置的同一终端通过 `INDEX_RESUME=1 sbatch --partition=qdhcnormal --account="$SCNET_ACCOUNT" --time=1-00:00:00 examples/elementa/index_cpu.sbatch` 恢复。需要一并索引 Spin 时，增加 `WITH_SPIN=1`；包装器自动使用单独的 `elementa-with-spin.sqlite3`，也可通过 `DATABASE` 指定其他新索引。索引 Spin 不等于已经获准将它用于训练。

如果已经在获准的 CPU 计算任务内，等价的直接命令为：

```bash
cd /work/home/einlovestan/projects/SOG-Net/CACE-SOG
conda activate "$HOME/envs/mlip"
python -m cace_sog_training index --export-dir /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/export --database /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3 --channels sog_candidates lattice_auxiliary pbe_lowdim_auxiliary
```

索引逐帧校验原始字节 SHA、必要 E/F 标签、数值有限性、原子数、晶胞和边界，不将所有帧数组驻留内存；不重新调用 DFT。检查可能顺序读取大量数据，所需时间取决于工作文件系统性能。输入异常应先修复或审核，不通过静默跳帧获得“成功”。

中断后用完全相同的路径和通道恢复：

```bash
python -m cace_sog_training index --export-dir /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/export --database /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3 --channels sog_candidates lattice_auxiliary pbe_lowdim_auxiliary --resume
```

原始文件或 manifest 改变时应使用新索引。部分索引不允许正式训练。若今后增加 Spin，使用包括 Spin 的完整通道参数建立另一个索引，例如 `indices/elementa-with-spin.sqlite3`；不改变代码或重新筛选 NAS。

`index --limit-records N` 仅用于开发测试，会留下 partial 索引，正式训练拒绝读取；不是正式数据量配额。`train --max-steps N` 同样是调试限制，报告保持 partial，不能将它当作全量训练完成。

## 5. 在已分配的单卡 A800 节点运行

以下 GPU 命令应在您已有的 A800 交互任务内运行，或通过下一节 sbatch 脚本提交。不要在未分配 GPU 的登录节点运行正式训练。调度器设置的 `CUDA_VISIBLE_DEVICES` 保持原值。

先进行小样本前向、力反传、优化器更新和保存/重载检查：

```bash
python -m cace_sog_training smoke --database /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3 --channels sog_candidates lattice_auxiliary pbe_lowdim_auxiliary --output runs/elementa-smoke --device cuda --batch-size 2 --steps 10
```

然后检查几个网格参数下的能量和力变化：

```bash
python -m cace_sog_training converge --database /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3 --channels sog_candidates lattice_auxiliary pbe_lowdim_auxiliary --output runs/elementa-convergence --device cuda --n-dl 2 1 0.5 --samples 4
```

`N_dl` 是当前实现的网格控制参数，不等于旧示例的 `NpointsMesh`。这个初始检查使用固定的未训练模型参数，只诊断数值变化和运行开销；不能当作训练后模型的收敛证明。根据报告检查能量、力变化与所需精度及内存开销，再确定正式值；下面 `1` 是命令示例，不代表已经完成您全部化学空间的收敛证明。小样本阶段仅检查程序，正式训练清单没有相同的数据量限制。

smoke 的 `--steps 10` 在固定小 batch 上执行实际小步优化，检查参数更新、力反传、保存重载，并记录有限步数的内存轨迹；这只是一项短时诊断，不能证明数百万帧和所有晶胞尺寸下都不会耗尽内存。包装器可通过 `SMOKE_STEPS` 改变检查步数。正式训练的进度日志也持续记录进程 CPU 峰值 RSS 与 CUDA 峰值已分配内存，以便观察长期运行；峰值是截至当前的最高值，不等于当前瞬时用量，也不包含全部文件系统缓存或其他作业的内存。若启用独立 DataLoader worker，其进程内存还需结合调度器的作业级统计查看。

确认检查结果后，全量训练：

```bash
python -m cace_sog_training train --database /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3 --channels sog_candidates lattice_auxiliary pbe_lowdim_auxiliary --run-dir runs/elementa-sog --device cuda --batch-size 8 --epochs 50 --n-dl 1
```

batch size 和 epochs 是工程起点，按实际 A800 显存、每轮耗时和验证误差调整；减小 batch 不会减少数据清单。默认训练处理全部选定 train 帧，流式统计误差，按 val 保存最佳模型；checkpoint 用于继续训练。

中断后从最近 checkpoint 继续同一运行：

```bash
python -m cace_sog_training train --database /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3 --channels sog_candidates lattice_auxiliary pbe_lowdim_auxiliary --run-dir runs/elementa-sog --device cuda --batch-size 8 --epochs 50 --n-dl 1 --resume runs/elementa-sog/last.pt
```

恢复时，保持数据库、通道顺序、状态策略、seed、batch size、损失权重、优化器参数和模型/网格设置与首次运行一致；尤其不能省略首次修改过的 `--n-dl` 后让它退回默认值。上面两条直接命令均使用 `--n-dl 1`。checkpoint 保存已完成优化步骤的位置、优化器和随机状态；发生中断后，从最近保存的位置继续，尚未保存的步骤可能重算。`--epochs 50` 是总目标轮数，不是额外再训练 50 轮。

在同一软件与硬件环境中，本地小型 CPU 测试已经验证连续训练与中断恢复的一致性；跨 PyTorch/CUDA 版本、不同 GPU 或非确定性算子时，不承诺逐位相同的浮点结果。应保留环境记录，并在目标 A800 上重做 smoke 和恢复检查。

最后对各通道 test 分别输出能量/原子及力的 MAE/RMSE：

```bash
python -m cace_sog_training evaluate --database /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3 --checkpoint runs/elementa-sog/best.pt --split test --output runs/elementa-sog/test --device cuda
```

对选定的训练后模型也应重新检查网格。下面固定同一个 checkpoint 的权重，只改变计算网格，不为每个网格重新训练：

```bash
python -m cace_sog_training converge --database /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3 --channels sog_candidates lattice_auxiliary pbe_lowdim_auxiliary --checkpoint runs/elementa-sog/best.pt --output runs/elementa-trained-convergence --device cuda --n-dl 2 1 0.5 --samples 4
```

正式模型的网格选择仍应依据预先确定的精度目标与验证数据，保留 test 的最终评估用途。这里的小样本诊断不是整个数据空间的收敛证明。

测试不更新模型、基线或阈值。力仍依赖位置梯度，评估入口会保留所需自动微分；避免在外层套用 inference_mode。

也可使用包内 shell 包装器，参数与环境变量见脚本：

```bash
bash examples/elementa/run_single_gpu.sh smoke
N_DL=1 BATCH_SIZE=8 EPOCHS=50 bash examples/elementa/run_single_gpu.sh train
N_DL=1 BATCH_SIZE=8 EPOCHS=50 RESUME=runs/elementa-sog/last.pt bash examples/elementa/run_single_gpu.sh train
bash examples/elementa/run_single_gpu.sh evaluate
CONVERGENCE_CHECKPOINT=runs/elementa-sog/best.pt CONVERGENCE_DIR=runs/elementa-trained-convergence bash examples/elementa/run_single_gpu.sh converge
```

## 6. 使用已确认的 sbatch 调度方式

本项目给出单卡脚本 `examples/elementa/train_a800.sbatch`，使用已确认的 A800 分区 `qdagnormal`。先在项目目录创建日志目录，确认 `SCNET_ACCOUNT` 已设置为实际有权限的 account；时间限制按您实际允许的值调整：

```bash
cd /work/home/einlovestan/projects/SOG-Net/CACE-SOG
mkdir -p logs
: "${SCNET_ACCOUNT:?请先 export SCNET_ACCOUNT 为实际可用的 account}"
sbatch --partition=qdagnormal --account="$SCNET_ACCOUNT" --time=1-00:00:00 examples/elementa/train_a800.sbatch
```

实际 account 由用户终端环境提供，未硬编码在脚本中。脚本申请一张 GPU，不靠未经核实的 `gpu:a800` 资源名称猜测型号。保持 `SCNET_ACCOUNT` 已设置后，可通过相同脚本提交 smoke 或最终评估：

```bash
STAGE=smoke sbatch --partition=qdagnormal --account="$SCNET_ACCOUNT" --time=00:30:00 examples/elementa/train_a800.sbatch
STAGE=evaluate sbatch --partition=qdagnormal --account="$SCNET_ACCOUNT" --time=04:00:00 examples/elementa/train_a800.sbatch
```

改变参数或恢复：

```bash
N_DL=1 BATCH_SIZE=8 EPOCHS=50 sbatch --partition=qdagnormal --account="$SCNET_ACCOUNT" --time=1-00:00:00 examples/elementa/train_a800.sbatch
N_DL=1 BATCH_SIZE=8 EPOCHS=50 RESUME=runs/elementa-sog/last.pt sbatch --partition=qdagnormal --account="$SCNET_ACCOUNT" --time=1-00:00:00 examples/elementa/train_a800.sbatch
```

时间值是提交语法示例，不能保证足够训练一轮或整个数据集。根据作业日志调整壁钟时间和 checkpoint 策略。使用计算中心实际提供的命令查看作业状态；收到作业号仅代表提交成功，不代表模型已训练完成。

## 7. 后续纳入审核后的 Spin

先保证数据符合一致电子态策略，保存审核记录，并将 Spin 文件与 manifest 一起传入。可用另一个索引和运行目录：

```bash
python -m cace_sog_training index --export-dir /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/export --database /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa-with-spin.sqlite3 --channels sog_candidates lattice_auxiliary pbe_lowdim_auxiliary spin_review
python -m cace_sog_training train --database /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa-with-spin.sqlite3 --channels sog_candidates lattice_auxiliary pbe_lowdim_auxiliary spin_review --spin-policy consistent-state --spin-policy-note '填写实际采用并已核实的电子态选择策略' --run-dir runs/elementa-sog-with-spin --device cuda --batch-size 8 --epochs 50 --n-dl 1
```

策略说明必须替换为实际审核记录，不能将上面的填写提示当作已完成审核。这只改变命令行的数据选择与策略声明；它不把标准模型变成显式自旋模型。如果仅有一部分 Spin 数据审核通过，应先形成受控的审核后数据副本/清单，不对整个未整理通道作虚假的一致状态声明。包装器采用 `WITH_SPIN=1 SPIN_POLICY=consistent-state SPIN_POLICY_NOTE='实际状态策略'` 环境变量，默认使用独立的 elementa-with-spin.sqlite3 与 runs/elementa-sog-with-spin，也可用 DATABASE、RUN_DIR 覆盖。

## 8. 验证范围与输出解释

代码测试中的 CPU/小型合成结构结果，与用户先前在 SCNet/A800 上完成的测试分别记录。适配 PR 中的本地测试不能代替您在 SCNet 上的 smoke、网格收敛、完整 Molten-NaCl 复现或 ELEMENTA 全量训练。

本次适配已执行的小型 CPU 检查覆盖：完整 CLI 索引、实际参数更新的 smoke、调试中断、checkpoint 恢复、按 val 选择 best.pt、分通道 test，以及固定训练后 checkpoint 的网格比较。它验证入口与数据流程，不代表已经在 SCNet 上完成 ELEMENTA 全量训练。

正式运行应检查：输入索引为 complete；已处理训练帧数等于所选 train 数量；训练/验证 loss 有限；最佳 checkpoint 和最近 checkpoint 实际存在；模型重载后可预测；test 分通道计数和误差报告完整。误差降低用于评价拟合能力；独立长程响应证据仍需相应物理测试。
