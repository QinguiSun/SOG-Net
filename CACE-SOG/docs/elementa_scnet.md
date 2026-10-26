# ELEMENTA → SCNet：CACE-SOG 大规模训练

本入口在现有 CACE-SOG 模型上增加磁盘位置索引、按需构图、流式指标、可靠 checkpoint 和命令行训练。保留筛选阶段的通道与 `train/val/test`，不重新随机切分，不设置组成或轨迹帧数配额。

**当前采用 Windows 本地预处理 → 一次上传 → SCNet A800 训练。** 团队已确认未购买 SCNet CPU 专用资源；请从 [Windows 与 SCNet 操作指南](elementa_windows_scnet.md) 开始，不再提交 `index_cpu.sbatch`。新命令 `seal` / `bind` 负责完整内容校验和跨系统路径重绑定，`dispatch` 负责准备或提交 GPU 作业。本文保留的 SCNet 建索引方式适用于另有 CPU 授权或接受占用 GPU 作业资源的环境。

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

当前路线在 Windows 新建**训练索引**并封装快照，SCNet 校验后生成绑定本机路径的新索引，不重复解析数百万帧。不要上传原 NAS 两个大文件，也不要把筛选阶段的 `index.sqlite3` 当成训练索引。manifest 中原始 Windows/NAS 来源路径仅作追溯；读取训练帧使用 `export_path` 等相对路径。详见新指南。

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
git checkout -b codex/elementa-large-scale-training --track origin/codex/elementa-large-scale-training
cd /work/home/einlovestan/projects/SOG-Net/CACE-SOG
conda activate "$HOME/envs/mlip"
python -m cace_sog_training --help
```

若本地已经存在同名分支，使用 `git checkout codex/elementa-large-scale-training`。当前目录就是 Python 新入口所在目录；这些命令不需要更换原来的 PyTorch/CUDA 安装。若您习惯从其他目录启动，可先设置 `export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`。不要为本入口直接运行旧版 `pip install -e .` 强制变更 ASE/PyTorch；已验证的环境应先保持原样，缺失依赖只针对实际错误处理。

部分 SCNet 节点的 Git 较旧，不提供 `git switch`；本文使用兼容的 `git checkout`。如果需要固定到已经测试的适配提交，也可在 fetch 后执行：

```bash
git checkout --detach 7bbe969ce15bcd440a32ff98a93e385449f5459e
```

这里显示 `detached HEAD` 是固定提交运行的正常状态，不影响训练。随后进入 CACE-SOG 目录继续执行原命令；需要返回主分支时使用 `git checkout main`。若 Git 报告本地修改会被覆盖，先保存自己的修改，再切换；不要为此使用强制 checkout 或 reset。

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

以下 CPU 提交模板要求当前登录资源入口实际提供并授权 `qdhcnormal`。若 `sinfo` 和账户关联只显示 `qdagnormal`，请使用第 10 节的入口选择说明；在 GPU 入口修改 account 参数并不能自动获得 CPU 分区权限。

建索引主要是 CPU/磁盘工作，不需要占用 A800。使用已分配的 CPU 计算任务，或下述 CPU sbatch 脚本；不要在登录节点上处理数百万帧。数据库与日志使用计算中心支持的工作文件系统，避免多任务同时写同一个索引。

您提供的 CPU 数据处理分区为 `qdhcnormal`，A800 分区为 `qdagnormal`。下面的示例省略 `--account`，由 Slurm 使用提交环境或默认账户；省略该参数不代表已验证当前登录入口对该分区的授权。如果提交出现 account/partition 错误，先按第 9 节查询实际关联，再确定是否需要显式指定有效账户或更换资源入口。

```bash
cd /work/home/einlovestan/projects/SOG-Net/CACE-SOG
conda activate "$HOME/envs/mlip"
mkdir -p logs
sbatch --partition=qdhcnormal --time=1-00:00:00 examples/elementa/index_cpu.sbatch
```

索引作业中断后，可通过 `INDEX_RESUME=1 sbatch --partition=qdhcnormal --time=1-00:00:00 examples/elementa/index_cpu.sbatch` 恢复。需要一并索引 Spin 时，增加 `WITH_SPIN=1`；包装器自动使用单独的 `elementa-with-spin.sqlite3`，也可通过 `DATABASE` 指定其他新索引。索引 Spin 不等于已经获准将它用于训练。

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

本项目给出单卡脚本 `examples/elementa/train_a800.sbatch`，默认使用您提供的 A800 分区 `qdagnormal`。示例省略 account，实际采用的账户及其分区权限以当前集群关联为准。先在项目目录创建日志目录；时间限制按您实际允许的值调整：

```bash
cd /work/home/einlovestan/projects/SOG-Net/CACE-SOG
mkdir -p logs
sbatch --partition=qdagnormal --time=1-00:00:00 examples/elementa/train_a800.sbatch
```

脚本申请一张 GPU，不靠未经核实的 `gpu:a800` 资源名称猜测型号。可通过相同脚本提交 smoke 或最终评估：

```bash
STAGE=smoke sbatch --partition=qdagnormal --time=00:30:00 examples/elementa/train_a800.sbatch
STAGE=evaluate sbatch --partition=qdagnormal --time=04:00:00 examples/elementa/train_a800.sbatch
```

改变参数或恢复：

```bash
N_DL=1 BATCH_SIZE=8 EPOCHS=50 sbatch --partition=qdagnormal --time=1-00:00:00 examples/elementa/train_a800.sbatch
N_DL=1 BATCH_SIZE=8 EPOCHS=50 RESUME=runs/elementa-sog/last.pt sbatch --partition=qdagnormal --time=1-00:00:00 examples/elementa/train_a800.sbatch
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

## 9. 提交时出现 Invalid account or account/partition combination

这是 Slurm 接收作业时拒绝账户/分区关联，索引或模型程序尚未启动。无需因此重建索引、改模型或重装 Conda。

即使命令没有显式指定 account，Slurm 仍会使用默认账户；SBATCH_ACCOUNT 等提交环境变量也可影响选择。账户关联由用户、集群、账户和分区共同决定，不能只凭这句错误认定必须增加 --account，也不能把分区可见当作提交授权。[Slurm 账户关联](https://slurm.schedmd.com/sacctmgr.html)、[提交参数与环境优先级](https://slurm.schedmd.com/sbatch.html)

请在发生错误的同一个登录终端执行以下只读查询，保留完整输出或权限错误：

```bash
sinfo -o "%P %a %l"
sacctmgr -P show user where name="$USER" format=User,DefaultAccount
sacctmgr -P show assoc where user="$USER" format=Cluster,Account,Partition,QOS
printenv SBATCH_ACCOUNT SBATCH_PARTITION SBATCH_QOS SBATCH_CLUSTERS SLURM_CLUSTERS
```

最后一条无输出表示这些变量未设置。部分中心限制普通用户查看 accounting 数据，此时保留错误，并使用 SCNet 的资源查询入口或技术支持核实该用户的 CPU/GPU 资源授权；不能据查询失败推断账户不存在。

需要据实际输出区分：当前登录的集群是否提供目标分区；有效关联是否允许该账户使用目标分区；默认账户是否匹配；是否有遗留的 SBATCH 变量覆盖脚本。只有核实了有效组合后，才通过命令行覆盖 --partition 或显式指定 --account。脚本内不写死未知账户，不尝试修改集群的授权数据库。

此前 GPU 作业能提交，也不自动证明相同登录入口或默认账户可使用 CPU 分区。若存在已成功作业，可提供其提交脚本中资源相关的 #SBATCH 行作为比对，不需要提供密码或密钥。

## 10. 当前入口只有 qdagnormal 时怎样建立索引

如果当前登录终端的 sinfo 只显示 qdagnormal，账户关联的 Cluster/Partition 也只列出 qdcs/qdagnormal，而 SBATCH 相关环境变量未设置，则 CPU 脚本请求的 qdhcnormal 不属于当前已确认可提交的关联。该情形不能通过猜 account 名解决；省略 account 时，现有默认账户已经有 GPU 分区关联。

本项目团队已确认没有购买 SCNet CPU 专用资源，目前应按新指南在 Windows 处理。只有以后另外获得 CPU 授权时，才查询相应入口的分区和关联，再考虑使用 `index_cpu.sbatch`；不能通过 sbatch 参数创建授权。

CPU 和 GPU 资源入口不一定共享工作文件系统。索引绑定实际路径、文件状态和哈希；如果数据存储不共享，不能只把 SQLite 文件复制回 GPU 环境继续使用。现在可用 `seal` 生成带完整哈希的快照，再在实际训练的数据位置用 `bind` 校验并生成新索引；也可重新建索引。

### 在当前 GPU 入口继续的备用方式

如果希望先在当前已授权入口跑通，并接受占用一张 A800 作业资源，可直接复用现有 GPU 批处理脚本，将 STAGE 明确设为 index：

```bash
cd /work/home/einlovestan/projects/SOG-Net/CACE-SOG
mkdir -p logs
STAGE=index sbatch --export=ALL --partition=qdagnormal --job-name=elementa-index-gpu --time=1-00:00:00 examples/elementa/train_a800.sbatch
```

该命令只执行索引与标签校验，不启动模型训练。现有脚本仍申请一张 GPU 和四核 CPU，并执行 CUDA 可见性检查；索引本身主要消耗 CPU 与磁盘，所以此路径会占用 A800 资源及相应配额/计费。它是资源入口受限时的备用方案，不是 CPU 专用分区已经获授权的证明。

--export=ALL 显式把 STAGE=index 传入作业，避免其他导出设置导致阶段参数丢失。日志仍保存在 logs/elementa-<作业号>.out 和 .err。该备用命令在最初适配提交中已经受支持，无需修改 Python 或重新安装环境。

如果此前已有同一输入与参数的部分索引，再额外设置 INDEX_RESUME=1；单纯 sbatch 提交被拒绝不会启动索引，也不产生新的部分索引。索引完整结束后，继续原 smoke/converge/train/evaluate 阶段。时间限制仍需符合该分区规则。
