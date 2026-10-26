# Windows 预处理，一次部署数据，DPDispatcher 提交 SCNet 训练

这条路线用于当前只有 `qdagnormal` A800 权限、没有独立 CPU 分区授权的场景：Windows 负责逐帧解析、标签检查、训练索引与基线准备；大文件只部署一次；SCNet 验证并绑定路径后，由 DPDispatcher 只传小配置/脚本并提交 GPU 作业。

训练保持全部所选帧及现有 `train/val/test`，默认选择 `sog_candidates`、`lattice_auxiliary`、`pbe_lowdim_auxiliary`，不纳入 `pending_review`，不重新随机划分。审核后的 Spin 可以通过通道参数扩展，但普通模型仍无显式自旋输入。

## 1. 路径与环境

| 内容 | Windows | SCNet |
|---|---|---|
| 已完成筛选的数据 | `C:\Users\qingu\Documents\elementa_cace_sog_windows\export` | `/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/export` |
| 本次训练索引 | `C:\Users\qingu\Documents\elementa_cace_sog_windows\training\elementa.sqlite3` | `/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3`（bind 后新建） |
| 密封准备包 | `C:\Users\qingu\Documents\elementa_cace_sog_windows\training\bundle` | `/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/bundle` |
| 代码 | 示例为 `C:\Users\qingu\Documents\SOG-Net\CACE-SOG` | `/work/home/einlovestan/projects/SOG-Net/CACE-SOG` |
| 持久训练输出 | 只下载小报告，checkpoint 可选下载 | `/work/home/einlovestan/runs/elementa-sog-001/train` |

Windows 的 `training\elementa.sqlite3` 是此次训练适配新建的索引，**不是**旧筛选器根目录中的 `index.sqlite3`。它们不能互换。索引、bundle 和临时 SQLite 文件放在空间充足的 Windows 本地盘；源 extxyz 保持不变。

空间按副本数量预留：Windows 除原 export 外，还要同时保存训练数据库与 bundle 内的数据库快照，约为 **2 倍训练数据库大小**，另加 SQLite 临时文件和事务日志。SCNet 同时保存上传的快照与 bind 后数据库，也约为 **2 倍训练数据库大小**，再加完整 manifest、所选 extxyz、模型、checkpoint 与日志。数据库大小随记录和字符串长度变化，不能只根据原始 extxyz 的体积推算总空间。

Windows 预处理仅需 Python、NumPy、ASE，不需要 PyTorch/CUDA。可以复用已经装好 NumPy/ASE 的筛选环境，或在 Windows 创建独立 CPU 环境：

```powershell
conda create -n elementa-prep python=3.10 pip -y
conda activate elementa-prep
python -m pip install numpy ase
```

取得下一节的适配源码、进入其 CACE-SOG 目录后，作为 DPDispatcher 控制端时，在这个本地环境安装仓库提供的控制端依赖：

```powershell
python -m pip install -r .\examples\elementa\dispatcher\requirements-controller.txt
```

SCNet 继续使用已经验证的 `$HOME/envs/mlip`，不为此重新安装 CUDA。训练时的邻居图构建、逐批数据读取等主机 CPU 工作，在同一个 GPU 作业分配的 CPU 核上执行；不需要另购/申请独立 CPU 分区。GPU 作业期间的资源使用与计费仍以平台实际规则为准。

## 2. Windows 与 SCNet 使用同一版代码

本次适配分支为 `codex/windows-preprocess-scnet-dispatch`。先审核代码；在 Windows 和 SCNet 分别获取分支。以下使用兼容旧 Git 的 checkout：

Windows 尚无该仓库时，可先执行下行；已有仓库则直接进入原目录，不覆盖自己的文件：

```powershell
git clone https://github.com/QinguiSun/SOG-Net.git 'C:\Users\qingu\Documents\SOG-Net'
Set-Location 'C:\Users\qingu\Documents\SOG-Net'
```

在各自的仓库目录执行：

```bash
git fetch origin
git checkout -b codex/windows-preprocess-scnet-dispatch --track origin/codex/windows-preprocess-scnet-dispatch
git rev-parse HEAD
```

如果本地分支已存在，使用 `git checkout codex/windows-preprocess-scnet-dispatch`。切换前保存自己的本地修改，不使用强制 checkout 或 reset。合并 main 后，也可检出包含该适配的已审核提交。

在 SCNet 上确认位置与环境：

```bash
cd /work/home/einlovestan/projects/SOG-Net/CACE-SOG
conda activate "$HOME/envs/mlip"
git rev-parse HEAD
python -c 'import sys; print(sys.prefix)'
conda info --base
```

把实际的 40 位 `git rev-parse HEAD` 输出用于稍后的 `expected_git_commit`；不要填 `main` 字符串或早期原始提交。`sys.prefix` 的绝对路径用于 `remote_conda_env`；如果非交互作业不能找到 conda，把 `conda info --base` 对应目录下的 `etc/profile.d/conda.sh` 填入 `remote_conda_sh`。实际环境、提交都从当前已验证会话取得，不猜安装位置。

## 3. 在 Windows 完成训练索引和 seal

在 Windows 的适配源码目录执行：

```powershell
Set-Location 'C:\Users\qingu\Documents\SOG-Net\CACE-SOG'
.\examples\elementa\prepare_windows.ps1
```

辅助脚本先构建完整训练索引，再执行 seal；它只改变当前进程的 Python/临时目录变量，退出时恢复，不永久修改 Windows 设置。若使用已有筛选虚拟环境，可以显式指定 Python：

```powershell
.\examples\elementa\prepare_windows.ps1 -PythonExe 'C:\Users\qingu\Documents\elementa_cace_sog_windows\.venv\Scripts\python.exe'
```

如果 C 盘空间不足，优先用 `-WorkDir 'D:\elementa-training'` 将数据库、bundle 和本次 SQLite 临时目录一起放到实际有空间的本地盘；相应上传源和本地配置路径也以新 WorkDir 为准。

索引中断后，保持源路径、内容和通道不变，恢复：

```powershell
.\examples\elementa\prepare_windows.ps1 -Resume
```

如果本机禁止运行 PowerShell 脚本，可以直接执行等价的 Python 命令，不需要永久改变执行策略：

```powershell
python -X utf8 -u -m cace_sog_training index --export-dir 'C:\Users\qingu\Documents\elementa_cace_sog_windows\export' --database 'C:\Users\qingu\Documents\elementa_cace_sog_windows\training\elementa.sqlite3' --channels sog_candidates lattice_auxiliary pbe_lowdim_auxiliary
python -X utf8 -u -m cace_sog_training seal --database 'C:\Users\qingu\Documents\elementa_cace_sog_windows\training\elementa.sqlite3' --bundle 'C:\Users\qingu\Documents\elementa_cace_sog_windows\training\bundle'
```

直接命令的索引恢复加 `--resume`。`seal` 只接受 complete 索引，bundle 必须是新目录；如果 seal 中断，保留完成的训练索引，换一个新的 bundle 目录重试，不删除已完成的索引。辅助脚本可通过 `-Bundle '新的本地目录'` 指定。不要使用调试用的 `--limit-records` 作为正式预处理，它会留下 partial 索引。

seal 产生：

```text
bundle/
  index.sqlite3    完整训练索引快照，含准备好的 train-only 元素能量基线
  bundle.json      最后写入的 complete 描述、数据文件清单与完整 SHA256
```

bundle 不复制原始数据。源导出文件继续保存在原位置，直到上传与远端绑定完成。Windows 逐帧验证 E/F、原子数、晶胞、三维 PBC、显式电荷/单位/收敛证据；本代的中性周期协议仍适用，缺失电子态证据保持来源假设。密封并不会认证长程响应或自旋状态。

这仍是一项完整的数据准备任务：当前原始 manifest 有约 3335 万帧记录，默认前三通道索引约 897 万帧。index 需要流式遍历 manifest 并解析所选帧；seal 还要复制/读取数据库，准备 train-only 基线，并完整哈希 manifest 与所有引用的数据文件。有界内存不等于耗时很短，数百万帧可能需要很多小时甚至数天。请根据阶段进度、磁盘吞吐与实际 CPU 情况判断，不给出固定完成时间。

## 4. 数据只上传一次

使用 SCNet 控制台当前提供的 SSH/SFTP 主机、端口和用户名；真实连接端点只保存在本地私有配置中，不写入公开仓库。平台重建入口后如连接信息变化，更新本地配置。私钥必须是 **Windows 本机实际可读**的文件；原 Mac 路径在 Windows 上不能使用，也不应提交到 GitHub。需要时从平台取得授权密钥并妥善保存在本机。

先在 SCNet E-Shell 创建持久目录：

```bash
mkdir -p /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/export
mkdir -p /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/bundle
mkdir -p /work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices
mkdir -p /work/home/einlovestan/runs/elementa-sog-001
```

在 WinSCP 使用 SFTP、控制台提供的主机/端口/用户和本机授权私钥连接，核对主机身份。使用 binary 二进制传输，保持字节、目录层级和换行；SFTP 支持临时文件续传。[WinSCP 官方续传说明](https://winscp.net/eng/docs/resume)

需要上传的内容：

1. Windows `training\bundle` 中的两个文件，保存到 SCNet `.../ELEMENTA_CACE_SOG/bundle`。
2. `bundle.json` 的 files 清单引用的 extxyz，按 `relative_path` 保存到远端 `export` 下。默认前三通道通常是三个通道各三个 split，共最多九个 extxyz 文件。
3. **完整原始** `manifest.jsonl`、`summary.json`、`run_status.json`（源导出中存在的状态文件），保存到远端 `export`。即使本次只用前三通道，也不能裁剪或重写 manifest，因为密封包记录的是它的完整 SHA。

其余 Spin/Pending 文件不属于前三通道的必要上传清单；可以保留在 Windows，后续需要时增量部署。也可以完整上传整个 export，但会使用更多存储和传输时间。不要额外套出 `export/export` 或 `bundle/bundle`。

传输完成后再开始 bind。部署后的源文件保持不变；新数据版本使用新的数据目录、bundle 和训练运行目录。DPDispatcher **不会**在每次作业上传数据：它的 forward_files 只含少量生成的脚本/配置，数据和索引使用这里的持久绝对路径。

## 5. 只在 Windows 本地保存连接配置

如果已经获得本地预填配置，保留该文件并跳过下面的 Copy-Item；不要用模板覆盖您填好的连接信息。否则复制公开模板到仓库外的工作目录，避免把连接密钥路径和私有配置加入 Git：

```powershell
$ElementaConfig = 'C:\Users\qingu\Documents\elementa_cace_sog_windows\training\dispatch.local.json'
Copy-Item -LiteralPath '.\examples\elementa\dispatcher\scnet.json' -Destination $ElementaConfig
```

编辑这个**本地副本**。主要字段对应：

| 配置字段 | 应填写的内容 |
|---|---|
| `expected_git_commit` | SCNet 已审核源码的实际 40 位提交 SHA |
| `machine.remote_profile.hostname` | SCNet 控制台给出的当前 SSH 主机 |
| `machine.remote_profile.port` | SCNet 控制台给出的当前 SSH 端口 |
| `machine.remote_profile.username` | `einlovestan` |
| `machine.remote_profile.key_filename` | Windows 上私钥的实际绝对路径；JSON 可用 `C:/Users/qingu/.ssh/scnet_key` 形式，这只是待替换的示例 |
| `machine.remote_root` | DPDispatcher 的远端持久 staging 目录，与数据和模型目录分开 |
| `remote_project_dir` | `/work/home/einlovestan/projects/SOG-Net/CACE-SOG` |
| `remote_export_dir` | `/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/export` |
| `remote_bundle` | `/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/bundle` |
| `remote_database` | `/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3`，首次 bind 时必须不存在 |
| `remote_run_dir` | `/work/home/einlovestan/runs/elementa-sog-001` |
| `remote_conda_env` | 在 SCNet 激活 mlip 后查询到的环境绝对路径 |
| `remote_conda_sh` | 如需要，填写实际 conda 初始化脚本绝对路径；不能猜安装位置 |

资源配置使用已确认的 `qdagnormal`、一张 GPU 与作业主机 CPU 核，不设置 account。模板中的时间、内存、CPU 核数是可调整的作业设置；不能保证默认壁钟时间足以完成全量训练。私钥内容永远不写进 JSON，不把本地配置、密钥或 DPDispatcher 工作目录推送到公开仓库。

默认只回传小型报告；checkpoint 留在 `remote_run_dir/train`，需要下载时显式启用 `download_checkpoints`。远端清理保持关闭，训练目录不放在可删除的 staging 子目录中。

## 6. 按阶段提交：bind → smoke → converge → train → evaluate

`dispatch --config` 默认只离线生成计划；只有加 `--submit` 才连接 SSH、提交 Slurm、监控并取回报告。先查看离线计划中的路径、提交和文件清单。

配置中的 `options` 只属于当前 stage，切换阶段时应同时更新 options，不能把 train 的 epochs 参数留给 bind。Windows 可定义如下辅助函数；它写入 UTF-8 无 BOM JSON：

```powershell
function Set-ElementaStage([string]$Stage, [hashtable]$Options) {
    $ElementaSettings = Get-Content -LiteralPath $ElementaConfig -Raw -Encoding UTF8 | ConvertFrom-Json
    $ElementaSettings.stage = $Stage
    $ElementaSettings.options = $Options
    $ElementaJson = $ElementaSettings | ConvertTo-Json -Depth 30
    [System.IO.File]::WriteAllText($ElementaConfig, $ElementaJson, [System.Text.UTF8Encoding]::new($false))
}
```

### 首次 bind

```powershell
Set-ElementaStage 'bind' @{}
python -m cace_sog_training dispatch --config $ElementaConfig
python -m cace_sog_training dispatch --config $ElementaConfig --submit
```

bind 在获得的 GPU 作业主机 CPU 上顺序校验完整 SHA，包括 bundle 索引快照、完整 manifest/状态文件和所引用的 extxyz，然后复制到新的目标数据库，更新 Windows→SCNet 绝对路径和目标文件时间。它避免重新进行数百万帧的 ASE 解析，但仍需完整读盘、CPU 哈希与 SQLite 工作，因此不是“零 CPU”或瞬时操作。它会占用所申请的 GPU 作业资源，尽管绑定本身不做 GPU 训练。

bind 的数据库复制与再校验也产生额外 I/O，上传完成后仍应为这一步预留时间、空间和作业时限。它不把所有数据装进 RAM，但不会跳过完整字节验证来换取速度。

密封 snapshot 保持不变；训练使用新的已绑定 `remote_database`。成功绑定后不为每轮训练重新 bind。若首次绑定失败，按错误核对传输文件；保留源 bundle，不把半成品目标库当作可用索引。需要人工重试时使用新目标数据库路径，不能覆盖一个已经用于训练的库。

### 小样本检查和初始网格诊断

```powershell
Set-ElementaStage 'smoke' @{batch_size=2; steps=10}
python -m cace_sog_training dispatch --config $ElementaConfig --submit

Set-ElementaStage 'converge' @{n_dl=@(2,1,0.5); samples=4}
python -m cace_sog_training dispatch --config $ElementaConfig --submit
```

smoke 检查优化器更新、力反传、保存/重载和固定小窗口内存；不能代替全量长期稳定性验证。初始网格比较不是训练后模型的收敛证明。查看报告后选择正式的 N_dl；这些小样本限制不影响正式训练清单。

### 全量训练

```powershell
Set-ElementaStage 'train' @{batch_size=8; epochs=50; n_dl=1}
python -m cace_sog_training dispatch --config $ElementaConfig --submit
```

模型与最近/最佳 checkpoint 持久保存在 `remote_run_dir/train`。选定通道的全部 train 帧按轮使用，包括最后不足一批的尾批；val 用于选 best.pt，test 不参加优化。seed、batch、损失与架构/网格等参数在恢复时必须保持一致，epochs 表示总目标轮数。

`resume_policy=auto` 允许用同一配置从 last.pt 继续。DPDispatcher 的重复提交/重试应沿用相同本地工作目录与配置，保留已有状态；训练重试如果找不到可用 checkpoint，应报错，不悄悄从头重算。不要同时启动两个指向同一训练目录的控制任务。监控端中断不等于远端作业已停止，先核对已有作业状态再恢复控制。

### 最终 test

```powershell
Set-ElementaStage 'evaluate' @{batch_size=8; split='test'}
python -m cace_sog_training dispatch --config $ElementaConfig --submit
```

默认使用同一实验 `train/best.pt`，按通道输出能量/原子与力 MAE/RMSE，保存到实验 test 目录并回传小报告。需要训练后网格检查时，使用 converge 的 checkpoint 选项指向远端 `train/best.pt`，并保持原数据、状态与提交版本。

## 7. 后续扩大数据与验证边界

第一次仅用前三通道，不意味着其余数据被删除。后续整理好一致的电子态策略后，可在 Windows 用包含 spin_review 的参数生成新索引和新 bundle，再增量部署其引用文件；训练命令/配置同时选择该通道并记录 spin_policy 和 spin_policy_note。标准几何模型不能自动区分同几何不同磁态。

三维周期、中性来源协议、基础 E/F 质量检查与状态审查仍适用。Windows 预处理和 SCNet 绑定不会认证每帧的显著长程响应。正式训练仍会在 GPU 作业主机上按需解码帧、构建邻居图并向 GPU 送数据；对于密封时相同的通道组合，复用 Windows 已准备的 train-only 基线。改变通道组合可能需要重新准备相应训练统计。

此文档描述代码提供的路线，不代表已经执行了您的真实 SSH、SFTP、SCNet 作业提交或全量 ELEMENTA 训练。先审核 PR，再用真实本地私有配置执行离线计划与小规模 GPU 验证。
