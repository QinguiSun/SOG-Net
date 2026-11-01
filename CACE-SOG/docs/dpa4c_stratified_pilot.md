# DPA4C 多元素诊断数据

此工具用于检查多元素能量对齐，不是全量训练入口，也不是无偏基准集。它在已绑定的 SQLite 索引中分别顺序扫描 `sog_candidates` 的全部 train 和 val 记录，对每个元素按组件保留固定数量的确定性哈希样本，再从组件中确定性选择帧，去重后按原 split 写入新的 DeePMD 数据目录。组件是防止数据泄漏的保守划分单位，不一定等于物理母结构。它不会读取 test、重新划分数据或修改源文件；选中的几何仍受原组件划分约束。每个 split 最多保留 `118 × per-element` 帧，实际数量通常更少。选择过程刻意增加元素覆盖，并不保证组成、母结构、力尺度或数据分布具有代表性。

`pilot_status.json` 记录选中数量、元素和训练组成矩阵的秩。满秩也不证明充分覆盖或拟合稳定；若秩不足，不要把结果当作已经校准的元素偏置。`frames.jsonl` 保存来源、split、组件和原子排列；`pilot_files.json` 包含输出文件的大小与 SHA-256。数据保持原 DFT 能量和力，不应用训练集基线。`[0,1]` 是显式的模型输入假设，并非确认的 DFT 电子态。

## SCNet 准备命令

这项准备只需要 CPU，但在当前 SCNet 授权范围内由非独占单 GPU 作业提供 CPU；作业会按平台规则计费。它要扫描约 398 万条 train/val 索引记录和随机读取选中帧，实际时间取决于共享存储性能。下面的时间上限是起点，不保证完成；中断时保留 partial 目录，用新路径重试。请使用完成 bind 且路径未改变的索引。

```bash
export DPA_ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c

sbatch --partition=qdagnormal --nodes=1 --gres=gpu:1 \
  --cpus-per-task=2 --mem=8G --time=02:00:00 \
  --job-name=dpa4c-pilot-prep \
  --output="$DPA_ROOT/dpa4c-pilot-prep-%j.out" \
  --export=ALL <<'JOB'
#!/bin/bash -l
set -e
DPA_ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
DPA_INDEX=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3
DPA_PILOT="$DPA_ROOT/stratified-pilot-$SLURM_JOB_ID"

cd /work/home/einlovestan/projects/SOG-Net/CACE-SOG
echo "CODE_COMMIT=$(git rev-parse HEAD)"
echo "PILOT_DIRECTORY=$DPA_PILOT"

export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

"$HOME/envs/mlip/bin/python" -u -m dpa4c_sog.stratified_pilot \
  --database "$DPA_INDEX" \
  --release-config "$DPA_ROOT/DPA4C-Neo-OMol25-100M-v20260820.json" \
  --output "$DPA_PILOT" \
  --per-element 4 --seed 42 --steps 200 \
  --assume-charge-spin 0 1

echo "PILOT_PREPARATION_COMPLETE"
JOB
```

结束后用 `sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode` 和 `cat "$DPA_ROOT/dpa4c-pilot-prep-<jobid>.out"` 检查状态，确认 `PILOT_PREPARATION_COMPLETE`、`pilot_status.json` 的 `status` 和训练组成秩。先审查清单和数据，再决定具体的 train-only 能量对齐试验；**不要直接把 seal 得到的 DFT 元素基线写成 DPA4C 的输出偏置**，也不要对 val 或 test 拟合校准参数。

该工具没有实现受限内存的完整 DeepMD 数据加载或启动统计，也没有实现/验证周期 SOG；这些仍是正式全量训练的独立前置工作。
