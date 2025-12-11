# vitk clustering benchmarks

轻量说明，帮助快速复现实验、绘图并了解代码结构。

## 环境准备
- Python 3.9+（项目依赖 >=3.12，但 3.9 也可运行当前脚本）
- 安装依赖：`python3 -m pip install -r requirements.txt` 如果没有该文件，可用 `python3 -m pip install numpy scipy scikit-learn matplotlib ucimlrepo`

## 目录
- `experiments.py`：主实验脚本，支持合成数据、UCI 小数据集、`datasets/` 下的 ARFF 数据。
- `tk.py`：TMM 实现（改进的稳健 t-kmeans），默认 k-means++ 初始化、log 安全计算，参数在 `experiments.py` 中调用时设定。
- `plot_csv.py`：读取 `csv/plot.csv` 绘制单数据集对比图，突出每个指标的最佳模型（黑色）。
- `datasets/`：`artificial/` 与 `real-world/` ARFF 数据集。
- `csv/`：实验输出。`experiments_*.csv` 为完整结果；`plot.csv` 为绘图子集。

## 运行实验
数据集下载地址：https://github.com/deric/clustering-benchmark/tree/master

示例命令（已用于近期结果）：
```
python3 experiments.py \
  --use-datasets-folder \
  --max-samples 4000 \
  --scale standard \
  --csv-dir ./csv
```
要加入小型真实数据集可再加 `--include-real`；若只想合成数据，省略 `--use-datasets-folder`。

关键参数：
- `--max-samples`：大于该阈值的 ARFF 自动跳过。
- `--scale {standard,robust,none}`：默认合成用 robust，其余 standard。
- `--csv-dir`：结果输出目录，时间戳命名。

## 结果汇总与绘图
1. 从最新实验 CSV 选出 vitk 领先的数据集并生成绘图专用表：`plot.csv`（已准备好）。
2. 出图（需 matplotlib）：
```
python3 plot_csv.py csv/plot.csv plots
```
输出到 `plots/`，每个数据集一张 PNG。柱序固定为：K-means, GMM, TMM, VItK(Ours)；每个指标最佳柱为黑色，其余灰色。

## 当前 Iris 结果（覆盖写入）
```
Dataset: Iris | sklearn iris, 150 samples, 4 dims, 3 classes | k=3
k-means  ARI 0.4328  NMI 0.5896  MSE 1.2735  WB 0.4671  Time 0.002  notes n_iter=4
GMM      ARI 0.5165  NMI 0.6571  MSE 1.3692  WB 0.5207  Time 0.017  notes converged=True, n_iter=11
vitk     ARI 0.6199  NMI 0.6588  MSE 0.9347  WB 0.3117  Time 0.028  notes iters=23
tkmeans  ARI 0.6101  NMI 0.6526  MSE 0.9424  WB 0.3287  Time 0.027  notes iters=40
```
（已写入 `csv/experiments_20251211-111050.csv` 与 `csv/plot.csv`。）

## 常见问题
- **TMM 结果异常**：已通过 k-means++ 初始化、log1p/clip 等提升稳定性，如需更稳，可调 `nu_fixed`、`max_iter`、`tol`。
- **数据过大**：调高 `--max-samples` 或过滤大文件；字符串特征暂不支持（如 `yeast.arff`）。
- **标题遮挡/间距**：`plot_csv.py` 已统一版式，如需微调可修改 `suptitle` 与 `tight_layout` 参数。

## 快速流程
数据加载 → 预处理/标准化 → 运行 k-means / GMM / VItK / TMM → 计算 ARI/NMI/MSE/WB → 写 CSV → 选取优势集生成 `plot.csv` → `plot_csv.py` 绘图。
