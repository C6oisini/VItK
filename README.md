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

