# vitk clustering benchmarks

轻量说明，帮助快速复现实验、绘图并了解代码结构。

## 环境准备
- Python 3.9+（项目依赖 >=3.12，但 3.9 也可运行当前脚本）
- 安装依赖：`python3 -m pip install numpy scipy scikit-learn matplotlib ucimlrepo`

## 目录
- `experiments.py`：主实验脚本，支持合成数据、UCI 小数据集、`datasets/` 下的 ARFF 数据。
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

