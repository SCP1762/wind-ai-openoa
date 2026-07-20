# Wind AI OpenOA

基于 OpenOA 的风电 SCADA 数据处理与功率建模项目。当前版本使用 ENGIE La Haute Borne 示例风电场数据，完成多源数据加载、特征工程、基线模型训练、图表生成和 Word 实验报告输出。

当前默认实验使用 SCADA、ERA5 再分析气象、MERRA2 再分析气象和资产信息。模型使用同一时刻的运行状态与气象背景特征预测同一时刻平均功率，因此它是“运行状态功率估计模型”，适合功率曲线拟合、残差分析和异常检测基线，不应直接表述为未来功率预测模型。

## 报告

项目文档已合并为一个 Word 报告：

```text
reports/wind_ai_openoa_report.docx
```

报告图表由脚本自动生成：

```text
reports/figures/
```

生成图表：

```bash
python scripts/generate_figures.py
```

生成 Word 报告：

```bash
python scripts/generate_report.py
```

如果当前终端没有激活 Conda 环境：

```bash
conda run -n wind-ai-openoa python scripts/generate_figures.py
conda run -n wind-ai-openoa python scripts/generate_report.py
```

## 快速开始

推荐使用 Python 3.11。项目依赖 OpenOA 3.2，`pyproject.toml` 中声明支持 Python 3.9 到 3.12。

### Conda

```bash
conda env create -f environment.yml
conda activate wind-ai-openoa
python scripts/download_data.py
python scripts/check_project.py
python scripts/run_experiment.py
python scripts/generate_figures.py
python scripts/generate_report.py
```

如果当前终端没有激活该环境，也可以使用：

```bash
conda run -n wind-ai-openoa python scripts/check_project.py
conda run -n wind-ai-openoa python scripts/run_experiment.py
conda run -n wind-ai-openoa python scripts/generate_figures.py
conda run -n wind-ai-openoa python scripts/generate_report.py
```

### venv + pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
python scripts/check_project.py
python scripts/run_experiment.py
python scripts/generate_figures.py
python scripts/generate_report.py
```

### Docker Compose

```bash
docker compose build
docker compose run --rm app python scripts/download_data.py
docker compose run --rm app python scripts/check_project.py
docker compose up jupyter
```

Jupyter 默认访问地址为 `http://localhost:8888`。该开发配置未设置 Jupyter Token，仅建议本机使用。

## 常用命令

```bash
make install   # 安装依赖和本地包
make data      # 下载示例数据
make check     # 验证 OpenOA PlantData 加载
make train     # 运行实验训练并生成模型/评估产物
make figures   # 根据数据和训练产物生成图表
make report    # 根据已有图表生成 Word 实验报告
make test      # 运行单元测试
make lab       # 启动 Jupyter Lab
```

## 项目结构

```text
wind-ai-openoa/
├── config/plant_meta.yml          # OpenOA 字段映射
├── data/                          # 原始数据和处理后数据
├── models/                        # 训练后的模型文件
├── notebooks/00_quickstart.ipynb
├── reports/                       # Word 报告、结果表和图表
├── scripts/
│   ├── download_data.py           # 下载并解压 OpenOA 示例数据
│   ├── check_project.py           # 验证数据对象
│   ├── run_experiment.py          # 运行默认实验
│   ├── train_baseline.py          # 训练和评估基线模型
│   ├── generate_figures.py        # 生成报告图表
│   └── generate_report.py         # 组装 Word 报告
├── src/wind_ai/
│   ├── evaluation.py              # 可复用评估指标
│   ├── features.py                # 特征工程
│   ├── openoa_loader.py           # OpenOA 数据加载
│   └── paths.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

## 主要输出

实验脚本会把训练后的模型写入 `models/`：

```text
models/power_baseline.joblib
```

实验结果表统一写入 `reports/tables/`：

```text
reports/tables/power_baseline_metrics.json
reports/tables/power_baseline_predictions.csv
reports/tables/evaluation_by_turbine.csv
reports/tables/evaluation_by_wind_speed_bin.csv
reports/tables/evaluation_by_power_bin.csv
reports/tables/evaluation_by_month.csv
reports/tables/evaluation_by_era5_wind_speed_bin.csv
```

默认训练使用增强特征：

```bash
python scripts/run_experiment.py --feature-set enhanced
```

也可以运行 SCADA-only 对照实验：

```bash
python scripts/run_experiment.py --feature-set scada
```

图表脚本会把所有图片统一写入 `reports/figures/`：

```text
reports/figures/raw_data/
reports/figures/experiment_results/
reports/figures/comparison_analysis/
```

当前图片分组如下：

```text
raw_data/
├── 01_data_source_rows.png
├── 02_monthly_scada_volume.png
├── 03_scada_monthly_mean_wind_speed.png
├── 04_era5_monthly_mean_wind_speed.png
├── 05_merra2_monthly_mean_wind_speed.png
└── 06_wind_speed_power_curve.png

experiment_results/
├── 01_actual_vs_predicted.png
├── 02_prediction_series.png
├── 03_residual_distribution.png
├── 04_mae_by_turbine.png
├── 05_mae_by_scada_wind_speed_bin.png
├── 06_mae_by_power_bin.png
├── 07_monthly_errors.png
└── 08_mae_by_era5_wind_speed_bin.png

comparison_analysis/
├── 01_scada_vs_era5_wind_bin_mae.png
├── 02_turbine_mae_rmse_comparison.png
└── 03_monthly_mae_rmse_comparison.png
```

报告脚本只根据已有结果表和图表生成 Word：

```text
reports/wind_ai_openoa_report.docx
```

## 数据与许可

示例数据来自 OpenOA 官方示例中的 ENGIE La Haute Borne 风电场数据包。数据来源和许可说明见 [DATA_LICENSE.md](DATA_LICENSE.md)。

OpenOA 使用 BSD-3-Clause 许可证；示例数据基于 ENGIE 开放数据并按 Open Licence 2.0 说明发布。发布报告、模型或衍生数据时请保留来源说明。
