# 清风溯源队模型算法程序

本材料只包含两条模型算法链路：A—H匿名区域的可控数据模拟、训练与有真值评估，以及北京前门—天坛—永定门内三站真实历史数据的短时浓度预测。

## 程序组成

- `src/data/`：模拟数据生成模块。
- `src/models/`：LSTM浓度预测与风场反向轨迹模型。
- `src/analysis/`：双向LSTM成因分类和浓度梯度分析模块。
- `scripts/generate_synthetic_dataset.py`：重新生成固定随机种子的模拟数据集。
- `scripts/run_simulated_analysis.py`：训练模拟数据模型并计算预测、分类与候选源排序指标。
- `scripts/train_beijing_neighborhood.py`：使用处理后的北京三站小时数据重新训练单站、多站LSTM模型。
- `scripts/predict_beijing_neighborhood.py`：加载北京三站模型，输出前门站未来1—6 h PM₂.₅预测。

## 数据作用

- `data/simulated/v1/`：A—H匿名区域的可控模拟数据，包含八个监测点、气象、十二个虚拟源、源活动、污染事件和源贡献真值，用于模型训练、算法联调和有真值评估。该数据不代表真实监测结果。
- `data/beijing_neighborhood_2019/`：2019年第一季度北京前门、天坛和永定门内三站的处理后小时数据、数据质量统计、站点距离、模型输入、测试预测和来源清单，用于邻域级真实数据预测验证。三站最大间距约2.56 km，该链路只验证浓度预测，不输出真实成因准确率或污染责任结论。

## 模型作用

- `models/simulated_v1/lstm_pm25_predictor.pt`：基于模拟数据训练的未来1—6 h PM₂.₅预测模型。
- `models/simulated_v1/cause_classifier.pt`：基于模拟事件标签训练的六类成因分类模型。
- `data/beijing_neighborhood_2019/models/`：北京三站单站和多站LSTM的三个随机种子模型，用于离线集成预测。

## 环境与运行

使用Conda创建环境：

```bash
conda env create -f environment.yml
conda activate air-pollution
```

也可在已有Python 3.10环境中安装：

```bash
pip install -r requirements.txt
```

重新生成模拟数据并训练模型：

```bash
python scripts/generate_synthetic_dataset.py
python scripts/run_simulated_analysis.py
```

使用北京真实数据重新训练模型：

```bash
python scripts/train_beijing_neighborhood.py
```

运行北京三站离线预测：

```bash
python scripts/predict_beijing_neighborhood.py \
  --bundle data/beijing_neighborhood_2019 \
  --input data/beijing_neighborhood_2019/example_input.csv \
  --model multi_station \
  --output beijing_forecast.json
```

北京模型输入为连续24个整点小时，字段包括`timestamp`、`pm25`、`pm25_天坛`和`pm25_永定门内`；时间必须带明确时区。输出为前门站未来1—6 h的PM₂.₅点预测。省略`--model multi_station`时，程序按照验证集预先锁定的规则使用配置文件中的默认基线。
