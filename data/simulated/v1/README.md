# A-H匿名区域小尺度空气污染机制数据集

本数据集是**纯模拟数据**，用于模型训练、管线联调和溯源算法合成基准，
不得作为真实案例、真实监测结果或污染责任认定依据。

## 数据规模

- 空气质量观测：103,680 行
- 气象记录：2,160 行
- 污染源活动：155,520 行
- 污染事件：144 个
- 源贡献真值：39,900 行

## 文件

- `stations.csv`：A-H八个随机排布监测点及相对坐标。
- `sources.csv`：12个虚拟污染源及基础排放谱。
- `weather.csv.gz`：逐小时气象和局地风场。
- `source_activity.csv.gz`：每个源的高分辨率活动与逐污染物排放率。
- `air_quality.csv.gz`：多站观测值、无噪声真值、事件标签和质量标志。
- `events.csv`：事件起止、峰值、主次源和成因真值。
- `source_contribution_truth.csv.gz`：事件峰值时各站各源贡献真值。
- `model_input_hourly.csv.gz`：兼容现有 LSTM 的中心站小时建模表。
- `metadata.json`：参数、单位、随机种子、行数和来源说明。

## 关键字段

`air_quality.csv.gz` 中 `pm25` 等字段表示含噪声观测，
`pm25_true` 等字段表示模拟器内部无噪声真值。`cause_label` 在整个事件期间
填写；`model_input_hourly.csv.gz` 只在事件峰值小时填写标签，使每个事件只
形成一个分类样本。

`qc_flag` 可能为 `valid`、`missing`、`sensor_spike` 或 `sensor_drift`。
所有文件均通过 `is_simulated=true` 明确标记模拟属性。
