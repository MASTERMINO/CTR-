# CTR-
广告点击率(CTR)是互联网计算广告中的关键环节， 预估准确性直接影响公司广告收入。

数据探索：data_exploration.py
特征工程：data_pre.py
模型训练及预测: CTR_model_ffm

操作系统：macOS Mojave 10.14.5
内存：16G
Python: 3.7.2

# 数据描述： 数据共包含 11 天的数据，其中 10 天为训练数据 train，1 天为测试数据 test。

（1） 文件说明
train: 训练数据，包括 10 天的 CTR 数据，数据按时间顺序排列。被点击的样本和没被点击的样本根据不同的策略已经进行了下采样。 
test ：测试数据，1 天的广告数据，用于测试模型的预测。 
sampleSubmission.csv：提交结果文件样例。

（2） 字段说明 
id: ad identifier （广告 ID） 
click: 0/1 for non-click/click （是否被点击，其中 0 为不被点击，1 为不被点击，此列为目 标变量）
hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC. （时间）
C1                    （类别型变量）
banner_pos            （广告位置）
site_id               （站点 ID） 
site_domain           （站点领域）
site_category         （站点类别）
app_id                （APP ID）
app_domai C14-C21      anonymized 
categorical variables （类别型变量）
