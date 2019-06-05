# CTR-
广告点击率(CTR)是互联网计算广告中的关键环节， 预估准确性直接影响公司广告收入。

数据探索：data_exploration.py
特征工程：data_pre.py
模型训练及预测: CTR_model_ffm.py

操作系统：macOS Mojave 10.14.5
内存：16G
Python: 3.7.2

# 数据描述： 数据共包含 11 天的数据，其中 10 天为训练数据 train，1 天为测试数据 test。

#train: 训练数据，包括 10 天的 CTR 数据，数据按时间顺序排列。被点击的样本和没被点击的样本根据不同的策略已经进行了下采样。 
#test ：测试数据，1 天的广告数据，用于测试模型的预测。 
#Submission.csv：提交结果文件样例。
