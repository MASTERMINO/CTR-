## ========= 导入必要的工具包 ========= ##
import pandas as pd  # 数据分析
import numpy as np   # 科学计算

import matplotlib.pyplot as plt
import seaborn as sns

# 设置显示的最大列数
pd.set_option('display.max_columns', 30)

# 读取数据
dpath_data  = './data/'

# 10M
train = pd.read_csv(dpath_data +'train', nrows=10000000, dtype={'id':str})

# 样本数和特征数
train.shape

# 特征的数据类型
train.info()
# 各属性的统计特性
train.describe()

# 1.类别型特征分布
for col in train.columns:
    print('='*25)
    print('%s属性的不同取值和出现的次数' %col)
    print(Combined_train_test[col].value_counts().shape)
    print(Combined_train_test[col].value_counts())
 
# 2.数值特征的直方图
numerical_features = ['C1',
                      'banner_pos',
                      'device_type',
                      'device_conn_type',
                      'C14',
                      'C15',
                      'C16',
                      'C17',
                      'C18',
                      'C19',
                      'C20',
                      'C21']

for col in numerical_features:
    train.hist(col)
    
# click 分布，看看各类样本分布是否均衡
sns.countplot(train.click)
plt.xlabel('click')
plt.ylabel('Number of occurrences')

for column in ["C1",'C14','C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21' ]:
    plt.figure()
    sns.countplot(train[column])
    
sns.countplot(x= 'C1', hue='click', data=train)
sns.countplot(x= 'banner_pos', hue='click', data=train)
sns.countplot(x= 'site_domain', hue='click', data=train)
sns.countplot(x= 'app_category', hue='click', data=train)
sns.countplot(x= 'app_id', hue='click', data=train)
sns.countplot(x= 'device_type', hue='click', data=train)
sns.countplot(x= 'device_conn_type', hue='click', data=train)
sns.countplot(x= 'C14', hue='click', data=train)
sns.countplot(x= 'C15', hue='click', data=train)

# 3.特征与目标之间的关系
## 3.1 一个月中的第几天和用户点击之间的关系
fig,ax = plt.subplots()
sns.barplot(data=train[['Day', 'click']], x="Day", y="click")
ax.set(title="Day distribution of counts")

## 3.2 一天中的第几个小时和用户点击之间的关系
fig,ax = plt.subplots()
sns.barplot(data=train[['Hour', 'click']], x="Hour", y="click")
ax.set(title="Hour distribution of counts")

## 3.3 广告位置和用户点击之间的关系
fig,ax = plt.subplots()
sns.barplot(data=train[['banner_pos', 'click']], x="banner_pos", y="click")
ax.set(title="Banner_pos distribution of counts")

## 3.4 站点类别和用户点击之间的关系
fig,ax = plt.subplots()
sns.barplot(data=train[['site_category', 'click']], x="site_category", y="click")
ax.set(title="Site_category distribution of counts")

## 3.5 APP类别和用户点击之间的关系
fig,ax = plt.subplots()
sns.barplot(data=train[['app_category', 'click']], x="app_category", y="click")
ax.set(title="App_category distribution of counts")

## 3.6 Device类型和用户点击之间的关系
fig,ax = plt.subplots()
sns.barplot(data=train[['device_type', 'click']], x="device_type", y="click")
ax.set(title="Device_type distribution of counts")

## 3.7 Device接入类型和用户点击之间的关系
fig,ax = plt.subplots()
sns.barplot(data=train[['device_conn_type', 'click']], x="device_conn_type", y="click")
ax.set(title="Device_conn_type distribution of counts")

# 特征之间的相关系数
# get the names of all the columns
cols = train.columns 

# Calculates pearson co-efficient for all combinations，通常认为相关系数大于0.5的为强相关
feat_corr = train.corr().abs()

plt.subplots(figsize=(12, 8))
sns.heatmap(feat_corr, annot=True)

# Mask unimportant features
sns.heatmap(feat_corr, mask=feat_corr < 1, cbar=False)
plt.show()

corrMatt = train[['C1',
                  'banner_pos',
                  'device_type',
                  'device_conn_type',
                  'click']].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
sns.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True)
# 设备类型和C1高度相关

# Set the threshold to select only highly correlated attributes
threshold = 0.5
# List of pairs along with correlation above threshold
corr_list = []
# size = data.shape[1]
size = feat_corr.shape[0]

# Search for the highly correlated pairs
for i in range(0, size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (feat_corr.iloc[i,j] >= threshold and feat_corr.iloc[i,j] < 1) or (feat_corr.iloc[i,j] < 0 and feat_corr.iloc[i,j] <= -threshold):
            corr_list.append([feat_corr.iloc[i,j],i,j]) #store correlation and columns index

# Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

# Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))
    
