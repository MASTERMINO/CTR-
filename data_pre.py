## ========= 导入必要的工具包 ========= ##
import pandas as pd  # 数据分析
import numpy as np   # 科学计算

import matplotlib.pyplot as plt
import seaborn as sns

# 设置显示的最大列数
pd.set_option('display.max_columns', 30)

# 读取数据
dpath_data  = './data/'
dpath_model = './model/'

# 10M
train = pd.read_csv(dpath_data +'train', nrows=10000000, dtype={'id':str})

# 1M
test = pd.read_csv(dpath_data +'test', nrows=1000000, dtype={'id':str})
test.insert(1, 'click', 0)

## ================ 特征工程 =============== ##
# 在进行特征工程的时候 我们不仅需要对训练数据进行处理
# 还需要同时将测试数据同训练数据一起处理
# 使得二者具有相同的数据类型和数据分布
Combined_train_test = pd.concat([train, test], copy=False)
Combined_train_test  = Combined_train_test.drop(['click'], axis=1)
label_train = train['click']
label_test  = test['click']

from datetime import datetime

data = pd.to_datetime(Combined_train_test['hour'], format='%y%m%d%H', exact=False)
Combined_train_test['Day']  = data.apply(lambda x: datetime.strftime(x, "%d"))
Combined_train_test['Hour'] = data.apply(lambda x: datetime.strftime(x, "%H"))

temp  = Combined_train_test.drop(['id', 'hour', 'Day', 'Hour'], axis=1)
Combined_train_test = pd.concat([Combined_train_test['id'], 
                                 Combined_train_test['Day'], 
                                 Combined_train_test['Hour'], temp], axis = 1, ignore_index=False)
del temp

sns.countplot(Combined_train_test.Day)
plt.xlabel('Day')
plt.ylabel('Number of occurrences')

sns.countplot(Combined_train_test.Hour)
plt.xlabel('Hour')
plt.ylabel('Number of occurrences')

# 空缺值填补
Empty_col_names = Combined_train_test.columns
#print((Combined_train_test[Empty_col_names]==0).sum())

# 将数据中特征为0的值替换为空值
Empty_col_names = Combined_train_test.columns
Combined_train_test[Empty_col_names] = Combined_train_test[Empty_col_names].replace(0, np.NaN)
#print((Combined_train_test[Empty_col_names]==0).sum())

## ===== 对类别型特征进行独热编码 ===== ##
categorical_features = ['device_type',
                        'device_conn_type',
                        'C18']

# 数据类型变为object，才能被get_dummies处理
for col in categorical_features:
    Combined_train_test[col] = Combined_train_test[col].astype('object')
    
Combined_train_test_cat = Combined_train_test[categorical_features]
Combined_train_test_cat = pd.get_dummies(Combined_train_test_cat)
Combined_train_test_cat.head(10)

## ===== 对类别型特征进行标签编码 ===== ##
from sklearn.preprocessing import LabelEncoder

# 暂存id
data_id = Combined_train_test['id']

# 删除不需要标签编码的列
label_data = Combined_train_test.drop(['id',
                                       'Day', 
                                       'Hour', 
                                       'device_type',
                                       'device_conn_type', 
                                       'C18'], axis=1)
label_name = []
for col in label_data.columns:
    gle = LabelEncoder()
    label_name.append(col)
    label_data[col] = gle.fit_transform(label_data[col])
    
label_data = pd.concat([label_data[label_name]], axis = 1, ignore_index=False)

## ===== 数据缩放 ===== ##
from sklearn.preprocessing import MinMaxScaler

# 构造输入特征的标准化器
ms = MinMaxScaler()

# 暂存name
feat_names = label_data.columns

# 用训练数据训练模型（得到均值和标准差）：fit
# 并对训练数据进行特征缩放：transform
temp = ms.fit_transform(label_data)
ms_label_data = pd.DataFrame(data = temp, columns = feat_names, index = label_data.index)

del temp
del label_data

## ===== 将训练数据和测试数据分开 ===== ##

# 编码结果数据合并
all_data = pd.concat([data_id, 
                      Combined_train_test['Day'], 
                      Combined_train_test['Hour'], 
                      ms_label_data, 
                      Combined_train_test_cat], axis = 1)

# 分开成训练数据和测试数据                  
train_data_X = all_data[:train.shape[0]]
test_data_X  = all_data[train.shape[0]:]

del train
del test
del all_data
del ms_label_data
del Combined_train_test
del Combined_train_test_cat

# 生成训练数据
train_data = pd.concat([train_data_X, label_train], axis = 1)

# 生成测试数据
test_data = pd.concat([test_data_X, label_test], axis = 1)

## ===== 保存独热编码、标签编码和数据缩放的结果 ===== ##
# 保存训练数据
train_data.to_csv(dpath_data +'FE_train.csv', index=False, header=True)

# 保存测试数据
test_data.to_csv(dpath_data +'FE_test.csv', index=False, header=True)
