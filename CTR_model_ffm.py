### ============= 模型训练 ============= ###

# FE_train.csv --> ffm_train.txt + ffm_valid.txt
# FE_test.csv --> ffm_test.txt

### ========= 设置数据存取路径 ========== ###
**************** 分割线 *******************
# raw data (for read)
train = dpath_data +"FE_train.csv"
test  = dpath_data +"FE_test.csv"

# input
train_ffm = dpath_data +'ffm_train.txt'
valid_ffm = dpath_data +'ffm_valid.txt'
test_ffm  = dpath_data +'ffm_test.txt'

# output
model_fit  = dpath_model +'model_fit.out'
model_pred = dpath_model +'model_pred.txt'

# submissions
model_sub = dpath_data +"Submission.csv"
**************** 分割线 *******************

cols = list(train_data.columns.drop(['id', 'click']))

# train set
df_train = pd.read_csv(train)

# test set
df_test = pd.read_csv(test)
# add a placeholder
df_test['click'] = -1

##----- 合并训练集和测试集 -----##
n_train = len(df_train)
df = df_train.append(df_test)

del df_train
del df_test

## ============= 将数据转化从ffm模型所需要的数据格式 ============ ##
def convert_to_ffm(df, numerics, categories, features, label, n_train, train_size):
    """
    :function: generation of train/valid/test set format as libffm 
    
    :parameters:
        :df, pandas dataframe include raw data of train and test.
        :numerics, name list of numerical features.
        :categories, name list of categorical features.
        :features, name list of all features.
        :Label, name of label in the df.
        :n_train, number of training samples.
        :train_size, the ratio of train_valid split.
    """
    catdict = {}
    # Flagging categorical and numerical fields
    for x in numerics:
        catdict[x] = 0
    for x in categories:
        catdict[x] = 1
    
    nrows = df.shape[0]
    
    # samples' number of train
    train_sample = n_train * train_size
    
    with open(train_ffm, "w") as file_train, \
         open(valid_ffm, "w") as file_valid, \
         open(test_ffm, "w")  as file_test:
        
        # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str(int(datarow[label]))
            # For  fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if(catdict[x]==0):  # numerical
                    datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
                else:  # categorical
                    datastring = datastring + " "+str(i)+":"+ str(int(datarow[x]))+":1"
            datastring += '\n'
            
            if n < train_sample:
                file_train.write(datastring)
            elif n < n_train:
                file_valid.write(datastring)
            else:
                file_test.write(datastring)
                
# 开始转化数据格式并写入对应的文件路径       
convert_to_ffm(df, numerics=[], categories=cols, features=cols, label='click', n_train=n_train, train_size=0.8)
  
import xlearn as xl

## Training task
# Use field-aware factorization machine (ffm)
ffm_model = xl.create_ffm()   

# Set the path of training dataset
ffm_model.setTrain(train_ffm)

# Set the path of validation dataset
ffm_model.setValidate(valid_ffm)

#  Parameters:
#  0. task: binary classification
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: accuracy

param = {'task': 'binary',
         'lr': 0.02, 
         'lambda': 0.0002,
         'epoch': 10,
         'opt': 'adagrad',
         'metric':'acc'
         }
         
# Start to train
# The trained model will be stored in fp_model_ffm
# 由于数据量较大，模型训练需要较长时间，PC算力有限。
ffm_model.fit(param, model_fit)

## Prediction task
# Set the path of test dataset
ffm_model.setTest(test_ffm)

# Convert output to 0-1
ffm_model.setSigmoid()

## Start to predict
# The output result will be stored in fp_pred_ffm
ffm_model.predict(model_fit, model_pred)

## ===== 生成提交文件 ===== ##

y_pred_fm = np.loadtxt(model_pred)
df_test = pd.read_csv(test, dtype={'id':str})
df_test['click'] = y_pred_fm

with open(model_sub, 'w') as f: 
    df_test.to_csv(f, columns=['id', 'click'], header=True, index=False)
