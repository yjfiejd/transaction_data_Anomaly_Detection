
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.chdir('/Users/a1/Downloads/百度云盘/机器学习算法配套案例实战/逻辑回归-信用卡欺诈检测')


# In[3]:


data = pd.read_csv("creditcard.csv")
data.head()


# In[4]:


#data.describe


# In[5]:


#看正负样本的比例，用value_counts来统计
count_class = pd.value_counts(data['Class']).sort_index() #默认按行标升序排列
#count_class.sort_index(ascending = False) 降序排列
count_class


# In[6]:


#查看count_class的类型, 这里取了pandas中dataFrame中的一列，所以为series格式
type(count_class)


# In[7]:


#用pandas画简单的图
count_class.plot(kind='bar')
plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')


# ## 数据预处理
# ### 1）正常样本的个数 与 异常样本的个数如下图， 样本数据不均衡该怎么办？
#     
#     #方法一：下采样 - 如上图减少0样本的个数，让它与1样本一样的少， 看看哪种更好？
#     #方法二：过采样 - 让1号样本变多，让它与0号样本一样的多

# ### 2）某些特征数值太大，比如这里的normAmount，保证特征之间的分布差异差不多
#     #归一化
#     #标准化

# In[8]:


#预处理
from sklearn.preprocessing import StandardScaler
#data中加上一列，把原来的Amount转换为新的特征
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1)) #注意reshape python3.0的用法
#删除原来的没用的特征，用drop(['',''], axis = 1) 表示列
data = data.drop(['Time','Amount'], axis = 1)
data.head()


# ### 1.1下采样策略 ： 使得0和1的样本一样少

# In[9]:


# 构造特征数据，：表示选择所有的行，x列选择中不包含‘Class’这一列label， y列中只选择包含label这一列
X = data.iloc[:, data.columns != 'Class']
Y = data.iloc[:, data.columns == 'Class']

#统计Class=1的样本有多少，然后在让0样本数量与1样本数量一致
number_records_fraud = len(data[data.Class == 1]) #统计异常样本个数
#取所有1样本（少）索引值，再把所有的索引值组成新的array
fraud_indices = np.array(data[data.Class == 1].index) 
#取所有0样本（多）的索引值
normal_indices = np.array(data[data.Class == 0].index)
#从0样本中随机取，让取出的0样本的个数等于1样本个数; np.random.choice()用法：http://blog.csdn.net/autoliuweijie/article/details/51982514
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
#随机取出来，取得里面值的index值
random_normal_indices = np.array(random_normal_indices)

#组合拼接，把index都存着
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

#利用组合号的index，从data取出数据
under_sample_data = data.iloc[under_sample_indices,:] #取出需要的列，取出所有行

#获得列新的数据集合
X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
Y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']

#打印新的数据集合，看下正样本与负样本均衡了么
print("Percentage of normal transction:", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of normal transction:", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of new dataset", len(under_sample_data))
fraud_indices #返回的是所有1样本的索引


# ### 交叉验证
#     #先洗牌
#     #再切分

# In[10]:


from sklearn.cross_validation import train_test_split

#对整个原始数据集切分,查看如何使用train_test_split
#http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

print("**********我是分割线**********")
print("Number transaction train dataset:", len(X_train))
print("Number transaction test dataset:", len(X_test))
print("Total number of transaction:", len(X_train)+len(X_test))


# #### 这里为什么对原始数据集也进行切分？ → 后续测试model时使用
#     #回答：因为下采样数据集切分，只是为了得到合适的model，真正验证时候，需要把model放到原始的测试集中，因为下采样的测试集第一小，第二分部规则不一定与原始数据集合一样

# In[11]:


#对下采样的后的数据集进行切分
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, Y_undersample, test_size = 0.3, random_state = 0)

print("**********我是分割线**********")
print("Number transaction train dataset:", len(X_train_undersample))
print("Number transaction test dataset:", len(X_test_undersample))
print("Total number of transaction:", len(X_train_undersample) + len(X_test_undersample))


# ### 建模操作- 逻辑回归模型
#     #模型评估方法，不能仅仅依靠精度，特别是样本不均衡的情况下
#     #需要用recall来制定model的评估标准
#     #Recall = TP/TP+FN  (True positive/ True positive + False nagative)

# In[12]:


#导入机器学习建模库，逻辑回归; 交叉验证（几份）&结果；混淆矩阵
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report


# In[13]:


#切分完数据集后，进行交叉验证
def printing_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(len(y_train_data), 5, shuffle=False)
    
    #正则化惩罚项,参数: 希望model浮动小，泛化能力更强，更能避免过拟合A,B，model
    #L2正则化+1/2 w平方，看谁的loss小，惩罚力度可以用λ调节
    #L1正则化，加|w|绝对值
    c_param_range = [0.01, 0.1, 1, 10, 100] #这个就是λ
    
    #可视化展示
    results_table = pd.DataFrame(index = range(len(c_param_range), 2), columns = ['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range
    
    j = 0
    for c_param in c_param_range:
        print("---------------------------------")
        print('C parameter:', c_param)
        print("---------------------------------")
        print('')
        
        recall_accs = []
        ## enumerate 转化为枚举值，iteration：枚举编号，indices：枚举值
        #http://blog.csdn.net/churximi/article/details/51648388
        for iteration, indices in enumerate(fold, start=1):#交叉验证，每次取不同的训练集，测试集
            
            #选择逻辑回归模型, 实例化模型
            lr = LogisticRegression(C = c_param, penalty = 'l1') #传进参数λ，选则l1正则化，也可以选l2
            
            #进行训练fit
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0],:].values.ravel())
            
            #进行预测再train里面的validation测试集中
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)
            
            #计算召回率，recall
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration', iteration, ':recall score = ', recall_acc)
            
            #recall的平均值
        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('mean recall score', np.mean(recall_accs))
        print('')

    #best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    best_c = results_table
    best_c.dtypes.eq(object) #因为best_c中的mean recall score 值的类型为‘object’,需要转换为'float'，这里找出类型为‘object’的列名，返回index
    new = best_c.columns[best_c.dtypes.eq(object)] #利用返回的列名，找出那一列，pandas.columns，
    best_c[new] = best_c[new].apply(pd.to_numeric, errors = 'coerce', axis=0) #对该列进行操作，把‘object’转换为‘float’类型
    best_c
    #通过idxmax()函数取得‘Mean recall score’中值最大的行号，通过行号找到这行，然后取这行列名为‘C_parameter’的值
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

    #如何找到值最大时候的索引值

    #选择最合适的C参数
    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    #print (best_c.dtypes)
    #print(best_c.dtypes.eq(object))
    print(new)
    #print(best_c.dtypes)
    
    return best_c


# In[14]:


best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)


# ### 混淆矩阵
#     #里面有预测值与真实值可以求一些指标, Recall值TP/TP+FN，精度值TP+FN/TP+FN+TN+FP
#     #在下采样中，recall值可以满足要求，但是当模型用在整体数据集中容易误杀太多，精度会降低，那么如何解决呢？
#     #要不要试一试 过采样？
#     #如果我啥都不用，用原始的数据，那模型效果怎样呢？

# In[15]:


#混淆矩阵
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html 官方画图实例
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[16]:


import itertools
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(X_test_undersample.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# In[17]:


lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# In[18]:


best_c = printing_Kfold_scores(X_train,y_train)


# In[19]:


lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

plt.figure(figsize=(10,10))

j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:,1] > i
    
    plt.subplot(3,3,j)
    j += 1
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_undersample,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s'%i) 


# In[20]:


import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[27]:


#导入数据
credit_cards = pd.read_csv('creditcard.csv')
#获取列名
columns = credit_cards.columns
#删除最后一列标签值 .delete()，取得特征值
features_columns = columns.delete(len(columns)-1)
credit_cards.head()


# In[24]:


print(len(features_columns))
print(len(columns))


# In[30]:


#看一下是不是少了最后一列
features = credit_cards[features_columns]
features.head()


# In[31]:


#所以我们有了features和labels
labels = credit_cards['Class']


# In[32]:


#切分数据集合,进行抽样，分为训练集&测试集
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)


# In[33]:


#采用SMOTE算法, 进行过采样
#Synthetic Minority Oversampling Technique
#http://blog.csdn.net/Yaphat/article/details/52463304?locationNum=7

oversampler = SMOTE(random_state=0)#实例化一个模型，让每次随机生成一样的数据：random_state=0
os_features, os_labels = oversampler.fit_sample(features_train, labels_train) #把训练集中的features&labels传入


# In[44]:


#看训练集中，label等于1的样本有多少个
len(os_labels[os_labels == 1])


# In[43]:


len(os_labels[os_labels == 0])


# In[36]:


#把array格式转换为DataFrame格式
os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)
#用过采样看模型表现 （数据越多，模型越好, 相当于N大）还有另一种方式就是少用过高的特征转换
best_c = printing_Kfold_scores(os_features, os_labels)

