import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

data = pd.read_csv("C:/Users/20386/Desktop/iris/iris_classification_BPNeuralNetwork-master/sklearn数据集/iris.csv", encoding='utf-8')
x = np.array(data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']])
real = np.array(data['species'])

# 第1种实现：KMeans算法
# kms = KMeans(n_clusters=1)  # 传入要分类的数目
# y = kms.fit_predict(x)

# 第2种实现：DBSCAN算法
print("DBSCAN算法实现：")
dbscan = DBSCAN(eps=0.5, min_samples=13)
dbscan.fit(x)
y = dbscan.labels_

count = 0
for i in range(len(real)):
    if abs(int(y[i])) == abs(int(real[i])):
        count = count + 1
print('正确：' + str(count))
acc = round(count / len(real), 4) * 100
print('正确率：' + str(acc) + '%')

