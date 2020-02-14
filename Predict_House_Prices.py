import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#Load data tu file csv
train = pd.read_csv("House-dataset/train.csv")
test = pd.read_csv("House-dataset/test.csv")

#Loai bo cac ten thuoc tinh 
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# Phan tich tuong quan du lieu bang heatmap
corr = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=1, square=True)
# plt.show()

#Ve heatmap ung voi 10 features anh huong nhat den house prices + chi so tuong quan
k = 10
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()

#Lay ra cac features anh huong nhat den house price dua vao heatmap
features = corr['SalePrice'].sort_values(ascending=False)
features = features[abs(features) >= 0.52] 

#Kiem tra data co gia tri NaN trong train va test
select = []
for i in features[features.index.values != 'SalePrice'].index.values:
    if(test[i].isnull().sum() > 0):
        select.append(i)
for v in select:
    test[v] = test[v].fillna(0)

# Xay dung model
pre_val = [v for v in features.index.values if v != 'SalePrice']
target_val = 'SalePrice'
x_train = train[pre_val]
y_train = train[target_val]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

#Thu du doan gia cua 5 ngoi nha bat ky trong test.csv
x_test = test[pre_val]
y_pre = model.predict(x_test[1000:1005])
count = 0
for i in y_pre:
    count += 1
    print('Gia ngoi nha thu ' + str(count) + ' la: ' + str(i) + '$')
