
# import numpy as np
# import matplotlib.pyplot as plt
 

# list_arry= [[1,2,3],[3,4,5], [6,7,8]]
# list_matrix= [[1,2,3],[3,4,5], [6,7,8]]
# # list_arry2= [2.3,4]
# # list_arry = list_arry1 + list_arry2
# # np_arry = np.array(list_arry)
# # print(list_arry)
# # print(np_arry+np_arry)
# # print(list_arry)
# print(list_matrix[1],[2])


# -------------------Buil-----------------

# arranged_date = np.arange(0, 10)
# arranged_date = np.zeros(0).astype('int')
# arranged_date = np.ones(10).astype('int')
# random_data = np.random.randn(1000)

# # print(arranged_date)
# # print(random_data)
# plt.plot(random_data, 'o')
# plt.show
# print(random_data)


# ------------------Data goturme-------------------

# data = np.loadtxt(r'......',delimiter=';',dtype='str')

# ---------------data save-----------------
# data = np.loadtxt(r'',delimiter=';',dtype='str')
# new_data = data[:, :2]
# np.savetxt(......,new_data, delimiter';)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# import numpy as np

# x = np.linspace(0, 1, 201)
# y = np.random.random(201)

# header = "X-Column, Y-Column\n"
# header += "This is a second line"
# f = open('AD_data.dat', 'wb')
# np.savetxt(f, [], header=header)
# for i in range(201):
#     data = np.column_stack((x[i], y[i]))
#     np.savetxt(f, data)

# f.close()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

ses = pd.read_csv('/home/orkhan/Downloads/Vowel_Data.csv',sep=',',header=0)
print(ses.shape)
print(ses.head)

x= ses.iloc[:,3:-1]
y = ses.iloc[:,-1]

lb = LabelBinarizer()
lb.fit(y)
ybin =  lb.transform(y)


x_train,x_test,y_train,y_test=train_test_split(x,ybin,train_size=0.8,test_size=0.2,shuffle=True,stratify=y,random_state=42)

knn=KNeighborsClassifier(n_neighbors=2,n_jobs=-1)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
print(y_test[8])
print(lb.inverse_transform(y_test[8].reshape(1,-1)))
print(knn.predict(x_test.iloc[8].values.reshape(1,-1)))
print(accuracy_score(y_test,y_pred))

n = list(range(1,5))
acc = list()

for i in n:
    knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    acc.append(accuracy)
    
print(acc)
import matplotlib.pyplot as plt

ciz=plt
ciz.plot(n,acc)
ciz.scatter(n,acc)
ciz.show()