import numpy as np
import matplotlib.pyplot as plt
 

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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------