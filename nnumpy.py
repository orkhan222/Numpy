import numpy as np
import time

# start_time = time.time()
# A = np.arange(100000)

# A ** 2
# Time_1 = time.time() - start_time

# start_time2 = time.time()
# L = range(10000)
# [i ** 2 for i in L]
# Time_2 = time.time() - start_time2

# print(Time_2 / Time_1)



# x = np.array([4,5,6])

# print(x)
# print(type(x))
# print(x.shape)
# print(x.size)
# print(x.ndim)
# print(x.nbytes)
# print(x.dtype)

# y = np.array([[1,2,3], [4,5,6]])

# print(y)
# print(type(y))
# print(y.shape)
# print(y.size)
# print(y.ndim)
# print(y.nbytes)
# print(y.dtype)


# z = np.array([[[1,2],[3,4]],[[1,2],3,4]])

# print(z)
# print(type(z))
# print(z.shape)
# print(z.size)
# print(z.ndim)
# print(z.nbytes)
# print(z.dtype)

# x = np.array

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

irisData = load_iris()

# Create feature and target arrays
X = irisData.data
y = irisData.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size = 0.2, random_state=42)

neighbors = np.arange(1, 50)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train, y_train)
	
	# Compute training and test data accuracy
	train_accuracy[i] = knn.score(X_train, y_train)
	test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


