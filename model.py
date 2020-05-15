from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import sklearn
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data

y = iris.target

knn = KNeighborsClassifier(n_neighbors = 6)

knn.fit(x,y)

x_new = [1,3,4,5]



saved_model = pickle.dump(knn,open("model.pkl",'wb'))

model = pickle.load(open("model.pkl",'rb'))

predict = model.predict(np.array(x_new).reshape(1,-1))

print(predict)

print(iris.target_names)