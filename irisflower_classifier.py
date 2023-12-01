import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv("C:/Users/lenovo/Desktop/College/ML/datasets/Iris.csv")
#print("Target Labels\n", iris["Species"].unique())
#fig = px.scatter(iris, x="SepalWidthCm", y="SepalLengthCm", color="Species")
#fig.show()
iris = iris.drop("Id",axis = 1)
x = iris.drop("Species", axis=1)
y = iris["Species"]
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train.values, y_train)
x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction[0]))
