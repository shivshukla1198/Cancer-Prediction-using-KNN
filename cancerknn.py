import numpy as np
import sklearn.datasets
breast_cancer=sklearn.datasets.load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target
print(X.shape,Y.shape)

#panda
import pandas as pd
data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
data['class']=breast_cancer.target
data.head()

#training and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
# random_state --> specific split of data, each value of random_state splits the data differently

#knn
from sklearn.neighbors import KNeighborsClassifier
neighcl = KNeighborsClassifier(n_neighbors=3)
neighcl.fit(X_train,Y_train)

#accuracy_score
from sklearn.metrics import accuracy_score
prediction_on_training_data = neighcl.predict(X_train)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print("Accuracy on Training Data  : ", accuracy_on_training_data)

# prediction on test_data
prediction_on_test_data = neighcl.predict(X_test)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print("Accuracy on Test Data  : ", accuracy_on_test_data)

input_data = (12.36,21.8,79.78,466.1,0.08772,0.09445,0.06015,0.03745,0.193,0.06404,0.2978,1.502,2.203,20.95,0.007112,0.02493,0.02703,0.01293,0.01958,0.004463,13.83,30.5,91.46,574.7,0.1304,0.2463,0.2434,0.1205,0.2972,0.09261)
# change the input_data to numpy_array to make prediction
input_data_as_numpy_array = np.asarray(input_data)
print(input_data)


# reshape the array as we predicting the output for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

#prediction
prediction = neighcl.predict(input_data_reshaped)
print(prediction) # returns a list with element [0] if Malignant; returns a listwith element[1], if begnign

if(prediction[0]==0):
  print("The breast cancer is Malignant")
else:
  print("The breast cancer is Benign")
