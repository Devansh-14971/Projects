import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('C:/Users/lenovo/Desktop/College/ML/datasets/train.csv') 
test = pd.read_csv('C:/Users/lenovo/Desktop/College/ML/datasets/test.csv')
#print('Training data \n',train.isnull.sum())
#print('Test data \n',test.isnull.sum())
median_value = train['Age'].median()
train['Age'] = train['Age'].fillna(median_value)
test['Age'] = test['Age'].fillna(median_value)
train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)
predictors = ['Pclass', 'IsFemale', 'Age']
X_train = train[predictors].values
X_test = test[predictors].values
y_train = train['Survived'].values
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(y_predict[:10],'\n Vs \n',test['Survived'].values[:10])

##from sklearn.linear_model import LogisticRegressionCV
##model_cv = LogisticRegressionCV(10)
##model_cv.fit(X_train, y_train)
