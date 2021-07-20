import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
%matplotlib inline
import seaborn as sns 
sns.set()
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()
train.shape

y_full = train["Survived"]

features = ["Pclass","Sex", "Age","IsAlone", "FamilySize", "Status","Embarked","Fare","Cabin_category","HasCabin"]
X_full = pd.get_dummies(train[features])
X_test_full = pd.get_dummies(test[features])

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full, train_size=0.8, test_size=0.2,random_state=0)

X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
model = RandomForestClassifier(max_depth=3, random_state=3)
model.fit(X_train,y_train)
val_predictions = model.predict(X_valid)
accuracy = accuracy_score(val_predictions,y_valid)
accuracy

model.fit(X_full, y_full)
predictions = model.predict(X_test_full)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)