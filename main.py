#Spam mail prediction using support vector machine (SVM)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #used to extract the features from the text
from sklearn.svm import LinearSVC #linear svm model
from sklearn.metrics import accuracy_score

raw_mail_data = pd.read_csv('spam.csv',encoding='latin-1')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

mail_data.head() #sample dataset in pandas dataframe

X=mail_data['v2'] #as in dataset
Y=mail_data['Category']
print(X)
print('...............')
print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)#20% of data as test data

feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')

X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.fit_transform(X_test)

Y_train=Y_train.astype('int') #specify the type you want in the parenthesis
Y_test=Y_test.astype('int')

model=LinearSVC()
model.fit(X_train_features,Y_train)

prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)
print("The accuracy on training data: ",accuracy_on_training_data)
#prediction on test data
prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
print("The accuracy on test data: ",accuracy_on_test_data)
#Prediction on new mail
input_mail = ["WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]

input_mail_features=feature_extraction.transform(input_mail)
prediction=model.predict(input_mail_features)

print(prediction)

if (prediction[0] == 0): #first element
  print("It is a spam mail")
else:
  print("It is a ham mail")
