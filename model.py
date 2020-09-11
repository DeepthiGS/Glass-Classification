import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
from mlxtend.classifier import  StackingCVClassifier
from sklearn.svm import  SVC
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import  confusion_matrix,classification_report,accuracy_score


import pickle


df = pd.read_csv('glass.csv')
# print(df.head())

df.drop_duplicates(inplace=True)
features = df.columns[:-1]

# label encoding
unique =(df['Type'].unique())
# print(unique)
label = preprocessing.LabelEncoder()
labelled = label.fit_transform(unique)
# print(labelled)
type = label.transform(df['Type'])
df['type'] = type



df.drop(columns=["Type"],inplace=True)




num_0 = len(df[df['type']==0])
num_1 = len(df[df['type']==1])
num_2 = len(df[df['type']==2])
num_3 = len(df[df['type']==3])
num_4 = len(df[df['type']==4])
num_5 = len(df[df['type']==5])

oversampled_data = pd.concat([ df[df['type']==0] ,df[df['type']==1],df[df['type']==2].sample(60,replace=True) ,
                               df[df['type']==3].sample(60,replace=True),df[ df['type']==4].sample(60,replace=True),
                               df[df['type']==5].sample(60,replace=True)],ignore_index=True)

# print(oversampled_data.head())


X = oversampled_data.iloc[:,:-1]
Y = oversampled_data.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test, Y_train,Y_test = train_test_split(X,Y,random_state=101,test_size=0.3,shuffle=True)


from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_train =SS.fit_transform(X_train)
# print(X_train)
X_test = SS.transform(X_test)
# print(X_test)
# print(X_train)

xgb = XGBClassifier()
rf = RandomForestClassifier()
knn = KNeighborsClassifier()
svc = SVC()

stack = StackingCVClassifier(classifiers=(xgb,rf,knn,svc),
                             meta_classifier=xgb,cv=10,
                             use_features_in_secondary=True,
                             store_train_meta_features=True,
                             shuffle=False,
                             random_state=54)

stack.fit(X_train,Y_train)
# value=[[1.51665,13.14,3.45,1.76,72.48,0.6,8.38,0,0.17]]
# pred = SS.transform(value)
# print(pred)
# result =stack.predict(pred)
# print(result)



pickle.dump(stack,open('model.pkl','wb'))


# it is able to load and predict
# model = pickle.load(open('model.pkl','rb'))
# from sklearn.preprocessing import StandardScaler
# data = pd.read_csv('glass.csv')
# SS = StandardScaler()
# SS.fit_transform(data.iloc[:,:-1])
# pred = SS.transform([[1.51994,13.27,0,1.76,73.03,0.47,11.32,0,0]])
# print(pred)
# print(model.predict(pred))