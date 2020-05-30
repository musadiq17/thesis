import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import AllKNN
df = pd.read_csv('chromiumFeatures.csv')
#print(df.head())
#print(df)
target = 'label'

X = df.loc[:, df.columns!=target]
Y = df.loc[:, df.columns==target]
#print(X)
#print(Y)
nr = AllKNN()
X_train_miss, Y_train_miss = nr.fit_sample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_train_miss, Y_train_miss, test_size=0.33, random_state=79)

clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf3 = GaussianNB()
evc = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('nb', clf3)], voting='hard')
#print(X_test)
result = evc.fit(X_train, np.ravel(Y_train))
Y_Test_Pred = result.predict(X_test)
print(Y_train)
print(Y_test)
#print(Y_Test_Pred)
accuracyy = accuracy_score(Y_test, Y_Test_Pred)
print('Accuracy Score:', accuracyy*100)
tn, fp, fn, tp = confusion_matrix(Y_test, Y_Test_Pred).ravel()
print('TN',tn, 'FP', fp, 'FN', fn,'TP', tp)
PD = (tp/(tp+fn))*100
print('Probability of Detection', PD)
PF = (fp/(fp+tn))*100
print('Probability of False Alarm', PF)
PREC = (tp/(tp+fp))*100
print('Precision', PREC)
f_measure = 2*PD*PREC/(PD+PREC)
print('F MEASURE', f_measure)
g_measure = (2*PD*(100-PF))/(PD+(100-PF))
print('G MEASURE', g_measure)