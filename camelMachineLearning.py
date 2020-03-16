import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import NearMiss
df = pd.read_csv('camelFeatures.csv')
#print(df.head())
#print(df)
target = 'label'
X = df.loc[:, df.columns!=target]
Y = df.loc[:, df.columns==target]
#print(X)
#print(Y)
nr = NearMiss(version=2, n_neighbors=3)
X_train_miss, Y_train_miss = nr.fit_sample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_train_miss, Y_train_miss, test_size=0.33, random_state=0)

gnb = GaussianNB()
nb = BaggingClassifier(gnb, n_estimators=10, random_state=0)

#print(X_test)
result = nb.fit(X_train, np.ravel(Y_train))
Y_Test_Pred = result.predict(X_test)
#print(Y_test)
#print(Y_test)
#print(Y_Test_Pred)
print('Accuracy Score:', accuracy_score(Y_test, Y_Test_Pred))
tn, fp, fn, tp = confusion_matrix(Y_test, Y_Test_Pred).ravel()
print('TN',tn, 'FP', fp, 'FN', fn,'TP', tp)
PD = tp/(tp+fn)
print('Probability of Detection', PD)
PF = fp/(fp+tn)
print('Probability of False Alarm', PF)
PREC = tp/(tp+fp)
print('Precision', PREC)
f_measure = 2*PD*PREC/(PD+PREC)
print('F MEASURE', f_measure)
g_measure = (2*PD*(100-PF))/(PD+(100-PF))
print('G MEASURE', g_measure)