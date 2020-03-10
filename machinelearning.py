import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from pylab import rcParams
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import NearMiss
df = pd.read_csv('Features.csv')
#print(df.head())
#print(df)
target = 'label'
#minority_class_len = len(df[df[target] == 1])
#print(minority_class_len)
#majority_class_indices = df[df[target] == 0].index
#print(majority_class_indices)
#random_majority_indices = np.random.choice(majority_class_indices,minority_class_len,replace=False)
#print(random_majority_indices)
#minority_class_indices = df[df[target] == 1].index
#print(minority_class_indices)

#under_sample_indices = np.concatenate([minority_class_indices,random_majority_indices])
#print(under_sample_indices)
#under_sample = df.loc[under_sample_indices]
#print(under_sample)

#sns.countplot(x=target, data=under_sample)
#plt.show()
X = df.loc[:, df.columns!=target]
Y = df.loc[:, df.columns==target]
#print(X)
#print(Y)
nr = NearMiss(version=2, n_neighbors=3)
X_train_miss, Y_train_miss = nr.fit_sample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_train_miss, Y_train_miss, test_size=0.33, random_state=0)

#KNN_with_out_bagging = KNeighborsClassifier(n_neighbors=5)
#print(X_train, Y_train)
#KNN_with_out_bagging.fit(X_train,np.ravel(Y_train))
#result = KNN_with_out_bagging.score(X_test, Y_test)
#print(result)
#KNN_with_bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=4), n_estimators=10, random_state=10)
#KNN_with_bagging.fit(X_train, np.ravel(Y_train))
#print(KNN_with_bagging.score(X_test,Y_test))
gnb = GaussianNB()
#nb = gnb.fit(X_train, np.ravel(Y_train))
#print(gnb.score(X_test,Y_test))
nb = BaggingClassifier(gnb, n_estimators=10, random_state=30)
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