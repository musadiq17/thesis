import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
#import RUS
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import NearMiss, ClusterCentroids, TomekLinks, NeighbourhoodCleaningRule,RandomUnderSampler, CondensedNearestNeighbour, EditedNearestNeighbours,OneSidedSelection, AllKNN
df = pd.read_csv('wicketFeatures.csv')
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
#nr = NearMiss(version=3, n_neighbors=3)
#nr = ClusterCentroids()
#nr = TomekLinks()
#nr = RandomUnderSampler()
#nr = NeighbourhoodCleaningRule(n_neighbors=3)
#nr = CondensedNearestNeighbour(n_neighbors=3)
nr = AllKNN()
#nr = OneSidedSelection(n_neighbors=3)
#nr = EditedNearestNeighbours(n_neighbors=3)
X_train_miss, Y_train_miss = nr.fit_resample(X, Y)


X_train, X_test, Y_train, Y_test = train_test_split(X_train_miss, Y_train_miss, test_size=0.33, random_state=0)
#clf2 = KNeighborsClassifier(n_neighbors=3)
clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
#clf2 = RandomForestClassifier(n_estimators=20, random_state=0)
clf3 = GaussianNB()
#clf3 = MultinomialNB()
#evc = BaggingClassifier(clf1, n_estimators=10, random_state=0)
evc = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('nb', clf3)], voting='hard')
#evc = AdaBoostClassifier(n_estimators=50, base_estimator=clf3,  learning_rate=1)
result = evc.fit(X_train, np.ravel(Y_train))
Y_Test_Pred = result.predict(X_test)
#print(evc.score(X_test,np.ravel(Y_test)))
#print(X_train)
#print(X_test)
#print(Y_train)
#print(Y_test)

#KNN_with_out_bagging = KNeighborsClassifier(n_neighbors=5)
#print(X_train, Y_train)
#KNN_with_out_bagging.fit(X_train,np.ravel(Y_train))
#result = KNN_with_out_bagging.score(X_test, Y_test)
#print(result)
#KNN_with_bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=4), n_estimators=10, random_state=10)
#KNN_with_bagging.fit(X_train, np.ravel(Y_train))
#print(KNN_with_bagging.score(X_test,Y_test))

#nb = gnb.fit(X_train, np.ravel(Y_train))
#print(gnb.score(X_test,Y_test))
#nb = BaggingClassifier(gnb, n_estimators=10, random_state=0)

#print(X_test)
#result = gnb.fit(X_train, np.ravel(Y_train))
#print(result)
#Y_Test_Pred = result.predict(X_test)
#Y_Test_Pred= gnb.fit(X_train, X_train).predict(X_test)
#print(Y_test)
#print(Y_Test_Pred)

#print('Accuracy Score:', accuracy_score(Y_test, Y_Test_Pred))
tn, fp, fn, tp = confusion_matrix(Y_test, Y_Test_Pred).ravel()
#print('TN', tn, 'FP', fp, 'FN', fn ,'TP', tp)
PD = (tp/(tp+fn))*100
#print('Probability of Detection', PD)
PF = (fp/(fp+tn))*100
#print('Probability of False Alarm', PF)
PREC = (tp/(tp+fp))*100
#print('Precision', PREC)
f_measure = (2*PD*PREC)/(PD+PREC)

#print('F MEASURE', f_measure)
g_measure = (2*PD*(100-PF))/(PD+(100-PF))
#if g_measure > 20:

print('G MEASURE', g_measure)


