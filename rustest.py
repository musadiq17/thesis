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
def generate_model_report(y_actual, y_predicted):
    print("Accuracy = " , accuracy_score(y_actual, y_predicted))
    print("Precision = " ,precision_score(y_actual, y_predicted))
    print("Recall = " ,recall_score(y_actual, y_predicted))
    print("F1 Score = " ,f1_score(y_actual, y_predicted))
    pass
def generate_auc_roc_curve(clf, X_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test,  y_pred_proba)
    auc = roc_auc_score(Y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="AUC ROC Curve with Area Under the curve ="+str(auc))
    plt.legend(loc=4)
    plt.show()
    pass
df = pd.read_csv('Features.csv')
#print(df.head())
target = 'label'
X = df.loc[:, df.columns!=target]
#print(X)
Y = df.loc[:, df.columns==target]
#print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.33,random_state=42)
#ax = sns.countplot(x=target, data=df)

#print(df[target].value_counts())
#print(100* (17/float(df.shape[0])))
#print(100* (32/float(df.shape[0])))

#print(Y_train[target].value_counts())
#clf = LogisticRegression().fit(X_train, Y_train)
#Y_Test_Pred = clf.predict(X_test)
#pd.crosstab(pd.Series(Y_Test_Pred, name = 'Predicted'),pd.Series(Y_test[target], name = 'Actual'))
#report = generate_model_report(Y_test, Y_Test_Pred)
#print(report)
minority_class_len = len(df[df[target] == 1])
#print(minority_class_len)
majority_class_indices = df[df[target] == 0].index
#print(majority_class_indices)
random_majority_indices = np.random.choice(majority_class_indices,minority_class_len,replace=False)
#print(random_majority_indices)
minority_class_indices = df[df[target] == 1].index
#print(minority_class_indices)
under_sample_indices = np.concatenate([minority_class_indices,random_majority_indices])
#print(under_sample_indices)
under_sample = df.loc[under_sample_indices]
#print(under_sample)

sns.countplot(x=target, data=under_sample)
plt.show()