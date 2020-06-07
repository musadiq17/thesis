
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from imblearn.under_sampling import RandomUnderSampler
df = pd.read_csv('derbyFeatures.csv')
#print(df.head())
#print(df)
target = 'label'
G_measure = 0
Recall = 0
Precision = 0
F_measure = 0
X = df.loc[:, df.columns!=target]
Y = df.loc[:, df.columns==target]
nr = RandomUnderSampler(random_state=16)


skf = KFold(n_splits=10, random_state=5, shuffle=True)
skf.get_n_splits(X,Y)

G_measure = 0
Recall = 0
Precision = 0
F_measure = 0

for train_index, test_index in skf.split(X, Y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = Y.loc[train_index], Y.loc[test_index]

    X_train_S, y_train_S = nr.fit_sample(X_train, y_train)
    #print(X_train_S)
    #print(y_train_S)
    clf1 = LogisticRegression()
    clf2 = DecisionTreeClassifier()
    #clf3 = MultinomialNB()
    clf3 = GaussianNB()
    evc = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('nb', clf3)], voting='hard')
    result = evc.fit(X_train_S, np.ravel(y_train_S))
    Y_Test_Pred = result.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, Y_Test_Pred).ravel()
    tp = 0.1+tp
    fp = 0.1+fp
    #print('TN', tn, 'FP', fp, 'FN', fn ,'TP', tp)
    PD = (tp / (tp + fn)) * 100

    Recall += PD
    #print('Probability of Detection', PD)
    PF = (fp / (fp + tn)) * 100
    #print('Probability of False Alarm', PF)
    PREC = (tp / (tp + fp)) * 100
    #PREC = PREC
    Precision += PREC
    #print('Precision', PREC)
    f_measure = (2 * PD * PREC) / (PD + PREC)

    F_measure += f_measure
    #print('F MEASURE', f_measure)
    g_measure = (2 * PD * (100 - PF)) / (PD + (100 - PF))
    G_measure += g_measure
    #print('G MEASURE',g_measure)

print('Precision', Precision/10)
print('Recall', Recall/10)
print('F_Measure', F_measure/10)
print('G_MEASURE', G_measure/10)

