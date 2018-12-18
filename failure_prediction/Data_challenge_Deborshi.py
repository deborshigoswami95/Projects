
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.metrics import recall_score, confusion_matrix 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectFromModel
import math
import seaborn as sns
import string
from mpl_toolkits.mplot3d import Axes3D as ax
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# In[58]:


#Handling Missing Values
def handle_missing_values(threshold, train, test):
    train = train.replace(to_replace = 'na', value = np.nan)
    train = train.astype('float64')
    drop_nan_cols = list(train.loc[:, (train.isnull().sum(axis=0) >= threshold)].columns)
    train = train.drop(drop_nan_cols, axis = 1)
    #mean imputation
    train = train.fillna(train.mean(skipna = True))
    
    test = test.replace(to_replace = 'na', value = np.nan)
    test = test.astype('float64')
    test = test.drop(drop_nan_cols, axis = 1)
    #mean imputation
    test = test.fillna(test.mean(skipna = True))
    
    return (train, test)




#Feature Engineering
def feature_engineering(X):
    L = list()
    
    for i in string.ascii_lowercase:
        for j in string.ascii_lowercase:
            L.append(i+j)
    
    LOL = pd.DataFrame()
    for k in L:
        column_list = [col for col in X if col.startswith(k)]
        if column_list:
            column = X[column_list].mean(axis=1)
            LOL[k] = column
    
    return(LOL)


#Scaling
def scaling(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    train = pd.DataFrame(scaler.transform(train))
    test = pd.DataFrame(scaler.transform(test))
    return (train, test)



#80/20 validation split for SVM
def validation_split(train_data, train_label):
    X_tr, X_te, L_tr, L_te = train_test_split( train_data, train_label, test_size=0.20, random_state=42)
    return(X_tr, X_te, L_tr, L_te)




#Callibrating cut-off probability using 10-fold cross validation
def callibration_cross(clf, cutoff): #accepts an object of the classifier's class and the list of cutoff probabilities to be tested
    def custom_cost(cutoff):
        def score(clf, X, y):
            S = pd.Series(clf.predict_proba(X)[:,1])
            S[S <= cutoff] = 0
            S[S > cutoff] = 1
            tn, fp, fn, tp = confusion_matrix(y, S).ravel()
            return fn*500 + fp*10
        return score

    costs = []
    for ctof in cutoff:
        cost = cross_val_score(clf, train, train_label.values.ravel(), cv = 10, scoring = custom_cost(ctof))
        costs.append(cost)
        
    sns.boxplot(cutoff, costs)
    plt.title('costs for Bagging CART prediction for a given cutoff probability')
    plt.xlabel('cut off value')
    plt.ylabel('costs')
    plt.show()
    return()



# In[69]:


train = pd.read_csv("aps_failure_training_set.csv", skiprows = 20)
test = pd.read_csv(("aps_failure_test_set.csv"), skiprows = 20)

#Visualizing class imbalance in training set
count = pd.value_counts(train['class'], sort = True).sort_index()
count.plot(kind = 'bar')
plt.title("class count in training set")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

#Visualizing class imbalance in testing set
count = pd.value_counts(test['class'], sort = True).sort_index()
count.plot(kind = 'bar')
plt.title("class count in test set")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()


# In[70]:


#DATA PREPARATION AND PREPROCESSING

#Map class values to 1{POS} and 0{NEG}
test['class'] = test['class'].map({'neg':0, 'pos':1})
train['class'] = train['class'].map({'neg':0, 'pos':1})

#Extract Training labels
train_label = pd.DataFrame(train.loc[:, train.columns == 'class'])
train = train.drop('class', axis = 1)

#Extract Test Labels
test_label = pd.DataFrame(test.loc[:, test.columns == 'class'])
test = test.drop('class', axis = 1)


threshold = 1500 #Eliminate all coulmns where number of NaN values is greater than threshold
#Handling Missing values
train, test = handle_missing_values(threshold, train, test)

#Engineering new features
train = feature_engineering(train)
test = feature_engineering(test)

#Handling Outliers using scaling
train, test = scaling(train, test)


# In[63]:


#Support Vector Machine
X_train, X_test, l_train, l_test = validation_split(train, train_label)

#tuning SVM
costs = []
kernels = ['linear', 'poly', 'rbf']
weights = [0.1, 1, 10, 50, 500]
for k in kernels:
    for i in weights:
        SVM = svm.SVC(C = 1.5, kernel = k, gamma = 0.01, class_weight = {0:1, 1:i}) #declare class SVM
        SVM.fit(X_train, l_train.values.ravel())
        predicted_label = SVM.predict(X_test)
        predicted_label = pd.DataFrame(predicted_label)
        tn, fp, fn, tp = confusion_matrix(l_test, predicted_label).ravel()
        print "cost: ", fn * 500 + fp * 10, "     false negatives: ", fn, "     false positives: ", fp
        print " "


#Final prediction
print "SVM classification"
SVM = svm.SVC(C = 1.5, kernel = 'poly', gamma = 0.01, class_weight = {0:1, 1:500})
SVM.fit(train, train_label.values.ravel())
predicted_label = SVM.predict(test)
print "Confusion Matrix"
print confusion_matrix(test_label, predicted_label)
tn, fp, fn, tp = confusion_matrix(l_test, predicted_label).ravel()
print "cost: ", fn * 500 + fp * 10, "     false negatives: ", fn, "     false positives: ", fp
print " "


# In[73]:


#Tuning Bagging cart

BG = BaggingClassifier(n_estimators = 40)
cutoff = [0.0001, 0.005, 0.05, 0.01, 0.1]
callibration_cross(BG, cutoff)

#final prediction
print "Bagging CART Classification"
print " "
BG = BaggingClassifier(n_estimators = 40)
BG.fit(train, train_label.values.ravel())
S = pd.Series(BG.predict_proba(test)[:,1])
p = 0.005
S[S <= p] = 0
S[S > p] = 1
tn, fp, fn, tp = confusion_matrix(test_label, S).ravel()
print i, " :   ", "cost: ", fn * 500 + fp * 10, "     false negatives: ", fn, "     false positives: ", fp
print " "
print "False Negatives :    ", fn
print "False Positives :    ", fp


# In[60]:


#Tuning Random Forest
cutoff = [0.0005, 0.005, 0.05, 0.01, 0.1]
RF = RandomForestClassifier(n_estimators = 80, max_features = 'sqrt', oob_score = False, warm_start = False, class_weight = {0:1, 1:0.5})
callibration_cross(RF, cutoff)

#Final Prediction on random forest
print "Random Forest Classification"
print " "
RF = RandomForestClassifier(n_estimators = 80, max_features = 'sqrt', oob_score = False, warm_start = False, class_weight = {0:1, 1:0.5})
RF.fit(train, train_label.values.ravel())
S = pd.Series(RF.predict_proba(test)[:,1])
p = 0.05
S[S <= p] = 0
S[S > p] = 1
tn, fp, fn, tp = confusion_matrix(test_label, S).ravel()
print i, " :   ", "cost: ", fn * 500 + fp * 10, "     false negatives: ", fn, "     false positives: ", fp
print " "
print "False Negatives :    ", fn
print "False Positives :    ", fp

