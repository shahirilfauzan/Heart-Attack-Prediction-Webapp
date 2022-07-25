# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:27:37 2022

@author: Shah
"""

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

#%%
def cramers_corrected_stat(matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(matrix)[0]
    n = matrix.sum()
    phi2 = chi2/n
    r,k = matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))  
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
#%% Constant
CSV_PATH = os.path.join(os.getcwd(),'dataset','heart.csv')
MODEL_PATH = os.path.join(os.getcwd(),'model','model.pkl')
#%% Step 1) Data Loading
df = pd.read_csv(CSV_PATH)

#%% Step 2) Data Inspection
df.info()
temp = df.describe().T

con = ['age','trtbps','chol','thalachh','oldpeak']

cat = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']

# Categorical
for i in cat:
    plt.figure()
    sns.countplot(df[i])
    plt.show()

# Continuous

for i in con:
    plt.figure()
    sns.distplot(df[i])
    plt.show()
    
df.boxplot(figsize=(12,6))

(df['thall']==0).sum()
(df['caa']==4).sum()

#thall 0 = mean null
#caa 4 = null, caa: number of major vessels (0-3)
df['thall'] = df['thall'].replace(0,np.nan) #replace 0 as Nans
df['caa'] = df['caa'].replace(4,np.nan) #replace 4 as Nans

df.isna().sum() #only caa and thall have NaNs
#%% Step 3) Data Cleaning

# 1) Outliers
# still within the range


# 2) Cleaning NaNs
#KNN
columns_name = df.columns

knn_i = KNNImputer()
df = knn_i.fit_transform(df) #return numpy array
df = pd.DataFrame(df) # to convert back into dataframe
df.columns = columns_name
df['thall'] = np.floor(df['thall'])
df['caa'] = np.floor(df['caa'])

temp = df.describe().T
# 3) Removing Duplicate
df.duplicated().sum()
df = df.drop_duplicates()

#%% Step 4) Features Selection

#cat vs cat
#cramer's V
for i in cat:
    print(i)
    matrix = pd.crosstab(df[i],df['output']).to_numpy()
    print(cramers_corrected_stat(matrix))
    
#only choose correlation above 0.40 for cat vs cat : exng,caa,cp,thall 

#con vs cat
for i in con:
    print(i)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i],axis=-1),df['output'])
    print(lr.score(np.expand_dims(df[i],axis=-1),df['output']))

#only choose correlation above 0.55 for con vs cat:trtbps,age,oldpeak,thalachh

# age,cp,trtbps,thalachh,exng,oldpeak,caa,thall selected features
#%% Step 5) Data Preprocessing
X = df.loc[:,['age','cp','trtbps','thalachh','exng','oldpeak','caa','thall']]
y = df['output']

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                   random_state=123)

#%%Model-development ---> pipeline

#LogisticRegression
pipeline_mms_lr = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('Logistic_Classifier', LogisticRegression())
    ]) #Pipeline([STEPS])

pipeline_ss_lr = Pipeline([
    ('Standard_Scaler', StandardScaler()),
    ('Logistic_Classifier', LogisticRegression())
    ]) #Pipeline([STEPS])

#Decision Tree
pipeline_mms_dt = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('Tree_classifier', DecisionTreeClassifier())
    ]) #Pipeline([STEPS])

pipeline_ss_dt = Pipeline([
    ('Standard_Scaler', StandardScaler()),
    ('Tree_classifier', DecisionTreeClassifier())
    ]) #Pipeline([STEPS])

#Random Forest Classifier
pipeline_mms_rf = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('Forest_classifier', RandomForestClassifier())
    ]) #Pipeline([STEPS])

pipeline_ss_rf = Pipeline([
    ('Standard_Scaler', StandardScaler()),
    ('Forest_classifier', RandomForestClassifier())
    ]) #Pipeline([STEPS])

#KNN
pipeline_mms_knn = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('KNN_classifier', KNeighborsClassifier())
    ]) #Pipeline([STEPS])

pipeline_ss_knn = Pipeline([
    ('Standard_Scaler', StandardScaler()),
    ('KNN_classifier', KNeighborsClassifier())
    ]) #Pipeline([STEPS])


#GradientBoost
pipeline_mms_gb = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('gb_Classifier', GradientBoostingClassifier())
    ]) #Pipeline([STEPS])

pipeline_ss_gb = Pipeline([
    ('Standard_Scaler', StandardScaler()),
    ('gb_Classifier', GradientBoostingClassifier())
    ]) #Pipeline([STEPS])

#SVC
pipeline_mms_svc = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('SVC_Classifier', SVC())
    ]) #Pipeline([STEPS])

pipeline_ss_svc = Pipeline([
    ('Standard_Scaler', StandardScaler()),
    ('SVC_Classifier', SVC())
    ]) #Pipeline([STEPS])


# Create A List To Store All The Pipeline
pipelines = [pipeline_mms_lr, pipeline_ss_lr,pipeline_mms_dt,pipeline_ss_dt,
             pipeline_mms_rf,pipeline_ss_rf,pipeline_mms_knn,pipeline_ss_knn,
             pipeline_mms_gb,pipeline_ss_gb,pipeline_mms_svc,pipeline_ss_svc]

for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
best_accuracy = 0
    
for i, pipe in enumerate(pipelines):
    print(pipe.score(X_test,y_test))
    if pipe.score(X_test,y_test) > best_accuracy:
        best_accuracy = pipe.score(X_test,y_test)
        best_pipeline = pipe

print('The best scaler and classifier for heart attack data is {} with accuracy of {}'. 
      format(best_pipeline.steps, best_accuracy))

#%%GridSearchCV ----> save the best estimator

pipeline_mms_lr = Pipeline([
                            ('Min_Max_Scaler', MinMaxScaler()),
                            ('Logistic_Classifier', LogisticRegression())
                            ]) #Pipeline([STEPS])

grid_param = [{'Logistic_Classifier__C':np.arange(0,2,0.1),
               'Logistic_Classifier__solver':['liblinear','lbfgs','saga'],
               'Logistic_Classifier__max_iter':[50,100,150,200]
             }]

gridsearch = GridSearchCV(pipeline_ss_lr,grid_param,cv=5,verbose=1,n_jobs=-1)
grid = gridsearch.fit(X_train,y_train)
gridsearch.score(X_test,y_test) 
print(grid.best_params_)

best_model = grid.best_estimator_

# model saving
with open(MODEL_PATH,'wb') as file:
    pickle.dump(best_model,file)

#%% Model evaluation

y_true = y_test
y_pred = best_pipeline.predict(X_test)

cr = classification_report(y_true,y_pred)
print(cr)

cm=confusion_matrix(y_test,y_pred)

labels=['not subscribed','subscribed']
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(confusion_matrix(y_test,y_pred))

