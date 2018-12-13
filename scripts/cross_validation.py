# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



'------------------PREPROCESSING--------------------'''

def importData(file):
    df = pd.read_csv(file, delimiter=";")
    return df

def separateFeatures(d):
    X = d.iloc[:, 1:-1].values
    y = d.iloc[:, 12].values
    return X,y


def removeOutliers(arr, vec):
    outliers = (np.abs(stats.zscore(arr)) < 4).all(axis=1)
    return arr[outliers], vec[outliers]


def standardize(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def split(X,y):
    return train_test_split(X, y, test_size = 0.1, random_state = 0)


def preprocess(file ,out, scale):
    df = importData(file)
    X,y = separateFeatures(df)
    X_train, X_test, y_train, y_test = split(X,y)
    if out:
        X_train,y_train = removeOutliers(X_train,y_train)
        X_test,y_test = removeOutliers(X_test,y_test)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return df, X, y, X_train, X_test, y_train, y_test 




'''-----------------------PREPROCESSING-------------------------'''
df, X, y, X_train, X_test, y_train, y_test  = preprocess ('winequality.csv', True , True)




'''---------CROSS VALIDATION FOR  KNN: parameter = number of neighbors----------'''
best_score = 0
param_options = []
param_range = range(1, 11)
param_scores = []
for i in param_range:
    classifier = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
    score = scores.mean()
    param_scores.append(score)
    if best_score < score:
        best_score = score
        best_param = i

plt.plot(param_range,param_scores)
plt.xlabel('Number of trees')
plt.ylabel('Accuracy')
print(best_param, best_score)
print(param_scores)
plt.savefig(filename='KNN_CV.png', dpi=1000, format = 'png')





'''---------CROSS VALIDATION FOR RANDOM FOREST: parameter = number of trees-------'''
best_score = 0
param_options = []
param_range = range(30, 61)
param_scores = []
for i in param_range:
    classifier = RandomForestClassifier(n_estimators = i)
    scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
    score = scores.mean()
    param_scores.append(score)
    if best_score < score:
        best_score = score
        best_param = i

plt.plot(param_range,param_scores)
plt.xlabel('Number of trees')
plt.ylabel('Accuracy')
print(best_param, best_score)
print(param_scores)
plt.savefig(filename='RF_CV.png', dpi=1000, format = 'png')








