'''-------------------PHD PAPER----------------------'''



'''----------------IMPORTING LIBRARIES--------------'''
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error



'''--------------FUNCTION DEFINITIONS---------------'''
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

def dfConverter (arr):
    return pd.DataFrame(arr, columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulfates', 'alcohol'])
        
def createModel(model):
    if (model == 'P'):
        return Perceptron()
    elif (model == 'KNN'):
        return KNeighborsClassifier(n_neighbors = 1)
    elif (model == 'RF'):
        return RandomForestClassifier(n_estimators = 58, max_features = 4)


def applyModel(model_name, X_train, y_train):
    model = createModel(model_name)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return model, y_pred_train, y_pred_test

    
def performance(model, y_train, y_pred_train, y_test, y_pred_test):
    training_score = accuracy_score(y_train, y_pred_train)
    test_score = accuracy_score(y_test, y_pred_test)
    x_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    cnf = confusion_matrix(y_test, y_pred_test, labels=[3, 4, 5, 6, 7, 8])
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    return training_score, test_score, x_score, cnf, mse_train, mse_test


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


def learn(model_name, X_train, y_train, y_test):
    model, y_pred_train, y_pred_test = applyModel(model_name, X_train, y_train)
    return performance(model, y_train, y_pred_train, y_test, y_pred_test)




'''--------------------LEARNING MAIN----------------------'''


# Prompt for user input
print("Preprocessing: enter 'y' for yes and 'n' for no")

while True:
    out_str = input("Remove outliers? -")
    if (out_str == 'y') or (out_str == 'n'):
        if out_str == 'y':
            out = True
        else:
            out = False
        break
    else:
        print("Invalid input, please enter 'y' or 'n'")
        

while True:
    scale_str = input("Standardize data? -")
    if (scale_str == 'y') or (scale_str == 'n'):
        if scale_str == 'y':
            scale = True
        else:
            scale = False
        break
    else:
        print("Invalid input, please enter 'y' or 'n'")
        


model_name = input("Enter machine learning model name: \n -'P' for Perceptron\n-'KNN' for K-Nearest Neighbors\n-'RF' for Random Forest\nor 'stop': ")
print('-------------------\n')

while True:
    
    if (model_name == 'stop'):
        print('Done learning')
        break
    
    elif (model_name == 'RF') or (model_name == 'KNN') or (model_name == 'P'):
        
        # 1st step: proprocess the data. Can choose to remove outliers and scale dataset
        df, X, y, X_train, X_test, y_train, y_test = preprocess ('winequality.csv', out , scale)
        
        # 2nd step: learning
        training_score, test_score, x_score, cnf, mse_train, mse_test = learn(model_name , X_train, y_train, y_test)
        
        # Display results
        print('Running', model_name)
        print('Training Error:', (1-training_score)*100)
        print('Test Error:', (1-test_score)*100)
        print('Cross-validation Error:', (1-x_score)*100)
        print('Confusion Matrix:\n', cnf)
        print('Training MSE: ', mse_train)
        print('Test MSE:',mse_test)
        model_name = input("Re-enter machine learning model name or 'stop': ")
        print('-------------------\n')

        
    else:
       model_name = input("Invalid model. Enter one of these: 'P', 'KNN', 'RF': ")
       print('-------------------\n')
