'''------------File for Distribution Plot and Bar Charts-------------------''''


#Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#set font
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 13}

plt.rc('font', **font)



# Import the dataset
dataset = pd.read_csv('winequality.csv', delimiter=";")
dataset["quality"] = dataset["quality"].astype(float)
y = dataset.iloc[:, 12].values




# Distribution of quality
plt.hist(y, bins = 10, range = (0, 10))
plt.xlabel('Quality')
plt.ylabel('Amount')
plt.title("Histogram for 'quality'")
plt.savefig(filename='hsit.png', dpi=1000, format = 'png')






 '''----------------------Bar chart for % error---------------------'''
# data to plot
n_groups = 3
training_error = (53.45, 0, 0.02)
test_error = (52.69, 36.67, 29.36)
cv_error = (55.47, 38.07, 32.55)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8
 
rects1 = plt.bar(index, training_error, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Training Error')
 
rects2 = plt.bar(index + bar_width, test_error, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Test Error')

rects3 = plt.bar(index + 2*bar_width, cv_error, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Cross-Validation Error')
 
plt.xlabel('Algorithms')
plt.ylabel('% Error')
plt.title('% Errors for each algorithm')
plt.xticks(index + bar_width, ('Perceptron', 'KNN', 'Random Forest'))
plt.legend()
plt.tight_layout()
plt.savefig(filename='bar_chart_error.png', dpi=1000, format = 'png')




 '''---------------------Bar chart for mse error--------------------'''
# data to plot
n_groups = 3
training_error = (0.82, 0, 0.00017)
test_error = (0.81, 0.619, 0.397)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8
 
rects1 = plt.bar(index, training_error, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Training MSE')
 
rects2 = plt.bar(index + bar_width, test_error, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Test MSE')

 
plt.xlabel('Algorithms')
plt.ylabel('MSE Error')
plt.title('MSE Errors for each algorithm')
plt.xticks(index + bar_width, ('Perceptron', 'KNN', 'Random Forest'))
plt.legend()
plt.tight_layout()
plt.savefig(filename='bar_chart_mse.png', dpi=1000, format = 'png')
