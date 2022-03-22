
from os import remove
import numpy as np
import pandas as pd
import re
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import feature_selection, metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.feature_selection import VarianceThreshold # Import feature selection features
from sklearn import tree

train_matrix = np.zeros((800,100001)) # create zero filled train_matrix
test_matrix = np.zeros((350,100001))

train_file = open('train.txt','r')
test_file = open('test.txt','r')
format_file = open('formatfile.txt','r')

# Read train file
lines = train_file.readlines()
train_matrix_labels = []
np.set_printoptions(threshold = 50)
for row in range(len(lines)): # cleans each line from file, splits it into its own array
    cur_line = lines[row].strip('\n')
    cur_line = re.split('\t| ',cur_line)
    if '' in cur_line:
            cur_line.remove('')
    for col in range(len(cur_line)): # each value is used to mark an index in the train_matrix
        if col == 0:
            train_matrix_labels.append(cur_line[col])
        else:
            cur_val = int(cur_line[col])
            train_matrix[row][cur_val] = 1

# Read test file
test_lines = test_file.readlines()
for row in range(len(test_lines)): # cleans each line from file, splits it into its own array
    cur_line = test_lines[row].strip('\n')
    cur_line = cur_line.split(' ')
    if '' in cur_line:
            cur_line.remove('')
    for col in range(len(cur_line)): # each value is used to mark an index in the test_matrix        
        cur_val = int(cur_line[col])
        test_matrix[row][cur_val] = 1

# Read test labels
test_labels_lines = format_file.readlines()
test_label_matrix = []
for i in range(len(test_labels_lines)):
    test_label_matrix.append(int(test_labels_lines[i]))
test_label_matrix = np.reshape(test_label_matrix,(350,1))


threshold_prob = 0.8
sel = VarianceThreshold(threshold_prob*(1-threshold_prob))
sel = sel.fit_transform(train_matrix)
print(sel)
# train_labels = np.reshape(train_matrix_labels,(800,1)) # Reshapes True labels into a 2d array


# print(labels)
X_train, x_test, y_train,y_test= train_test_split(sel,train_matrix_labels,test_size=0.4, random_state=8 )

# Build Decision Tree Classifier and fit using binary train_matrix and label train_matrix
dtc = DecisionTreeClassifier()
dtc = dtc.fit(train_matrix,train_matrix_labels)
y_pred = dtc.predict(x_test)



# Information about the tree
print("number of leaves ",dtc.get_n_leaves())
print("depth of the tree ",dtc.get_depth())
# print(len(test_matrix),len(test_label_matrix))
print(dtc.score(test_matrix,test_label_matrix))
