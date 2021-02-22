import pandas as pd
import math
import random
from itertools import combinations

'''
This function takes in four dataframes and creates a list of points using x_train
and their associated flowers using y_train. Then the function uses euclidean distances
to calculate the distances of the x_test points to all other x_train points and picks the
nearest neighbors, the number of neighbors is based on a variable within the function.
'''
def KNN_dataframes(x_train, x_test, y_train, y_test):
    print("------------------------------------------")
    print(x_train.columns[0] + " x " + x_train.columns[1])
    correct_guess = 0
    number_of_neighbors = 5
    #This just populates a list with impossibly large numbers for this given dataset
    nearest_neighbors = [(100,"NaN")] * number_of_neighbors

    for i in range(0,x_test.shape[0]):
        test_point = (x_test.iloc[i,0],x_test.iloc[i,1])

        for j in range(0,x_train.shape[0]):
            train_point = (x_train.iloc[j,0],x_train.iloc[j,1])
            distance = math.sqrt((test_point[0]-train_point[0])**2 + \
                                 (test_point[1]-train_point[1])**2)

            neighbor = (distance, str(y_train['Flower'][j]))


            if max(nearest_neighbors)[0] > neighbor[0]:
                nearest_neighbors.remove(max(nearest_neighbors))
                nearest_neighbors.append(neighbor)


        #Sum the neighbors
        sum_list = []
        for j in range(0,len(nearest_neighbors)):
            sum_list.append(nearest_neighbors[j][1])

        setosa_count = sum_list.count('Iris-setosa')
        versicolor_count = sum_list.count('Iris-versicolor')
        virginica_count = sum_list.count('Iris-virginica')

        #Guess
        if setosa_count > versicolor_count and setosa_count > virginica_count:
            guess = "Iris-setosa"
        elif versicolor_count > virginica_count and versicolor_count > setosa_count:
            guess = "Iris-versicolor"
        elif virginica_count > setosa_count and virginica_count > versicolor_count:
            guess = "Iris-virginica"
        else:
            #If there is a twoway tie
            if setosa_count == versicolor_count:
                guess = random.choice(['Iris-setosa','Iris-versicolor'])
            elif setosa_count == virginica_count:
                guess = random.choice(['Iris-setosa','Iris-virginica'])
            elif versicolor_count == virginica_count:
                guess = random.choice(['Iris-versicolor','Iris-virginica'])
            #If there is a threeway tie
            else:
                guess = random.choice(['Iris-setosa','Iris-versicolor','Iris-virginica'])
        #Determine if guess is accurate
        #print("Guess: "+guess)
        #print("Answer: "+y_test.iloc[i,0])
        if guess == y_test.iloc[i,0]:
            correct_guess += 1

        #Append test point to x_train
        x_train.append(x_test.loc[i,:])
        #Reset nearest_neighbors
        nearest_neighbors = [(100,"NaN")] * number_of_neighbors


    #Accuracy = correct guesses / total number of tests
    accuracy = correct_guess/x_test.shape[0]
    print(accuracy)
    return accuracy


data = pd.read_csv("bezdekIris.data",header=None)
columns = ['Sepel Length','Sepel Width','Petal Length','Petal Width','Flower']

data.columns=columns
#This shuffles the data for training/testing purposes
data = data.sample(frac=1).reset_index(drop=True)

#Splits the data between inputs (X) and outputs (Y)
X = data[['Sepel Length','Sepel Width','Petal Length','Petal Width']]
Y = data[['Flower']]

"""
I'm using a train test split of 90:10, because there are 150 entries
that means 135 training datums and 15 test points. Technically KNN doesn't
need training but these are used to validate that the model can make predictions
"""
train_test_split = 135

x_train = X[0:train_test_split]
x_test = X[train_test_split:].reset_index(drop = True)
y_train = Y[0:train_test_split]
y_test = Y[train_test_split:].reset_index(drop = True)

#Gets every combination of columns in X
ensemble = list(combinations(columns[0:4], 2))

#Convert all elements of ensemble to lists rather than tuples
for i in range(0,len(ensemble)):
    ensemble[i] = list(ensemble[i])

accuracy_total = 0

for combo in ensemble:
    accuracy_total += KNN_dataframes(x_train.loc[:,combo],x_test.loc[:,combo],y_train, y_test)

print("------------------------------------------")
print("Total Accuracy")
print(accuracy_total/len(ensemble))
