# Artificial Intelligence (neural network) classification model for customer churn analysis

# Artificial Neural Network created by Keras API using TensorFlow backend

# Pre-requisites- Theano/Keras/Tensorflow libraries

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''
Indexes in X ->                  0           1         2      3   4     5           6               7         8                9               10
RowNumber   CustomerId	Surname  CreditScore	Geography	Gender	Age	Tenure Balance	  NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary Exited
1           15634602    Hargrave 619	       France    Female	42	2	    0           1              1         1 	              101348.88	      1
2           15647311    Hill     608	       Spain     Female	41	1	    83807.86    1              0	       1	              112542.58	      0
3           15619304    Onio     502	       France    Female	42	8	    159660.8    3              1	       0	              113931.57	      1
4           15701354    Boni     699	       France    Female	39	1	    0           2              0	       0	              93826.63	      0
'''

# Importing the dataset
dataset = pd.read_csv('C://Amit//ML//U//datasets//Artificial_Neural_Networks//Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
# Geography
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
# Gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# To remove ordering from 0,1,2 France, Spain, Germany by creating Dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Removing dummy variable trap- by removing first column (picking from index =1) from the 3 dummy variaable columns created for Geographies
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Constructing the ANN

# Importing the Keras libraries and packages
import keras

# Initialize NN
from keras.models import Sequential

# Build layers of NN
from keras.layers import Dense

# Initialising the ANN (the ANN will be built using sequence of layers)
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(                                         # Add the layer
        Dense(
                output_dim = 6,                         # (11 + 1)/2 = Avg. of input, output layers or alternatively  k-fold CV using cross validation set
                init = 'uniform',                       # randomly initialize the weights to uniformly distributed small weights
                activation = 'relu',                    # Rectifier activation function
                input_dim = 11                          # As 11 input/independent features; needed only for first layer      
            )
        )

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(
                output_dim = 1,                         # Classfication problem- 0 or 1, 3 if no. of classes =3
                init = 'uniform',   
                activation = 'sigmoid'))                # Classfication problem- 0 or 1 (can also rank the customers by probs.)
                                                        # activation = 'softmax' if more than two categories

# Compiling the ANN- apply gradient descent on the NN
classifier.compile(
                optimizer = 'adam',                     # Find the best weights using a type of SGD algo.- adam
                loss = 'binary_crossentropy',           # loss function within SGD to find optimal weights- logarithmic loss function for two classes
                                                        # If more than 2 classes- categorical_crossentropy
                metrics = ['accuracy']                  # To improve the performance of the model, the algo. uses 'accuracy' metrics
        )

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Training set accuracy = 83 %
# Epoch 100/100
# 8000/8000 [==============================] - 1s - loss: 0.3936 - acc: 0.8375 


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Choose threshold > 50%
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''Calculate accuracy
accuracy = (1550+175)/2000  = 86% without being an artist !!
We can improve it further
'''

'''
Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
So should we say goodbye to that customer ?

'''
X_1 = [[0,0,600,0,40,3,60000,2,1,1,50000]]
# Feature scaling
X_1 = sc.transform(X_1)
# array([[-0.5698444 , -0.57369368, -0.52111599, ...,  0.64259497,0.9687384 , -0.87203322]])

y1_pred = classifier.predict(X_1)
# array([[ 0.14906093]], dtype=float32)

y1_pred = (y1_pred > 0.5)
# False


# Part 4 - Evaluating the ANN (Directly after Part 1 - Data Preprocessing)

'''
Variance problem- Running the model again on a different TEST set can lead to very different test results
Solution- Using 10-fold CV, training set into 10 folds (iterations) each having a different test set and then average (of error and SD)
from all the iterations.
'''
# Evaluating the ANN

# Use Keras Wrapper for sklearn CV functionality 
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Function to build ANN architecture from Part-2
def build_classifier():
    from keras.models import Sequential

    # Build layers of NN
    from keras.layers import Dense
    
    # Initialising the ANN (the ANN will be built using sequence of layers)
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(                                         # Add the layer
            Dense(
                    output_dim = 6,                         # (11 + 1)/2 = Avg. of input, output layers or alternatively  k-fold CV using cross validation set
                    init = 'uniform',                       # randomly initialize the weights to uniformly distributed small weights
                    activation = 'relu',                    # Rectifier activation function
                    input_dim = 11                          # As 11 input/independent features; needed only for first layer      
                )
            )
    
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    classifier.add(Dense(
                    output_dim = 1,                         # Classfication problem- 0 or 1, 3 if no. of classes =3
                    init = 'uniform',   
                    activation = 'sigmoid'))                # Classfication problem- 0 or 1 (can also rank the customers by probs.)
                                                            # activation = 'softmax' if more than two categories
    
    # Compiling the ANN- apply gradient descent on the NN
    classifier.compile(
                    optimizer = 'adam',                     # Find the best weights using a type of SGD algo.- adam
                    loss = 'binary_crossentropy',           # loss function within SGD to find optimal weights- logarithmic loss function for two classes
                                                            # If more than 2 classes- categorical_crossentropy
                    metrics = ['accuracy']                  # To improve the performance of the model, the algo. uses 'accuracy' metrics
            )
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs = 2)

mean = accuracies.mean()    # 83%
# 0.833499995619 
variance = accuracies.std()
# 0.00881759525571
# Less variance is good

# Part - 6 Improving the ANN 
#  (above 83% as obtained in CV scores mean above)

'''
Pending

Problem- Overfitting can occur; signs of overfiting would be high variance observed in CV scores above
Solution- Dropout regularization to reduce overfitting if needed
          ~~~~~~~~~~~~~~~~~~~~~
'''

'''
Part - 6 Tuning the ANN hyperparameters (batch_size = 10, nb_epoch =100, optimizer='adam') using GridSearch
        ~~~~~~~~
'''

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Function to build ANN architecture from Part-2
def build_classifier(optimizer):
    from keras.models import Sequential

    # Build layers of NN
    from keras.layers import Dense
    
    # Initialising the ANN (the ANN will be built using sequence of layers)
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(                                         # Add the layer
            Dense(
                    output_dim = 6,                         # (11 + 1)/2 = Avg. of input, output layers or alternatively  k-fold CV using cross validation set
                    init = 'uniform',                       # randomly initialize the weights to uniformly distributed small weights
                    activation = 'relu',                    # Rectifier activation function
                    input_dim = 11                          # As 11 input/independent features; needed only for first layer      
                )
            )
    
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    classifier.add(Dense(
                    output_dim = 1,                         # Classfication problem- 0 or 1, 3 if no. of classes =3
                    init = 'uniform',   
                    activation = 'sigmoid'))                # Classfication problem- 0 or 1 (can also rank the customers by probs.)
                                                            # activation = 'softmax' if more than two categories
    
    # Compiling the ANN- apply gradient descent on the NN
    classifier.compile(
                    optimizer = optimizer,                     # Find the best weights using a type of SGD algo.- adam
                    loss = 'binary_crossentropy',           # loss function within SGD to find optimal weights- logarithmic loss function for two classes
                                                            # If more than 2 classes- categorical_crossentropy
                    metrics = ['accuracy']                  # To improve the performance of the model, the algo. uses 'accuracy' metrics
            )
    return classifier

classifier = KerasClassifier(build_fn = build_classifier) # Removed batch_size = 10, nb_epoch = 100

# create a dict containing the hyper-params to be optimized

parameters = {'batch_size':[25,32], 
              'nb_epoch':[100,500],
              'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring = 'accuracy',cv=10)


grid_search = grid_search.fit(X_train,y_train)


best_parameters = grid_search.best_params_
# {'batch_size': 25, 'nb_epoch': 500, 'optimizer': 'adam'}
best_score = grid_search.best_score_
# 0.85; improvement over 0.83













    



