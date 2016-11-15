# pytrain

Machine Learning library for python
![alt text](https://github.com/becxer/pytrain/raw/master/tmp/pytrain.png "pytrain")

## Algorithms

+ Decision Tree(ID3)
+ Gaussian NaiveBayes
+ NaiveBayes
+ KNN
+ Neural Network(FNN)
+ Logistic Regression
+ Linear Regression
+ DBSCAN
+ Apriori
+ Kmeans
+ HierarchicalClustering
+ BinarySVM
+ SVM [Not yet]
+ HMM [Not yet]
+ CRF [Not yet]

## Requirements

 - Numpy
 - Python 2 or 3

## Installation

    $ sudo pip install --upgrade pytrain
    
## Basic Usage

    import numpy as np
    from pytrain.NeuralNetwork import FNN

    # Simple dataset
    train_mat = [[0.12,0.25],[3.24,4.33],[0.14,0.45],[7.30,4.23]]
    train_label = [[0,1],[1,0],[0,1],[1,0]]

    test_a = [0.10,0.33]
    test_b = [4.0,4.5]

    # Train model (FNN)
    hidden_layer = [3,2]
    fnn = FNN(train_mat, train_label, hidden_layer)
    fnn.fit(lr = 0.01, epoch = 2000, err_th = 0.001, batch_size = 4)

    # Test model (FNN)
    res_a = np.rint(fnn.predict(test_a))
    res_b = np.rint(fnn.predict(test_b))

    print("X %s => Y %s" % (test_a, res_a))
    print("X %s => Y %s" % (test_b, res_b))

    ———————— output ————————

    X [0.1, 0.33] => Y [ 0.  1.]
    X [4.0, 4.5] => Y [ 1.  0.]

[See more examples here](https://github.com/becxer/pytrain/tree/master/examples)

## How to contribute

    Fork this repository, and write your algorithm, pull request.
    Don't forgot proper test code in test_pytrain.
    Test code should be work successfully in below command.
    
    $ python test.py
