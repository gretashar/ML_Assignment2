# import mlrose_hiive as mlrose
from cmath import inf
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import neural_network
from sklearn.model_selection import validation_curve
from learning_curve import single_valid, plot_learning_curve
from sklearn.model_selection import GridSearchCV
# import mlrose_hiive as mlrose

#Source: https://www.kaggle.com/code/hhllcks/neural-net-with-gridsearch/notebook
#https://www.kaggle.com/code/jamesleslie/titanic-neural-network-for-beginners/notebook
#https://www.kaggle.com/code/carlmcbrideellis/very-simple-neural-network-for-classification/notebook
def nn_test(df, output, title):

    X = df.drop([output], axis=1)
    y = df[output]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    cols = X_train.columns    
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])

    # print(X_train)
    # print(y_train)  
    # Define decay schedule for GA

    
    # Initialize neural network object and fit object
    start_GA = time.time()
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [10,10], activation = 'relu', \
                                    algorithm = 'genetic_alg', max_iters = 100000, \
                                    bias = True, is_classifier = True, learning_rate = 0.01, \
                                    early_stopping = True, clip_max = 5, max_attempts = 10, \
                                    random_state = 1, curve=True, pop_size=500)

    nn_model1.fit(X_train, y_train)
    
    end_GA = time.time()
    GA_time = end_GA-start_GA
    fitness_curve = nn_model1.fitness_curve
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train)

    y_train_accuracy = accuracy_score(y_train, y_train_pred)

    print("Y Train Accuracy:")
    print(y_train_accuracy)


    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test)
    
    print("Y Test Accuracy:")
    y_test_accuracy = accuracy_score(y_test, y_test_pred)

    print(y_test_accuracy)
    
    print("Training Classification Report:")
    print(classification_report(y_train, y_train_pred))
    print("Testing Classification Report:")
    print(classification_report(y_test, y_test_pred))

    
    x = np.arange(0, len(fitness_curve), dtype=int)

    # Plotting the Graph
    plt.plot(x, fitness_curve,linewidth=3.0, color="g")
    plt.grid()
    plt.title("Genetic Algorithm for Neural Network")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Values")
    plt.savefig('./plots/NN_GA.png')  
    plt.clf()


    print("\n*******************************************************\n")

    print('Timing for GA for NN: {:.6f}'. format(GA_time))
    print('Loss for GA: ')
    print(nn_model1.loss)

    print("\n*******************************************************\n")
    
    print(title+' GA Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_test_pred)))
    print(title+' GA Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_train_pred))) 
    print(title+' GA Model F1 score: {0:0.4f}'. format(f1_score(y_test, y_test_pred)))
    print(title+' GA Training-set F1 score: {0:0.4f}'. format(f1_score(y_train, y_train_pred))) 
    print(title+' GA Training set score: {:.4f}'.format(nn_model1.score(X_train, y_train)))
    print(title+' GA Test set score: {:.4f}'.format(nn_model1.score(X_test, y_test)))




    parameters = {'solver': ['adam'], 'max_iter': [500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':[(10,10), (10,10,10), (10,10,10,10)]}
    
    # parameters = {'solver': ['lbfgs','adam'], 'max_iter': [500,1000,1500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
    clf_grid = GridSearchCV(neural_network.MLPClassifier(), parameters, n_jobs=-1)
    clf_grid.fit(X,y)
    
    print("-----------------Original Features--------------------")
    print("Best score: %0.4f" % clf_grid.best_score_)
    print("Using the following parameters:")
    print(clf_grid.best_params_)
    
    
    estimator = clf_grid.best_estimator_   
        
    start_train = time.time()
    estimator.fit(X_train,y_train)
    end_train = time.time()
    train_time = end_train-start_train
    
    start_train_predict = time.time()
    y_train_pred = estimator.predict(X_train)
    end_train_predict = time.time()
    train_predict_time = end_train_predict-start_train_predict  
    
    start_test_predict = time.time()
    y_test_pred = estimator.predict(X_test)
    end_test_predict = time.time()
    test_predict_time = end_test_predict-start_test_predict
    
    print("\n*******************************************************\n")
    
    print(title+' Timing for Training: {:.6f}'. format(train_time))
    print(title+' Timing for Train Predict: {:.6f}'. format(train_predict_time))
    print(title+' Timing for Test Predict: {:.6f}'. format(test_predict_time))

    print("\n*******************************************************\n")

    
    print(title+' ANN Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_test_pred)))
    print(title+' ANN Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_train_pred))) 
    print(title+' ANN Model F1 score: {0:0.4f}'. format(f1_score(y_test, y_test_pred)))
    print(title+' ANN Training-set F1 score: {0:0.4f}'. format(f1_score(y_train, y_train_pred))) 
    print(title+' ANN Training set score: {:.4f}'.format(estimator.score(X_train, y_train)))
    print(title+' ANN Test set score: {:.4f}'.format(estimator.score(X_test, y_test)))




    
if __name__ == "__main__":
    
    data = "./data/heart.csv"
    df = pd.read_csv(data)
    nn_test(df, "output", "Heart Failure Prediction")
