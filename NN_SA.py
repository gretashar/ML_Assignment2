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
    # Define decay schedule for SA
    schedule = mlrose.ExpDecay()
    
    # Initialize neural network object and fit object
    start_SA = time.time()
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [10,10], activation = 'relu', \
                                    algorithm = 'simulated_annealing', max_iters = 100000, \
                                    bias = True, is_classifier = True, learning_rate = 0.01, \
                                    early_stopping = True, clip_max = 5, max_attempts = 10, \
                                    random_state = 1, curve=True, schedule=schedule)

    nn_model1.fit(X_train, y_train)
    
    end_SA = time.time()
    SA_time = end_SA-start_SA
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
    plt.plot(x, fitness_curve,linewidth=3.0, color="orange")
    plt.grid()
    plt.title("Simulated Annealing for Neural Network")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Values")
    plt.savefig('./plots/NN_SA.png')  
    plt.clf()


    print("\n*******************************************************\n")

    print('Timing for SA for NN: {:.6f}'. format(SA_time))
    print('Loss for SA: ')
    print(nn_model1.loss)

    print("\n*******************************************************\n")
    
    print(title+' SA Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_test_pred)))
    print(title+' SA Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_train_pred))) 
    print(title+' SA Model F1 score: {0:0.4f}'. format(f1_score(y_test, y_test_pred)))
    print(title+' SA Training-set F1 score: {0:0.4f}'. format(f1_score(y_train, y_train_pred))) 
    print(title+' SA Training set score: {:.4f}'.format(nn_model1.score(X_train, y_train)))
    print(title+' SA Test set score: {:.4f}'.format(nn_model1.score(X_test, y_test)))
  
if __name__ == "__main__":
    
    data = "./data/heart.csv"
    df = pd.read_csv(data)
    nn_test(df, "output", "Heart Failure Prediction")
