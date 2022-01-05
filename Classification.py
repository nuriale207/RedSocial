
import sys

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from numpy import genfromtxt
if __name__ == "__main__":

    # if (len(sys.argv) - 1 != 2):
    #     print("This runnable generates a vocabulary csv file from the TEXT column of a given csv ")
    #     print("Parameter 1: path to the csv file. It contains the aforementioned column.")
    #     print("Parameter 2: path to save the vocabulary csv file.")

    #Se obtiene el DF con las medidas de centralidad
    df = pd.read_csv("data/caracteristicsTrainReduced.csv", dtype="float64")

    #Se separan las clases
    classes=df["class"]
    print(classes)

    #Se elimina la columna clase del DataFrame
    train_data=df.drop(['class'],axis=1)

    #Se convierten los datos de entrenamiento a np array eliminando los Nan values
    train_data=train_data.to_numpy(dtype='float', na_value=0)
    print(train_data)

    #CLASIFICADOR 1: Naive Bayes

    NB_results="RESULTADOS NAIVE BAYES \n\n"
    nb_clf=GaussianNB()
    nb_clf.fit(train_data, np.array(classes))

    scores = cross_validate(nb_clf, train_data, np.array(classes), cv=10, scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'))

    accuracy_nb=np.mean(scores['test_accuracy'])
    NB_results+="Accuracy: "+str(accuracy_nb) +"\n"

    precision_nb=np.mean(scores['test_precision'])
    NB_results+="Precision: "+str(precision_nb) +"\n"

    recall_nb=np.mean(scores['test_recall'])
    NB_results+="Recall: "+str(recall_nb) +"\n"

    f_measure_nb=np.mean(scores['test_f1'])
    NB_results+="F-measure: "+str(f_measure_nb) +"\n"

    roc_auc_nb=np.mean(scores['test_roc_auc'])
    NB_results+="ROC_auc: "+str(roc_auc_nb) +"\n"

    print(NB_results)
    #CLASIFICADOR 2: KNN

    KNN_results="RESULTADOS KNN \n\n"

    best_accuracy=0
    bestK=0
    bestScore=[]
    for i in range(1,100):
        knn_clf=KNeighborsClassifier(n_neighbors=i)
        knn_clf.fit(train_data, np.array(classes))

        scores = cross_validate(knn_clf, train_data, np.array(classes), cv=10,
                                scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'))

        accuracy_knn=np.mean(scores["test_accuracy"])
        if(accuracy_knn>best_accuracy):
            best_accuracy=accuracy_knn
            bestK=i
            bestScore=scores

    print(best_accuracy)
    print(bestK)
    KNN_results+="Best K: "+str(bestK)+"\n"

    accuracy_kn = np.mean(bestScore['test_accuracy'])
    KNN_results += "Accuracy: " + str(accuracy_kn) + "\n"

    precision_kn= np.mean(bestScore['test_precision'])
    KNN_results += "Precision: " + str(precision_kn) + "\n"

    recall_kn = np.mean(bestScore['test_recall'])
    KNN_results += "Recall: " + str(recall_kn) + "\n"

    f_measure_kn = np.mean(bestScore['test_f1'])
    KNN_results += "F-measure: " + str(f_measure_kn) + "\n"

    roc_auc_kn = np.mean(bestScore['test_roc_auc'])
    KNN_results += "ROC_auc: " + str(roc_auc_kn) + "\n"

    print(KNN_results)

    #CLASIFICADOR 3: MULTILAYER PERCEPTRON

    MLP_results = "RESULTADOS MULTILAYER PERCEPTRON \n\n"
    confList=[(100,),(100,50),(100,75,50),(100,75,50,25),(100,75,50,25,12),(100,80,60,40,20,10),(100,80,60,40,20,10,5)]
    best_accuracy = 0
    bestConf=()
    bestScore=[]
    for conf in confList:
        MLP_clf=MLPClassifier(random_state=1, max_iter=300,hidden_layer_sizes=conf,activation="relu")
        scores = cross_validate(MLP_clf, train_data, np.array(classes), cv=10,
                                     scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'))
        accuracy=np.mean(scores['test_accuracy'])

        if(accuracy>best_accuracy):
            best_accuracy=accuracy
            bestConf=conf
            bestScore=scores

    MLP_results += "Best Configuration: " + str(bestConf) + "\n"

    accuracy_mlp = np.mean(bestScore['test_accuracy'])
    MLP_results += "Accuracy: " + str(accuracy_mlp) + "\n"

    precision_mlp = np.mean(bestScore['test_precision'])
    MLP_results += "Precision: " + str(precision_mlp) + "\n"

    recall_mlp = np.mean(bestScore['test_recall'])
    MLP_results += "Recall: " + str(recall_mlp) + "\n"

    f_measure_mlp = np.mean(bestScore['test_f1'])
    MLP_results += "F-measure: " + str(f_measure_mlp) + "\n"

    roc_auc_mlp = np.mean(bestScore['test_roc_auc'])
    MLP_results += "ROC_auc: " + str(roc_auc_mlp) + "\n"

    print(MLP_results)

    #CLASIFICADOR 4: RANDOM FOREST

    RF_results = "RESULTADOS RANDOM FOREST \n\n"
    rf_clf = RandomForestClassifier(max_depth=2)
    rf_clf.fit(train_data, np.array(classes))

    scores = cross_validate(rf_clf, train_data, np.array(classes), cv=10,
                            scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'))

    accuracy_nb = np.mean(scores['test_accuracy'])
    RF_results += "Accuracy: " + str(accuracy_nb) + "\n"

    precision_nb = np.mean(scores['test_precision'])
    RF_results += "Precision: " + str(precision_nb) + "\n"

    recall_nb = np.mean(scores['test_recall'])
    RF_results += "Recall: " + str(recall_nb) + "\n"

    f_measure_nb = np.mean(scores['test_f1'])
    RF_results += "F-measure: " + str(f_measure_nb) + "\n"

    roc_auc_nb = np.mean(scores['test_roc_auc'])
    RF_results += "ROC_auc: " + str(roc_auc_nb) + "\n"

    print(RF_results)


