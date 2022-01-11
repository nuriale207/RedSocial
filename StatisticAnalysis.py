
import sys

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from numpy import genfromtxt

from scipy import stats
from scipy import stats

if __name__ == "__main__":

    if (len(sys.argv) - 1 != 2):
        print("This runnable makes the statistical analysis of the classifiers results  ")
        print("Parameter 1: path to the classifiers evaluation metrics. It contains at least  4 columns named NB_accuracy,RF_accuracy,KNN_accuracy and MLP_accuracy")
        print("Parameter 2: path to save the generated files: a statistical test report")

    else:
        #Se obtiene el DF con las medidas de centralidad
        #df = pd.read_csv(sys.argv[1], dtype="float64")
        df = pd.read_csv(sys.argv[1], dtype="float64")

        accuracy_RF=df["RF_accuracy"]

        accuracy_NB=df["NB_accuracy"]

        accuracy_KNN=df["KNN_accuracy"]

        accuracy_MLP=df["MLP_accuracy"]

        accuracy_DT=df["DT_accuracy"]

        results_list={"RF_accuracy":accuracy_RF,"NB_accuracy":accuracy_NB,"KNN_accuracy":accuracy_KNN,
                      "MLP_accuracy":accuracy_MLP,"DT_accuracy":accuracy_DT}
        report=""
        for i in range(len(results_list.values())):
            for j in range(i+1,len(results_list.values())):
                report+="SHAPIRO TEST\n\n"

                shapiro1=stats.shapiro(list(results_list.values())[i])
                report+= list(results_list.keys())[i]+": "+str(shapiro1)+"\n"
                shapiro2 = stats.shapiro(list(results_list.values())[j])
                report+= list(results_list.keys())[j]+": "+str(shapiro2)+"\n"

                if(shapiro1.pvalue>0.05 and shapiro2.pvalue>0.05):
                    anova_test = f_oneway(list(results_list.values())[i], list(results_list.values())[j])
                    report += "ANOVA TEST between: "+list(results_list.keys())[i]+" and "+list(results_list.keys())[j]
                    report+=" : "+str(anova_test)+"\n\n"
                else:
                    kruskal_test=stats.kruskal(list(results_list.values())[i], list(results_list.values())[j])
                    report += "KRUSKAL TEST between: " + list(results_list.keys())[i] + " and " + \
                              list(results_list.keys())[j]
                    report += " : " + str(kruskal_test) + "\n\n"


        print(report)

        f4 = open(sys.argv[2] + "/statisticsReport.txt", 'a')
        f4.write(report)
        f4.write("\n")
        f4.close()