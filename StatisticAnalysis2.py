
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


        #Se obtiene el DF con las medidas de centralidad
        #df = pd.read_csv(sys.argv[1], dtype="float64")
        #df = pd.read_csv(sys.argv[1], dtype="float64")

        df1 = pd.read_csv("data/centralidad/classifiersEvaluationMetrics.csv", dtype="float64")
        df2 = pd.read_csv("data/centralidad+twitter/classifiersEvaluationMetrics_twitter.csv", dtype="float64")
        df3 = pd.read_csv("data/twitter/classifiersEvaluationMetrics.csv", dtype="float64")


        accuracy_MLP=df1["MLP_accuracy"]
        accuracy_RF1=df2["RF_accuracy"]
        accuracy_RF2=df3["RF_accuracy"]


        results_list={"centralidad":accuracy_MLP,"centralidad_twitter":accuracy_RF1,"twitter":accuracy_RF2}
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

        f4 = open("data/comparativa/statisticsReportData.txt", 'a')
        f4.write(report)
        f4.write("\n")
        f4.close()