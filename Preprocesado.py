
import sys
#encoding: utf-8
from statistics import mean

sys.path.append("..")
#print(sys.path)
import pandas as pd

if __name__ == "__main__":

    if (len(sys.argv) - 1 != 0):
        print("This runnable executes the preproccess of a csv file. It "
              "generates a CSV file with the following columns: Id_A,Id_B,class. ")

    else:

        ##Step1: Load CSV file
        dfRaw=pd.read_csv("data/train.csv",encoding=  "utf-8")
        # dfRaw = pd.read_csv("ejemplo.csv",encoding=  "utf-8")

        print(len(dfRaw))
        # mediaA=mean([dfRaw["A_follower_count"],dfRaw["A_following_count"],dfRaw["A_listed_count"],dfRaw["A_mentions_received"],
        #             dfRaw["A_retweets_received"],dfRaw["A_mentions_sent"],dfRaw["A_retweets_sent"],dfRaw["A_follower_count"],
        #             dfRaw["A_posts"],dfRaw["A_network_feature_1"],dfRaw["A_network_feature_2"],dfRaw["A_network_feature_3"]])


                # listText=text.split(" ")
        # print(listText[0])
        listaTextos=[]
        iDdic=dict()
        idIndex=0
        idA=[]
        idB=[]
        clas=[]
        pares=set()
        dfReduced=dfRaw
        for i in range(len(dfRaw)):

            mediaA = dfRaw["A_follower_count"][i]+ dfRaw["A_following_count"][i] +dfRaw["A_listed_count"][i]+dfRaw["A_mentions_received"][i]+ dfRaw["A_retweets_received"][i]+dfRaw["A_mentions_sent"][i]+dfRaw["A_retweets_sent"][i]+dfRaw["A_follower_count"][i]+dfRaw["A_posts"][i]+dfRaw["A_network_feature_1"][i]+dfRaw["A_network_feature_2"][i]+dfRaw["A_network_feature_3"][i]
            mediaA=mediaA/12
            if(mediaA not in iDdic):
                iDdic[mediaA]=idIndex
                idIndex+=1

            mediaB= dfRaw["B_follower_count"][i]+ dfRaw["B_following_count"][i] +dfRaw["B_listed_count"][i]+dfRaw["B_mentions_received"][i]+ dfRaw["B_retweets_received"][i]+dfRaw["B_mentions_sent"][i]+dfRaw["B_retweets_sent"][i]+dfRaw["B_follower_count"][i]+dfRaw["B_posts"][i]+dfRaw["B_network_feature_1"][i]+dfRaw["B_network_feature_2"][i]+dfRaw["B_network_feature_3"][i]
            mediaB=mediaB/12

            if (mediaB not in iDdic):
                iDdic[mediaB] = idIndex
                idIndex += 1

            if(iDdic[mediaA]<=500 and iDdic[mediaB]<=500 and (mediaA,mediaB) not in pares):
                idA.append(iDdic[mediaA])
                idB.append(iDdic[mediaB])
                clas.append(dfRaw["Choice"][i])
                pares.add((mediaA,mediaB))
            else:
                dfReduced=dfReduced.drop(labels=i,axis=0)

        print(idA)
        print(idB)

        class1=sum(clas)
        class0=len(clas)-class1
        print("Elements of class 0: "+str(class0))
        print("Elements of class 1: "+str(class1))

        preprocesseData = {'idA': idA, 'idB': idB,"class":clas}
        df = pd.DataFrame(preprocesseData)
        print(str(idIndex))

        #df.to_csv(path_or_buf="data/preprocessedTrainReduced.csv", sep=',', encoding="utf-8",index=False)
        dfReduced.to_csv(path_or_buf="data/twitterDataReduced.csv", sep=',', encoding="utf-8",index=False)
