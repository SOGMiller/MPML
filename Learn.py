import numpy as np 
import Data as dta
import pandas as pd
import Perspective as pst 
import DrawGraph as dr
import copy
from statistics import mode 

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from statistics import mean 

import pickle
import pathlib

path = str(pathlib.Path(__file__).parent.absolute())
np.set_printoptions(precision=8,suppress=True)

Debug = False
instance_in_focus = 1
threshold_Value = 1

if (Debug):
    print("Sample Data from main dataframe:")
    #print(df.loc[[0]]) 
    #print(df.loc[[instance_in_focus]]) 
    print("\n")

#Write a function that returns the acuracy of the clasifier using the selected Algo in a traditinal manner
def algoAcuracy(df,clf,target):
    dta.convert_discrete(df,threshold_Value)
    x_train, x_test, y_train, y_test = dta.data_setup(df,target)
    clf = clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)
    return (accuracy_score(y_test,predictions))


# Write a function that when given a list of datafraes (Perspectives) will run a single ML algo on them and return the acuracy for each and the modles in a list.
def trainPersectives(perspectiveList,target,thresh,classifierAlgo,dataSet_name):
    print("==> Training Base Models...")

    count = 0

    modles = []
    clf = classifierAlgo
    if (Debug):
        print ("Perspectives Prediction Acuracy:")

    for dataF in perspectiveList:
        print ("==> Still Training...")
        dta.convert_discrete(dataF,thresh)
        x_train, x_test, y_train, y_test = dta.data_setup(dataF,target)
        clf = clf.fit(x_train,y_train)
        clf2 = copy.deepcopy(clf)
        modles.append(clf2)

        filename = "perspective--P{} - {}".format(count,dataSet_name)
        dr.save_models(clf2,filename)

        count+=1

        # if (Debug):
        predictions = clf.predict(x_test)
        print ("Perspective Predictions:")
        print (accuracy_score(y_test,predictions))

    if (Debug):
        print("\n")

    return modles


# Write a function that gets the instance predictions
def instancePrediction(perspectiveList,target,models):
    data = []
    name = []
    allAvgConfidenceLevels = []
    avgConfidence = [] #store the averge confidence level for all persectives


    for i in range(0,len(perspectiveList)):
        counter = 0
        perspect = perspectiveList[i].drop(target, axis=1).values
        name.append("perspect"+str(i))

        predictions = []
        confidence = [] #store the confidence level for each prediction
        
        for inst in perspect:
            predictions.append((models[i].predict(list((inst).reshape(1,-1)))[0]))
            confidence.append((models[i].predict_proba(list((inst).reshape(1,-1)))[0])*100)

            # if(i == 1):
            #     print ("Perspective {}".format(i))
            #     print (models[i].predict(list((inst).reshape(1,-1)))[0])
            #     print ((inst).reshape(1,-1))
 
            if (Debug):
                if(i == 0):
                    if (counter == instance_in_focus):
                        print ("Pridiction for given instance From a given model:")
                        print(models[0].predict(list((inst).reshape(1,-1)))[0])

                        print ("Confidence of single model:")
                        print((models[0].predict_proba(list((inst).reshape(1,-1)))[0])*100)
                        print(perspectiveList[0].loc[[instance_in_focus]]) 
                        print ("\n")
                
            counter+=1
        
        data.append(predictions)
        avgConfidence.append(confidence)

    allConfidenceleverls = list(zip(*avgConfidence))

    for i in range(0,len(allConfidenceleverls)):
        allAvgConfidenceLevels.append(list(map(mean,list(zip(*allConfidenceleverls[i]))))) 
        #Takes the avg confidence level for each 

    row2 = list(zip(*allAvgConfidenceLevels))
    row2 = list(row2[0])

    data.append(perspectiveList[0][target])
    data.append(allAvgConfidenceLevels)
    rows = list(zip(*data))
    name.append(target)
    name.append("confidence")
    new_df = pd.DataFrame(rows,columns=name)

    if (Debug):
        print("Perspecctives")
        print (new_df.head())
        print("Confidence for instance in focus")
        print(allAvgConfidenceLevels[instance_in_focus])
        print("\n")

    new_df.to_csv(path+"/MPML.csv",index=False)

    return new_df


# Write another function that when given a list of modles and the target values on which the modles were trained will learn how the modles interact to give the best result,
def combinePerspectives (target,algo,MPMLdf,dataSet_name):
    print("==> Combining Perspectives...") 

    # MPMLdf = pd.read_csv("/home/sean/Downloads/Research/Projects/MPML Library/MPML.csv")
    MPMLdf = MPMLdf.drop("confidence", axis=1)
    x_train, x_test, y_train, y_test = dta.data_setup(MPMLdf,target)

    clf = algo
    clf = clf.fit(x_train,y_train)

    dr.save_models(clf,"combined_perspectives - {}".format(dataSet_name))

    predictions = clf.predict(x_test)                                          
    print ("Acuracy Using Algo as Combination Method - {}".format(accuracy_score(y_test,predictions)))

    # return ("Acuracy Using Algo as Combination Method - {}".format(accuracy_score(y_test,predictions)))


def MPML(dataFrame,cmb,target,thresh,dataSet_name,perspectiveList=[],drop=-1):
   
    # cmb - combination method
    if(perspectiveList == []):
        perspectiveList = pst.generatePerspectives(dataFrame,target)

    if (drop != -1):
        del perspectiveList[drop]

    if (Debug):
        print (perspectiveList[0].columns)
        print (perspectiveList[1].columns)

    if (Debug):
        print ("Length of Perspective List = {}\n".format(len(perspectiveList)))

    themodles = trainPersectives(perspectiveList,target,thresh,cmb,dataSet_name) 


    print ("End MPML")

    # if (save):
        #save all the modles created and stored in the themodles
        # pass
    
    vote_df = instancePrediction(perspectiveList,target,themodles)
    combinePerspectives (target,cmb,vote_df,dataSet_name)
    print(majorityVote(vote_df))

    if (Debug):
        print ("Majority Vote:")
        # majorityVote(vote_df)

    return themodles, perspectiveList


def majorityVote(df=pd.DataFrame()):
    print("==> Calcualting Majority Vote... :)")
    if (df.empty):
        vote_df = pd.read_csv(path+"/MPML.csv")
    else:
        vote_df = df

    total = len(vote_df.values)
    right = 0
    wrong = 0 

    def getMajority(votes):
        counter = 0
        num = votes[0] 
        for i in votes: 
            curr_frequency = list(votes).count(i) 
            if(curr_frequency> counter): 
                counter = curr_frequency 
                num = i 
        return num 

    def getClass(votes):
        return votes[-2]

    count = 0
    for votes in vote_df.values:
        if (getMajority(votes[:-2]) == getClass(votes)):
            right += 1
            if (Debug):
                if (count == instance_in_focus):
                    print("Got it Right")
                    print ("Votes - {}".format(votes[:]))
                    print ("Actual - {}".format(getClass(votes)))
        else:
            wrong+=1 

            if (Debug):
                print ("Actual - {}".format(getClass(votes)))
                
        
        count+=1

    if (Debug):
        print ("Right - {}".format(right))
        print ("Wrong - {}".format(wrong))

    
    return float(right/total)


# algo = GaussianNB()
# dataFrame = pd.read_csv (path + '/botnet_train3.csv')
# MPML(dataFrame,algo,"class",1,"dataSet_name",perspectiveList=[])







