import numpy as np
import Data as dta
import pandas as pd
import Perspective as pst
import copy
from statistics import mode 

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from statistics import mean 

np.set_printoptions(precision=8,suppress=True)

Debug = False
instance_in_focus = 1
threshold_Value = 1

# # Data 1 is not 2 is bot
df = dta.load_data("/home/sean/Downloads/Research/Projects/MPML Library/botnet_train3.csv", 50000)
# df = df.fillna(0)

Algo = GaussianNB()
# Algo = tree.DecisionTreeClassifier()
# Algo = svm.SVC(gamma='scale',probability=True)

if (Debug):
    print("Sample Data from main dataframe:")
    print(df.loc[[0]]) 
    print(df.loc[[instance_in_focus]]) 
    print("\n")

# clf = tree.DecisionTreeClassifier()
# clf = svm.SVC(gamma='scale')
# clf = KNeighborsClassifier(n_neighbors=3)
clf = GaussianNB()

dta.convert_discrete(df,threshold_Value)
x_train, x_test, y_train, y_test = dta.data_setup(df,"class")

clf = clf.fit(x_train,y_train)

predictions = clf.predict(x_test)


#------------------------------------------------------------------------------------------------------------------

# print (x_train.corr())

# ddd = x_train.corr()

# print (ddd["avg_byte"])
# print (ddd["protocol"])
# print (ddd["dst_port"])
# print (ddd["percent_push"])

#------------------------------------------------------------------------------------------------------------------

# print (len(predictions))
# if (Debug):
print ("Original Prediction Acuracy:")
print (accuracy_score(y_test,predictions))
print("\n")


# Write a function that when given a list of datafraes (Perspectives) will run a single ML algo on them and return the acuracy for each and the modles in a list.
def trainPersectives(perspectiveList,target,thresh,classifierAlgo):

    modles = []
    clf = classifierAlgo

    if (Debug):
        print ("Perspectives Prediction Acuracy:")

    for dataF in perspectiveList:
        dta.convert_discrete(dataF,thresh)
        x_train, x_test, y_train, y_test = dta.data_setup(dataF,target)
        
        clf = clf.fit(x_train,y_train)

        clf2 = copy.deepcopy(clf)
        modles.append(clf2)

        predictions = clf.predict(x_test)

        # print (len(predictions))

        if (Debug):
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
        
        #Takes the avg confidence leerl for each 


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

    new_df.to_csv("/home/sean/Downloads/Research/Projects/MPML Library/MPML.csv",index=False)

    return new_df


# Write another function that when given a list of modles and the target values on which the modles were trained will learn how the modles interact to give the best result,

def combinePerspectives (target,algo,MPMLdf):
    # MPMLdf = pd.read_csv("/home/sean/Downloads/Research/Projects/MPML Library/MPML.csv")
    MPMLdf = MPMLdf.drop("confidence", axis=1)
    x_train, x_test, y_train, y_test = dta.data_setup(MPMLdf,target)

    clf = algo
    clf = clf.fit(x_train,y_train)

    predictions = clf.predict(x_test)

    # if (Debug):                                           
    print ("Acuracy Using Algo as Combination Method:")
    print (accuracy_score(y_test,predictions))
    print("\n")


def MPML(dataFrame,cmb,target,thresh,perspectiveList=[]): 

    # cmb - combination method
    if(perspectiveList == []):
        perspectiveList = pst.generatePerspectives(dataFrame,target)

    # del perspectiveList[0]['src_port']

    if (Debug):
        print (perspectiveList[0].columns)
        print (perspectiveList[1].columns)
        # print (perspectiveList[2].columns)

    # del perspectiveList[1]

    if (Debug):
        print ("Length of Perspective List = {}\n".format(len(perspectiveList)))
        

    themodles = trainPersectives(perspectiveList,target,thresh,cmb) 

    vote_df = instancePrediction(perspectiveList,target,themodles)

    combinePerspectives (target,cmb,vote_df)

    if (Debug):
        print ("Majority Vote:")
        majorityVote(vote_df)

    return themodles


def majorityVote(vote_df):

    # vote_df = pd.read_csv("/home/sean/Downloads/Research/Projects/MPML Library/MPML.csv")

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

        # print ("Before - {}".format(votes))
        
        if (getMajority(votes[:-2]) == getClass(votes)):
            right += 1
            if (Debug):
                if (count == instance_in_focus):
                    print("Got it Right")
                    print ("Votes - {}".format(votes[:]))
                    print ("Actual - {}".format(getClass(votes)))
        else:
            wrong+= 1

            # if (Debug):
            #     if (votes[-2]==2):
            #         print("Got it Wrong")
            #         print (count)

                # if (count == instance_in_focus):
                #     print("Got it Wrong")
                #     print ("Votes - {}".format(votes[:]))
                #     print ("Actual - {}".format(getClass(votes)))
        
        count+=1

    # if (Debug):
    # print ("Majority vote score:\n")
    # print ("Total - "+str(total) )
    # print ("Right - "+str(right) )
    # print ("Wrong - "+str(wrong))
    # print ("Acuracy - "+str(float(right/total)))

    # print ("Majority vote Acuracy - "+str(float(right/total)))

    return float(right/total)



# Try diffrent combination methods for diffrent algos and complete the libray by making it such that puting everyting in one fuction and run it.

MPML(df,Algo,"class",threshold_Value)
# MPML(df,tree.DecisionTreeClassifier(),"class",threshold_Value)

# majorityVote()
