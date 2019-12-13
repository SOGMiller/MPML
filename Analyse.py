import Learn as lr 
import Data as dta
import Perspective as pst

import copy

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import svm


df = dta.load_data("/home/sean/Downloads/Research/Projects/MPML Library/botnet_train3.csv", 50000)

thresh = 5 
# Algo = GaussianNB()
# Algo = tree.DecisionTreeClassifier()
Algo = svm.SVC(gamma='scale',probability=True)

# dta.convert_discrete(df,thresh)

# perspectiveList = pst.generatePerspectives(df,"class")

# lr.MPML(df,GaussianNB(),"class",thresh,perspectiveList)

# Write a function that removes one perspective at a time and records the result without each.

def analysePerspective(DataFrame,target,instNum):

        print (df.iloc[instNum])
        
        models = []
        pridictions = []
        y = 0
        y_hat = 0
        d = 0

        y2 = 0
        y2_hat = 0
        d2 = [0,0]

        impactRatings = []

        perspectiveList = pst.generatePerspectives(DataFrame,target)
        # print ("\nResult with all Perspectives")
        models = lr.MPML(df,Algo,"class",thresh,perspectiveList)

        new_df = lr.instancePrediction(perspectiveList,target,models)
        y = lr.majorityVote(new_df)
        print("\n--------------------------------------------------------------------------------------")

         #this returns the avg confidence for the instance
        print ("Majority vote Acuracy with all perspective = {}".format(y))
        print ("Results for instance #{}".format(instNum))
        print ("1 is Not | 2 is Bot")
        print("\n======================================================")
        print (new_df.iloc[instNum])

        y2 = new_df.iloc[instNum][-1]


        for x in range(0,len(models)):
                x_train, x_test, y_train, y_test = dta.data_setup(perspectiveList[x],"class")                
                pridictions.append(accuracy_score(y_test,models[x].predict(x_test)))


        for i in range(0,len(models)):

                print("\n--------------------------------------------------------------------------------------")

                print ("Result without Perspective {}".format(i))
                
                modles2 = copy.deepcopy(models) 
                perspectiveList2 = copy.deepcopy(perspectiveList)  
                del modles2[i]
                del perspectiveList2[i]

                print("Persective {} Acuracy on it's own = ".format(i)+str(pridictions[i])+"")
                new_df = lr.instancePrediction(perspectiveList2,target,modles2)
                y_hat = lr.majorityVote(new_df)

                print ("Majority vote Acuracy without this perspective = {}".format(y_hat))

                d = y - y_hat
                impactRatings.append(d)
                print ("Majority vote Impact Score = {}".format(d))

                print ("Confidence Leverl without perspective {}".format(i))
                print (new_df.iloc[instNum][-1])

                y2_hat = new_df.iloc[instNum][-1]

                # print ()
                # lr.combinePerspectives (target,GaussianNB(),new_df)

        # print (max(impactRatings))

       # for i in range(0,len(models)):

                perspective = perspectiveList[i].drop(target, axis=1).values
                print("\n======================================================")
                print ("Current Perspetive Prediction and Confidence:")
                print ("Prediction = {}".format(models[i].predict(list((perspective[instNum]).reshape(1,-1)))[0]))
                print ("Confidence = {}".format((models[i].predict_proba(list((perspective[instNum]).reshape(1,-1)))[0])*100))

                print("\nConfidence impact score: y - ŷ = d")
                d2[0] = (y2[0] - y2_hat[0])
                d2[1] = (y2[1] - y2_hat[1])
                print (d2)

        analyseFeatures(perspectiveList,"class",Algo,instNum)

      

#Write a function that will give the confidence and pridiction results given a single model and instance.

def analyseFeatures(perspectiveList,target,clf,instNum):

        confidence = []
        y = 0
        y_hat = 0
        d = [0,0]

        for i in range(0,len(perspectiveList)):

                perspectiveList2 = perspectiveList

                print ("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print ("Confidence with all features")

                dta.convert_discrete(perspectiveList2[i],thresh)
                perspective0 = perspectiveList2[i].drop(target, axis=1).values

                x_train, x_test, y_train, y_test = dta.data_setup(perspectiveList2[i],target)

                clf = clf.fit(x_train,y_train)

                print (clf.predict(list((perspective0[instNum]).reshape(1,-1)))[0])
                confidence = clf.predict_proba(list((perspective0[instNum]).reshape(1,-1)))[0]*100

                print (confidence)

                y = confidence

                for feature in perspectiveList2[i].columns:

                        perspectiveList3 = copy.deepcopy(perspectiveList2)

                        

                        if (feature != target):  #  to not use class label as a feature
                                print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                                
                                # print (perspectiveList[i].columns)

                                print ("Confidence of P{} without feature - {}".format(i,feature))

                                del perspectiveList3[i][feature]

                                # print (perspectiveList3[i].columns)

                                dta.convert_discrete(perspectiveList3[i],thresh)
                                perspective = perspectiveList3[i].drop(target, axis=1).values

                                x_train, x_test, y_train, y_test = dta.data_setup(perspectiveList3[i],target)
                
                                clf = clf.fit(x_train,y_train)

                                # print (clf.predict(list((perspective[instNum]).reshape(1,-1)))[0])

                                y_hat = clf.predict_proba(list((perspective[instNum]).reshape(1,-1)))[0]*100

                                # print (clf.predict_proba(list((perspective[instNum]).reshape(1,-1)))[0]*100)


                                print("\nConfidence impact score: y - ŷ = d for features")
                                d[0] = (y[0] - y_hat[0]) 
                                d[1] = (y[1] - y_hat[1])
                                print (d)

                                print ("\n******************************************************************************")
                                print ("Relations of P{} features - {}".format(i,feature))

                                for f2 in perspectiveList3[i].columns: 
                                        relation = pst.getFeaturesRelations(feature,f2)
                                        print ("{} & {} => {}".format(feature,f2,relation))


analysePerspective(df,"class",345)


# Make this function more effecent by allowing it to not re-create all the models all the time. Find a way to build the modles once then iterate through them to get the result on each dataset. 