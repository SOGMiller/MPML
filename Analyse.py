import Learn as lr 
import Data as dta
import Perspective as pst
import copy
import DrawGraph as dg

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import svm

import matplotlib.pyplot as plt
import os

import pathlib
path = str(pathlib.Path(__file__).parent.absolute())

thresh = 1 

# Write a function that removes one perspective at a time and records the result without each.
def analysePerspective(DataFrame,target,instNum,Algo,algo_name,dataSet_name,summary=True,perspectives=[]):

	

	title = "Instance #{} with {}".format(instNum,algo_name)
	folder_name = "saves/{}".format(title)

	dir = os.path.join("saves/{}".format(title))
	if not os.path.exists(dir): 
		os.mkdir(dir)
	file = open(folder_name+"/"+title+"_model_output.txt","w+",encoding='utf-8')

	actual_class = -1

	file.write (str(DataFrame.iloc[instNum]))
	print ("\n")
	
	models = []
	pridictions = []
	y = 0
	y_hat = 0
	d = 0

	y2 = 0
	y2_hat = 0
	d2 = [0,0]

	impactRatings = []

	# perspectiveList = pst.generatePerspectives(DataFrame,target)
	# print ("\nResult with all Perspectives")
	if (perspectives == []):
		models, perspectiveList = lr.MPML(DataFrame,Algo,target,thresh,dataSet_name)
	else:
		models, perspectiveList = lr.MPML(DataFrame,Algo,target,thresh,dataSet_name,perspectives)

	print ("Analysing Instance!...")
	new_df = lr.instancePrediction(perspectiveList,target,models)
	# y = lr.majorityVote(new_df)

	file.write("\n--------------------------------------------------------------------------------------")

	#this returns the avg confidence for the instance
	file.write ("\nMajority vote Acuracy with all perspective = {}".format(y))
	file.write ("\nResults for instance #{}".format(instNum))
	file.write ("\n1 is Not | 2 is Bot")
	file.write("\n======================================================")
	# file.write(Algo)
	file.write ("\n"+str(new_df.iloc[instNum])) # -2 shows the actual class
	actual_class = new_df.iloc[instNum][-2]

	print ("Actual Class at line 67 - ")
	print (actual_class)

	y2 = new_df.iloc[instNum][-1]

	print ("Settig Up Predictions")
	for x in range(0,len(models)):
		x_train, x_test, y_train, y_test = dta.data_setup(perspectiveList[x],target)                
		pridictions.append(accuracy_score(y_test,models[x].predict(x_test)))

	for i in range(0,len(models)):

		file.write("\n--------------------------------------------------------------------------------------")

		file.write ("\n")
		file.write ("Result without Perspective {}".format(i))
		file.write ("\n")
		
		modles2 = copy.deepcopy(models) 
		perspectiveList2 = copy.deepcopy(perspectiveList)  
		del modles2[i]
		del perspectiveList2[i]

		file.write("\nPersective {} Acuracy - ".format(i)+str(pridictions[i])+"")

		new_df = lr.instancePrediction(perspectiveList2,target,modles2)

		if(summary == False):
			# y_hat = lr.majorityVote(new_df)
			# file.write ("\nMajority vote Acuracy without this perspective = {}".format(y_hat))

			d = y - y_hat
			impactRatings.append(d)
			file.write ("\nMajority vote Impact Score = {}".format(d))

		file.write ("\nConfidence Level without perspective {}".format(i))
		file.write (str(new_df.iloc[instNum][-1]))

		y2_hat = new_df.iloc[instNum][-1]

		# print ()
		# lr.combinePerspectives (target,GaussianNB(),new_df)

		if(summary == False):
			perspective = perspectiveList[i].drop(target, axis=1).values
			file.write("\n======================================================")
			file.write ("\nCurrent Perspetive Prediction and Confidence:")
			file.write ("\nPrediction = {}".format(models[i].predict(list((perspective[instNum]).reshape(1,-1)))[0]))
			file.write ("\nConfidence = {}".format((models[i].predict_proba(list((perspective[instNum]).reshape(1,-1)))[0])*100))

			# file.write (lr.combinePerspectives(target,Algo,DataFrame,dataSet_name))


			file.write("\nConfidence impact score: y - ŷ = d")
			d2[0] = (y2[0] - y2_hat[0])
			d2[1] = (y2[1] - y2_hat[1])
			file.write ("\n"+str(d2))

	if(summary == False):
		analyseFeatures(perspectiveList,target,Algo,instNum,actual_class,algo_name,file)

      
#Write a function that will give the confidence and pridiction results given a single model and instance.
def analyseFeatures(perspectiveList,target,clf,instNum,actual_class,algo_name,file):

	feature_impact_list = [] #stores the featurs and their impact scores [[avg_byte,0.3390], [protocol,0.0114]
	confidence = []
	y = 0
	y_hat = 0
	d = [0,0]

	for i in range(0,len(perspectiveList)):

		perspectiveList2 = perspectiveList

		file.write ("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
		file.write ("\nConfidence with all features")

		dta.convert_discrete(perspectiveList2[i],thresh)
		perspective0 = perspectiveList2[i].drop(target, axis=1).values

		x_train, x_test, y_train, y_test = dta.data_setup(perspectiveList2[i],target)

		clf = clf.fit(x_train,y_train)

		file.write ("\n"+str(clf.predict(list((perspective0[instNum]).reshape(1,-1)))[0]))
		confidence = clf.predict_proba(list((perspective0[instNum]).reshape(1,-1)))[0]*100

		print ("This is what is being appended to the confidence list - line 155")
		print (clf.predict_proba(list((perspective0[instNum]).reshape(1,-1))))

		file.write ("\n"+str(confidence))

		print ("This is what is in y")
		print (y)

		y = confidence

		for feature in perspectiveList2[i].columns:
			perspectiveList3 = copy.deepcopy(perspectiveList2)
            
			if (feature != target):  #  to not use class label as a feature
				file.write("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                                
				# print (perspectiveList[i].columns)
				print ("Analysing Feature {}".format(feature))

				file.write ("\nConfidence of P{} without feature - {}".format(i,feature))
				del perspectiveList3[i][feature]

				# print (perspectiveList3[i].columns)

				dta.convert_discrete(perspectiveList3[i],thresh)
				perspective = perspectiveList3[i].drop(target, axis=1).values

				x_train, x_test, y_train, y_test = dta.data_setup(perspectiveList3[i],target)

				clf = clf.fit(x_train,y_train)

				#confidence without curent feature
				# print (clf.predict(list((perspective[instNum]).reshape(1,-1)))[0])

				y_hat = clf.predict_proba(list((perspective[instNum]).reshape(1,-1)))[0]*100
				file.write ("\n"+str(clf.predict_proba(list((perspective[instNum]).reshape(1,-1)))[0]*100))

				print ("This is what is in y_hat")
				print (y_hat)

				file.write("\nConfidence impact score: y - ŷ = d for features")
				d[0] = (y[0] - y_hat[0]) 
				d[1] = (y[1] - y_hat[1])
				file.write ("\nthis is d -> {}".format(d))

				print ("Actual Class at line 191 - ")
				print (actual_class)
				print (d)
				print (d[int(actual_class -1)])

				feature_impact_list.append(["P{}".format(i) + "-" +feature,d[int(actual_class -1)]])

				file.write ("\n******************************************************************************")
				file.write ("Relations of P{} features - {}".format(i,feature))

				for f2 in perspectiveList3[i].columns: 
					relation = pst.getFeaturesRelations(feature,f2)
					file.write ("\n{} & {} => {}".format(feature,f2,relation))

				file.write ("\n ==> Feature impact list <== \n")
				file.write ("\n"+str(feature_impact_list)+"\n")

	title = "Instance #{} with {}".format(instNum,algo_name)
	folder_name = "saves/{}".format(title)

	graph_title = "Instance #{} Impact Signiture with {}".format(instNum,algo_name)
	print (title)

	dg.plotGraph(feature_impact_list,graph_title,folder_name)
	file.close()


def individualAccuracy(DataFrame,clf,target,perspectiveList=[]):
	pridictions = []

	if(perspectiveList==[]):
		models, perspectiveList = lr.MPML(DataFrame,clf,target,1,"Name")
	else:
		models, perspectiveList = lr.MPML(DataFrame,clf,target,1,perspectiveList)

	for x in range(0,len(models)):
		x_train, x_test, y_train, y_test = dta.data_setup(perspectiveList[x],target)                
		pridictions.append(accuracy_score(y_test,models[x].predict(x_test)))
		print("Persective {} Accuracy on it's own = ".format(x)+str(pridictions[x])+"")

# import pandas as pd
# algo = GaussianNB()
# dataFrame = pd.read_csv (path + '/botnet_train3.csv')
# analysePerspective(dataFrame,"class",28,algo,"NB","Data",False)

 

