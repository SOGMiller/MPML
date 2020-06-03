import os
import argparse
import Perspective as pr
import Learn as lr
import Analyse as alz
import DrawGraph as dr

from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

import pickle
import numpy as np



path = os.getcwd()

parser = argparse.ArgumentParser(description= 'Multi-Perspective Machine Learning (MPML) A Machine Learning Model for Multi-faceted Learning problems')
parser.add_argument('file', type=str, help="The name of the CSV file - [file_name.csv]")
parser.add_argument('target', type=str, help="The name of the target column")
parser.add_argument('-ins','--instance', type=int, help="The number of instances to use by default 50,000 is used if it is less than that it uses all")
parser.add_argument('-si','--sig', action='store_true',  help="[Calcuate Significant Change Values]")
parser.add_argument('-rl','--grl', action='store_true',  help="[Generate Relations]")
parser.add_argument('-gp','--gen_per', action='store_true',  help="[Generate Perspectives]")
parser.add_argument('-a','--all', action='store_true',  help="Runs all [Calcuate Significan Change Values] [Generate Relations] and [Generate Perspectives] ")

parser.add_argument('-sv','--use_saves', action='store_true',  help="[Runs MPML with saved models]")
parser.add_argument('-cp','--custom_per', action='store_true', help="Create custom perspectives")

parser.add_argument('-rp','--remove_per', type=int, help="Remove the selectd perspective Number from the final modle [0 - ~]")
parser.add_argument('-az','--analyse', type=int, help="Enter instance index to be anlysed")
parser.add_argument('-azl','--analyseList', type=str, help="List of instances to analyse")
parser.add_argument('-ap','--add_per', action='store_true',  help="Add a perspective") 

parser.add_argument('-om','--overall', type=int, help="Enter instance index to be anlysed on the overall Model")

parser.add_argument('-load','--LoadModel', action='store_true', help="Load a saved model")

parser.add_argument('-l','--learn', action='store_true',  help="[Use MPML to learn a classifier]")
parser.add_argument('-al','--algo', type=str, help="The name of the clasifier to use [dt - Decision Tree] [svm - Support Verctor Machine] [nb - Gaussian Naive Bayes]")
parser.add_argument('-cb','--combine',action='store_true', help="The method used to combine all the perspectives [mv - majority vote]")
parser.add_argument('-v','--view', type=str, help="View [ft - List all features] [pft - features in perspecties] [pnum - number of perspecties] [pacc - Accuracy of each persective individually]")
args = parser.parse_args()


def select_custom_perspectives():
	selected_features = []
	temp_list = []
	feature_list = pr.viewFeatures(df,args.target)
	print ("------------Features-------------")  
	for i in range(0,len(feature_list)):
		print (feature_list[i] +" - {}".format(i))
	print ("---------------------------------")

	fea = input("\n Input Perspectives (Format 1,2,3-4,5,6-7,8,9): ")

	print (fea.split("-"))

	for per in fea.split("-"):
		for num in per.split(","):
			temp_list.append(feature_list[int(num)])
		print (temp_list)
		selected_features.append(temp_list)
		temp_list = []

	perspectiveList = pr.generatePerspectives(df,args.target,selected_features)

	return perspectiveList


if __name__ == '__main__':

	#Perspecctive.py---------------------------------------------------------------------
	print (path +"/"+ args.file)
	filePath = (path +"/"+ args.file)

	# print (args.analyseList)

	# lsy = args.analyseList.split(",")
	# print (lsy)

	# the value of pack push in p2 is a high indicator of bot activity but since it is the minority in its group its impact is -ve

	# write code to get impact of a single feature on the entire model

	if (args.instance == None):
		df = pr.LoadCSV(filePath)
	else:
		print (args.instance)
		df = pr.LoadCSV_instNum(filePath,args.instance)

	if (args.sig == True):
		print("==> Generating Significant change values")
		pr.sigValCsv(df,args.target)
		print("==> writing to SigChange.csv file")

	if (args.grl == True):
		print("==> Generating Relationship scores")
		pr.generateRelations(df,args.target)
		print("==> writing to Relations.csv file")

	if (args.gen_per == True):
		print("==> Generating Perspectives")
		if (args.custom_per == False):
			perspectiveList = pr.generatePerspectives(df,args.target)
		else:
			perspectiveList = select_custom_perspectives()


	if (args.all == True):
		print("==> Running ALL")
		pr.sigValCsv(df,args.target)
		pr.generateRelations(df,args.target)
		if (args.custom_per == False):
			perspectiveList = pr.generatePerspectives(df,args.target)
		else:
			perspectiveList = select_custom_perspectives()

	if (args.view == "ft"):
		print (pr.viewFeatures(df,args.target))

	if (args.view == "pft"):
		print("==> Generating Perspectives")
		for group in pr.viewPerspectives(df):
			print (group)

	if (args.view == "pnum"):
		print("==> Number of Perspectives - {}".format(pr.countPerspectives()))
  
  #Learn.py --------------------------------------------------------------------------
	algoName = -1
	algos = ["dt","svm","nb"]

	# print(args.algo)

	if(args.algo in algos):
		
		if(args.algo == "nb"):
			algoName = 2
			clf = GaussianNB()
		elif(args.algo == "dt"):
			algoName = 0
			clf = tree.DecisionTreeClassifier()
		elif(args.algo == "svm"):
			algoName = 1
			clf = svm.SVC(gamma='scale',probability=True)

	if(args.learn == True):
		if (args.custom_per == False):
			lr.MPML(df,clf,args.target,1,args.file)
		else:
			# perspectiveList = select_custom_perspectives()
			lr.MPML(df,clf,args.target,1,args.file,perspectiveList)
		

	if(args.learn == True and args.remove_per != None):
		print ("=======> Results with perspevtive removed <=======")
		lr.MPML(df,clf,args.target,1,args.file,[],args.remove_per)

	if(args.learn == True and args.combine == True):
		print ("Acuracy Using Majority Vote - {}".format(lr.majorityVote()))
  
  #Analyse.py --------------------------------------------------------------------------
	# if(args.use_saves):
		# alz.analysePerspective_savedModels()

	#Allow the user to view the acuracy of each perspective on its own.
	if(args.view == "pacc"):
		alz.individualAccuracy(df,clf,args.target)
		print ("Acuracy Using Majority Vote - {}".format(lr.majorityVote()))


	if(args.analyse != None):
		if (args.custom_per == False):
			alz.analysePerspective(df,args.target,args.analyse,clf, algos[algoName].upper(),args.file,False)
		else:
			alz.analysePerspective(df,args.target,args.analyse,clf,algos[algoName].upper(),args.file,False,perspectiveList)

	if(args.analyseList != None):
		for inst in args.analyseList.split(","):
			alz.analysePerspective(df,args.target,int(inst),clf, algos[algoName].upper(),args.file,False,perspectiveList)
		dr.quad_plot(args.analyseList.split(","),algos[algoName].upper())

	# Write a function that gets the relative impact of a feature on the entire model.

	if(args.LoadModel):
		with open('saves/perspective--P1 - botnet_train3_edit.csv','rb') as saved_file:
			mp = pickle.load(saved_file)

		# [ 3, 0,	0,	1634,	6667] - intance #55 change to [ 3, 0,	0,	1624,	3667] and got it wrong
		# [5, 0,	0,	2077,	53] - intance #94 change to [5, 0,	0,	1077,	553] and got it wrong

		instance = [5, 1,	14.2857,	1077,	553]
		print (np.array(instance))

		print (mp.predict(np.array(instance).reshape(1,-1)))

		








	 








