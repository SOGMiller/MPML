import numpy as np
import Data as dta
import pandas as pd
import Classes as gp

from sklearn.feature_selection import RFE
from sklearn import tree
import copy

import pathlib

path = str(pathlib.Path(__file__).parent.absolute())

def LoadCSV(FilePath):
    dataFrame = dta.load_data(FilePath, 50000)
    dataFrame = dataFrame.fillna(0)
    return dataFrame

def LoadCSV_instNum(FilePath,instNum):
    dataFrame = dta.load_data(FilePath, instNum)
    dataFrame = dataFrame.fillna(0)
    return dataFrame

def viewFeatures(df,target):
    return list(df.drop([target], axis=1))


def getFeatures(df,target):
    #Get List of features (Column names)
    features = list(df.drop([target], axis=1))

    # Craete a list of features and their types (real= False or descrete =True) - formant - [["fname1",False],["fname2",True]]
    fTypes = []
    for col in features:
        fTypes.append([col,dta.is_discrete(df,col)])

    return fTypes

#This fuction calculates the average change that occurs between the instancs of real-valued featurs.
#That average change is called the significant change value which indecates the minimum value for which we will
#record a change for that particular feature
def cal_sig_change(dataFrame,feature):
    
    #Fill in blanks or nan with 0.0s
    dataFrame[feature] = dataFrame[feature].fillna(0.0)
    #get the length of the column
    col_length = len(dataFrame[feature].values)

    #The number of times a change is recorded is n-1 where n is the number of rows in that column
    #get the celeing of each change

    #List to store the Change values
    ChaValues = []

    #Use count to keep track so we don't go over the limit
    count = 0
    for val in dataFrame[feature]:
        #print (type(val))
        #print (val)
        if (count !=0 and count != len(dataFrame[feature])):
            ChaValues.append(abs(prev_val - val))
        count+=1
        prev_val = val
    
    return sum(ChaValues)/col_length

#This function generates a csv file with the real valued features and their significant change value
def sigValCsv(dataFrame,target):
    data = []
    for fea in getFeatures(dataFrame,target):
        if (fea[1] == False):
            data.append([fea[0],cal_sig_change(dataFrame,fea[0])])
    
    new_df = pd.DataFrame(data,columns=['Feature','Significant Value'])
    new_df.to_csv(path + "/SigChange.csv") 
    

#A function that generates a list of all possible pairs of features 
def pairWise(dataFrame,target):
    #Stores all posible pairs of features
    pairs = []

    #Get List of features (Column names)
    features = list(dataFrame.drop([target], axis=1))

    for fea in features:
        #Loop throught with the current feature "fea" fixed
        for fea2 in features[1:]:
            pairs.append([fea,fea2])
        #Remove the first element from the list
        features = features[1:]
    return pairs

#gets the significant change value of a particular feature 
def getSigValue(feature):
    
    sigDf = pd.read_csv(path + "/SigChange.csv")    

    #Count here is used to keep track of the index so we know what sig value we need to get
    count = 0
    for fea in sigDf["Feature"]:
        if (fea == feature):
            return sigDf["Significant Value"][count]
        count+=1
    return -1

#Returns true if the previous value cnaged with the current false otherwise. real is set to true if it is a valued feature and the sig value is passed in.
def didChange(curr,prev,real = False, sigVal = 0):
    if(real == True and sigVal != 0):
        if (abs(prev - curr) > sigVal):
            return True
        else:
            return False
    elif(curr != prev):
        return True
    else:
        return False

#Calculate type one scores - If the attribute values for both features change at the same time.
#Calculate type two scores - If the attribute values for both features change and the class label change at the same time.
def getScores(f1,f2,target,dataFrame):

    sigValf1 = -1
    sigValf2 = -1
    sigValTarget = -1

    score = 0
    score2 = 0
    length = len(dataFrame[f1])
    
    #if either feature of continous value then we would wnat to get the significant change value for that feature
    if(dta.is_discrete(dataFrame,f1)!= True):
        sigValf1 = getSigValue(f1) #Get the sigChange Value 
    if(dta.is_discrete(dataFrame,f2)!= True):
        sigValf2 = getSigValue(f2) #Get the sigChange Value 
    if(dta.is_discrete(dataFrame,target)!= True):
        sigValTarget = getSigValue(target) #Get the sigChange Value 

    #Both features are descrete
    if (sigValf1 == -1 and sigValf2 == -1):
        #print ("Case 1")
        #keep track of prevous values with a prev_indext variable
        prev_index = 0
        for i in range(1, length):
            if (didChange(dataFrame[f1][i],dataFrame[f1][prev_index]) and didChange(dataFrame[f2][i],dataFrame[f2][prev_index])):
                score+= 0.001

                #Check if the target is descrete valued 
                if(sigValTarget == -1):
                    #Check if the target changed as well
                    if(didChange(dataFrame[target][i],dataFrame[target][prev_index])):
                        score2+=1 #If it did update score2
                else: #if it is not descrete
                    #Check if the target changed as well
                    if(didChange(dataFrame[target][i],dataFrame[target][prev_index],real=True, sigVal=sigValTarget)):
                        score2+=1 #If it did update score2

            prev_index+=1

    #only f1 is descrete
    if (sigValf1 == -1 and sigValf2 != -1):
        #print ("Case 2")
        #keep track of prevous values with a prev_indext variable
        prev_index = 0
        for i in range(1, length):
            if (didChange(dataFrame[f1][i],dataFrame[f1][prev_index]) and didChange(dataFrame[f2][i],dataFrame[f2][prev_index],real=True, sigVal=sigValf2)):
                score+= 0.001

                #Check if the target is descrete valued 
                if(sigValTarget == -1):
                    #Check if the target changed as well
                    if(didChange(dataFrame[target][i],dataFrame[target][prev_index])):
                        score2+=1 #If it did update score2
                else: #if it is not descrete
                    #Check if the target changed as well
                    if(didChange(dataFrame[target][i],dataFrame[target][prev_index],real=True, sigVal=sigValTarget)):
                        score2+=1 #If it did update score2

            prev_index+=1

    #only f2 is descrete
    if (sigValf1 != -1 and sigValf2 == -1):
        #print ("Case 3")
        #keep track of prevous values with a prev_indext variable
        prev_index = 0
        for i in range(1, length):
            if (didChange(dataFrame[f1][i],dataFrame[f1][prev_index],real=True, sigVal=sigValf1) and didChange(dataFrame[f2][i],dataFrame[f2][prev_index])):
                score+= 0.001

                #Check if the target is descrete valued 
                if(sigValTarget == -1):
                    #Check if the target changed as well
                    if(didChange(dataFrame[target][i],dataFrame[target][prev_index])):
                        score2+=1 #If it did update score2
                else: #if it is not descrete
                    #Check if the target changed as well
                    if(didChange(dataFrame[target][i],dataFrame[target][prev_index],real=True, sigVal=sigValTarget)):
                        score2+=1 #If it did update score2
                        
            prev_index+=1

    #boath are real
    if (sigValf1 != -1 and sigValf2 != -1):
        #print ("Case 4")
        #keep track of prevous values with a prev_indext variable
        prev_index = 0
        for i in range(1, length):
            if (didChange(dataFrame[f1][i],dataFrame[f1][prev_index],real=True, sigVal=sigValf1) and didChange(dataFrame[f2][i],dataFrame[f2][prev_index],real=True, sigVal=sigValf2)):
                score+= 0.001

                # print("f1prev - {}\n f2prev - {}".format(dataFrame[f1][prev_index],dataFrame[f2][prev_index]))
                # print("f1i - {}\n f2i - {}".format(dataFrame[f1][i],dataFrame[f2][i]))
                # print ("i - {}\n prev_index - {}".format(i,prev_index))

                #Check if the target is descrete valued 
                if(sigValTarget == -1):
                    #Check if the target changed as well
                    if(didChange(dataFrame[target][i],dataFrame[target][prev_index])):
                        score2+=1 #If it did update score2
                else: #if it is not descrete
                    #Check if the target changed as well
                    if(didChange(dataFrame[target][i],dataFrame[target][prev_index],real=True, sigVal=sigValTarget)):
                        score2+=1 #If it did update score2

            prev_index+=1
    return score, score2

#Using the results of the pairwise fuction this fuction genates a relationsip score for every posible pair of features and stores it in a csv file
def generateRelations(dataFrame,target):
    #store the reltionships in this list then create a dataframe then store as csv [f1,f2,r-score]
    data = []
 
    length = len(pairWise(dataFrame,target))
    count = 1

    for pair in pairWise(dataFrame,target):
        score, score2 = getScores(pair[0],pair[1],target,dataFrame)

        data.append([pair[0],pair[1],sum([score,score2])])

        print ("{} of {}...".format(count,length))
        count+=1
    
    new_df = pd.DataFrame(data,columns=['Feature 1','Feature 2','Score'])
    new_df.to_csv(path + "/Relations.csv")


#This function returns the max relation given a list of relations in this format - ['Id', 'MSSubClass', 1613]
def max_score(scoresList):
    maxList = []
    currentMax = 0
    for r in scoresList:
        if (r[2]>currentMax):
            currentMax = r[2]
            maxList = r
    return maxList


#write a function to get all relations of a given feature
def get_relations(feature):
    location = path + "\Relations.csv"
    rdf = pd.read_csv(location)
    rdfNames = list(rdf)

    f1 = list(rdf[rdfNames[1]])
    f2 = list(rdf[rdfNames[2]])
    score = list(rdf[rdfNames[3]])

    rList = []

    for i in range(0,len(list(f1))):
        if (f1[i] == feature or f2[i] == feature):
            rList.append([f1[i],f2[i],score[i]])
    return rList


# Write a function to get the link between 2 given features
def getFeaturesRelations(feature1,feature2):
    location = path + "/Relations.csv"
    rdf = pd.read_csv(location)
    rdfNames = list(rdf)

    f1 = list(rdf[rdfNames[1]])
    f2 = list(rdf[rdfNames[2]])
    score = list(rdf[rdfNames[3]])

    for i in range(0,len(list(f1))):
        if ((f1[i] == feature1 and f2[i] == feature2) or (f1[i] == feature2 and f2[i] == feature1)):
            return score[i]
    return -1


#This function returns the feature with the strongest link for the given feature
def get_best_link(feature):
    # print(feature)
    lst = max_score(get_relations(feature))
    # print (lst)

    if (lst == []):
        return -1

    if (feature == lst[0]):
        return lst[1]
    else:
        return lst[0]
    return -1

#Wrrite a function to group features with the strongest links
def groupFeatures (dataFrame):
    print("==> Grouping Features...")

    #get a list of all features
    all_features = list(dataFrame)
    discard = []

    # print (all_features)

    #While the list of all features is not empty
    while (all_features!= []):

        #Pull the next feature from that list
        feature = all_features[0]
        print ("Curent Feature - {}".format(feature))

        #Get this feature's stongest link
        best_link = get_best_link(feature)
        print ("Best Link - {}".format(best_link))

        #Check if the strongest link is in a group
        print (gp.Groups.is_grouped(best_link))
        print (best_link)

        if(best_link == -1):
            all_features.remove(feature)
            discard.append(feature)

        elif(gp.Groups.is_grouped(best_link)):

            #get the group that it is in
            best_link_group = gp.Groups.get_group(best_link)

            #put this geature in the same group as its stongest link
            best_link_group.add_member(feature)

            #remove the feature from the list of all features
            all_features.remove(feature)
        
        else:
            #Create a group and put both features in it
            new_group = gp.Groups(feature)
            new_group.add_member(best_link)

            #remove both features from the main list of features
            all_features.remove(feature)
            # if (best_link in all_features):
            # print (all_features)
            all_features.remove(best_link)

# Write a function that when given a set of features that represents a single perspective returns a dataframe 
# with only those features and the data to go along with them. This function should also create a csv file of the 
# created Perspective.

def createPerspective (fList,dataFrame,targetName,file_name):
    features = []
    dropList = []

    for col in dataFrame.columns: 
        features.append(col)

    for f in features:
        if f in fList:
            pass #do nothing
        else:
            if (f != targetName):
                dropList.append(f)

    newDataFrame = dataFrame.drop(dropList, axis=1)

    newDataFrame.to_csv(path + "/" +file_name)

    return newDataFrame
            


# Create a function that generate all perspectives as a list of dataFrames given the list of all the 
# grouped features.

def generatePerspectives (dataFrame,targetName,selected_features=[]):

	groupFeatures(dataFrame)
	print("==> Generating Perspectives...")

	i = 0
	perspectiveList = []

	if(selected_features == []):
		
		groups = getTopFeatures(dataFrame,targetName)
		#gp.Groups.print_all_groups()
	else:
		groups = selected_features

	for group in groups:
		file_name = "perspect"+str(i)+".csv"
		perspectiveList.append(createPerspective (group,dataFrame,targetName,file_name))
		i+=1

	print("From Perspective.py")
	print (len(perspectiveList))
	print("From Groups")
	print (groups)

	return perspectiveList

def viewPerspectives(dataFrame):
    groupFeatures(dataFrame)
    groups = gp.Groups.print_all_groups()
    return groups 

#Write a function that couts the number of perspect[#].csv files to determine the number of perspectives
def countPerspectives():
    count = 0
    flag = 0

    while (flag != -1):
        try:
            f = open("perspect"+str(count)+".csv")
            count += 1
        except IOError:
            flag = -1
        finally:
            f.close()
            
    return count

# Write a function to accept a df and return the name (string) of the top ranking features
def getTopFeatures(dataFrame,targetName):
	algo = tree.DecisionTreeClassifier()

    # Insert code to - Add top features to the group if it is not already in it.
	dta.convert_discrete(dataFrame,1)
	x_train, x_test, y_train, y_test = dta.data_setup(dataFrame,targetName)

	rfe = RFE(estimator=algo,step=1)
	rfe = rfe.fit(x_train,y_train)

	selected_rfe_features_df = pd.DataFrame({'Feature': list(x_train.columns),'Ranking':rfe.ranking_})

	rate_list = list(selected_rfe_features_df.values)

	oldGroups = copy.deepcopy(gp.Groups.print_all_groups())
	groups = gp.Groups.print_all_groups()

	def getFeatureRating(f_list):
		return f_list[1]

	def getFeatureName(f_list):
		return f_list[0]

	top_features = []

	for f_list in rate_list:
		if(getFeatureRating(f_list) == 1 ):
			top_features.append(getFeatureName(f_list))

	
	for feature in top_features:
		print("Best Link for {} is {}".format(feature,get_best_link(feature)))
		if(get_best_link(feature) not in top_features):
			top_features.remove(feature)

	for per in groups:
		for feature in top_features:
			if(feature not in per):
				per.append(feature)

	groups.append(top_features)

	for OG in oldGroups:
		groups.append(OG)
	

	print ("Top Features")
	print (groups)

	return groups



# dataFrame = pd.read_csv (path + '/botnet_train3.csv')
# sigValCsv(dataFrame,"class")
# generateRelations(dataFrame,"class")
# generatePerspectives (dataFrame,"class")


 