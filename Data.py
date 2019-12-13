import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

#Accepts a string as input and returns a pandas dataframe
def load_data(data):
    return pd.read_csv(data)

#Accepts a string and the number of rows to load as input and returns a pandas dataframe
def load_data(data, rows):
    return pd.read_csv(data, nrows=rows)

#Accepts a dataframe and a column name as parameter and converts the nominal values to an interger 
#Starting from 1 up to n, where n is the number of nominal values
def auto_int(dataFrame,col):
    NameList = dataFrame[col].unique() #Get the number of unique elements in a DataFrame 
    counter = 1
    for name in NameList:
        dataFrame[col] = dataFrame[col].replace([name], counter) #Replace Values in a dataframe with something else
        counter += 1

#Check if a column is discrete or continuous
#Find the percentage of number of unique values to the total number of values. If it is greater than the threshold (thrsh) 10% it
#is consedered to be continious. Returns False if the column is nominal and True if its descrete.
def is_discrete(dataFrame,col,thrsh = 1):
 
    UniqueValues =  len(dataFrame[col].unique())*1.0
    rows = dataFrame[col].shape[0]
    percent = (UniqueValues/rows)*100.00
    
    if percent > thrsh:
        return False
    return True

#This function converts all descrete valued columns to int given a dataframe and threshold percentage
#if the unique atribute value cont is > the threshold value then it will be maked as not descreet
def convert_discrete(the_df,thrsh):
    colmns = list(the_df)
    for col in colmns:
        if (is_discrete(the_df,col,thrsh)):
            auto_int(the_df,col)

# This function prepares a dataset for training by spliting it into x and y
def data_setup(dataFrame,target): 

    x = dataFrame.drop(target, axis=1) 
    y = dataFrame[target]

    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test






