"""Second attempt of solving Kaggle's Titanic machine learning problem using an ensemble of Logistic Regression
and Gradient Boosting Classifier
Author : Leticia Wanderley
Date : 25 November 2015
"""

import pandas as pd
import re #import regular expression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

"""Extracts passengers titles from their names then store this titles in a new column on the dataframe
"""
def set_titles(data_frame):
	data_frame['Title'] = data_frame.Name.map(lambda x : re.search(' ([A-Za-z]+)\.', x).group(1)).astype(str) #creates a new column named Title and filled with the title extracted from the name 
	#print(data_frame['Title'].unique()) print all the titles present on the data 
	title = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 5, "Major": 6, "Col": 6, "Mlle": 7, 
	"Mme": 7, "Don": 8, "Lady": 9, "Countess": 9, "Jonkheer": 9, "Sir": 8, "Capt": 6, "Ms": 1, "Dona": 9} #maps each title to an integer, some titles are very rare and are compressed into the same codes as other titles (title 'Dona' is only present on the test set)
	data_frame['TitleCode'] = data_frame.Title.map(title).astype(int) #creates a new column named TitleCode containing the respective code for the row's title		

"""Fill the missing ages based on the median age of people with similar titles
"""
def fill_age_title_based(data_frame):
	set_titles(data_frame)
	median_ages = np.zeros(10) #initializes a array filled with 10 zeros to store the titles median age
	for i in range(0,10):
		median_ages[i] = data_frame[data_frame['TitleCode'] == i]['Age'].dropna().median()

	data_frame['AgeFill'] = data_frame.Age #creates a new column which is a copy of the Age column 
	#print(data_frame[data_frame['Age'].isnull()][['Title','Age','AgeFill']].head(10)) prints the 10 first rows which have a null value as age (just for checking)
	for j in range(0,10):
		data_frame.loc[(data_frame.Age.isnull()) & (data_frame.TitleCode == j), 'AgeFill'] = median_ages[j] #fills each row with null value as age with the median of its title

"""Creates a new column which will store the gender of each passenger as an int
"""
def convert_sex_column_int(data_frame):
	data_frame['Gender'] = data_frame.Sex 
	data_frame.loc[data_frame["Sex"] == "male", "Gender"] = 0
	data_frame.loc[data_frame["Sex"] == "female", "Gender"] = 1

"""Creates a new column which will store the code of the embarcation port of each passenger as an int
"""
def convert_embarked_column_int(data_frame):
	data_frame["EmbarkedCode"] = data_frame.Embarked
	data_frame["EmbarkedCode"] = data_frame["EmbarkedCode"].fillna("S")
	data_frame.loc[data_frame["EmbarkedCode"] == "S", "EmbarkedCode"] = 0
	data_frame.loc[data_frame["EmbarkedCode"] == "C", "EmbarkedCode"] = 1
	data_frame.loc[data_frame["EmbarkedCode"] == "Q", "EmbarkedCode"] = 2

"""Generates a new features in the dataframe, the quantity of people on the passenger's family
based on the values of the columns Parch and SibSp
"""
def generate_new_features(data_frame):
	data_frame["FamilySize"] = data_frame["SibSp"] + data_frame["Parch"]

"""Cleans the data that is missing or not in the right format applying summarization
"""
def clean_data(data_frame):
	fill_age_title_based(data_frame)
	convert_sex_column_int(data_frame)
	convert_embarked_column_int(data_frame)
	data_frame["Fare"] = data_frame["Fare"].fillna(data_frame["Fare"].median())
	generate_new_features(data_frame)

"""Creates .csv file containing the test predictions
"""
def create_submission(train, test, predictors):
	algorithms = [
		[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
		[LogisticRegression(random_state=1), predictors]
	]
	
	full_predictions = []
	for alg, predictors in algorithms:
		#fits the algorithm using the full training data
		alg.fit(train[predictors], train["Survived"])
		#predicts using the test dataset, all the columns are converted to floats to avoid an error
		predictions = alg.predict_proba(test[predictors].astype(float))[:,1]
		full_predictions.append(predictions)
	
	#because of the gradient boosting classifier generating better predictions, it is weighted higher
	predictions = (full_predictions[0] * 3 + full_predictions[1])/4
	predictions[predictions <= .5] = 0
	predictions[predictions > .5] = 1
	predictions = predictions.astype(int)
	
	submission = pd.DataFrame({
		"PassengerId": test["PassengerId"],
		"Survived": predictions
	})
	submission.to_csv("kaggle2.csv", index=False)

titanic = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")
predictors = ["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedCode", "TitleCode", "FamilySize"]

clean_data(titanic)
clean_data(titanic_test)
create_submission(titanic, titanic_test, predictors)





