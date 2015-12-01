"""Fourth attempt of solving Kaggle's Titanic machine learning problem using an ensemble of Logistic Regression
and Gradient Boosting Classifier
Author : Leticia Wanderley
Date : 25 November 2015
"""

import pandas as pd
import re #import regular expression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import operator
import math

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
	data_frame["EmbarkedS"] = np.zeros(len(data_frame))
	data_frame["EmbarkedC"] = np.zeros(len(data_frame))
	data_frame["EmbarkedQ"] = np.zeros(len(data_frame))
	data_frame["Embarked"] = data_frame["Embarked"].fillna("S")
	data_frame.loc[data_frame["Embarked"] == "S", "EmbarkedS"] = 1
	data_frame.loc[data_frame["Embarked"] == "C", "EmbarkedC"] = 1
	data_frame.loc[data_frame["Embarked"] == "Q", "EmbarkedQ"] = 1
	data_frame["EmbarkedCode"] = data_frame.Embarked
	data_frame.loc[data_frame["Embarked"] == "S", "EmbarkedCode"] = 0
	data_frame.loc[data_frame["Embarked"] == "C", "EmbarkedCode"] = 1
	data_frame.loc[data_frame["Embarked"] == "Q", "EmbarkedCode"] = 2

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

"""Generates a new features in the dataframe, the quantity of people on the passenger's family
based on the values of the columns Parch and SibSp
"""
def generate_family_features(data_frame):
	data_frame["FamilySize"] = data_frame["SibSp"] + data_frame["Parch"]
	
	# Get the family ids with the apply method
	family_ids = data_frame.apply(get_family_id, axis=1)

	# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
	family_ids[data_frame["FamilySize"] < 3] = -1
	data_frame["FamilyId"] = family_ids

#OVERFITTING
def clean_fill_cabin(data_frame):
	data_frame["CabinCode"] = data_frame.Cabin
	data_frame.loc[(data_frame.Cabin.isnull()), 'CabinCode'] = 0
	data_frame.loc[data_frame["Cabin"].str[0] == "A", "CabinCode"] = 1
	data_frame.loc[data_frame["Cabin"].str[0] == "B", "CabinCode"] = 2
	data_frame.loc[data_frame["Cabin"].str[0] == "C", "CabinCode"] = 3
	data_frame.loc[data_frame["Cabin"].str[0] == "D", "CabinCode"] = 4
	data_frame.loc[data_frame["Cabin"].str[0] == "E", "CabinCode"] = 5
	data_frame.loc[data_frame["Cabin"].str[0] == "F", "CabinCode"] = 6
	data_frame.loc[data_frame["Cabin"].str[0] == "G", "CabinCode"] = 7
	data_frame.loc[data_frame["Cabin"].str[0] == "T", "CabinCode"] = 8


"""Cleans the data that is missing or not in the right format applying summarization
"""
def clean_data(data_frame):
	fill_age_title_based(data_frame)
	generate_family_features(data_frame)
	convert_sex_column_int(data_frame)
	convert_embarked_column_int(data_frame)
	clean_fill_cabin(data_frame)
	data_frame["Fare"] = data_frame["Fare"].fillna(data_frame["Fare"].median())

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
"""Creates .csv file containing the test predictions
"""
def create_submission(train, test, predictors):
	algorithms = [
		#[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedCode", "FamilySize", "TitleCode", "FamilyId"]],
    	#[LogisticRegression(random_state=1), ["Pclass", "Gender", "Fare", "FamilySize", "TitleCode", "AgeFill", "EmbarkedCode"]],
    	[RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=14, min_samples_leaf=4), ["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedC", "EmbarkedQ", "EmbarkedS", "TitleCode", "FamilySize", "FamilyId"]]
		#[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
		#[LogisticRegression(random_state=1), predictors],
		#[RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4), ["Pclass", "Gender", "Fare", "TitleCode", "CabinCode"]]
		#[RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2), predictors_1]
	]
	
	full_predictions = []
	for alg, predictors in algorithms:
		#fits the algorithm using the full training data
		alg.fit(train[predictors], train["Survived"])
		#predicts using the test dataset, all the columns are converted to floats to avoid an error
		predictions = alg.predict_proba(test[predictors].astype(float))[:,1]
		full_predictions.append(predictions)
	
	#because of the gradient boosting classifier generating better predictions, it is weighted higher
	#predictions = (full_predictions[1] * 3 + full_predictions[0])/4
	predictions = full_predictions[0]
	predictions[predictions <= .5] = 0
	predictions[predictions > .5] = 1
	predictions = predictions.astype(int)
	
	submission = pd.DataFrame({
		"PassengerId": test["PassengerId"],
		"Survived": predictions
	})
	submission.to_csv("kaggle10.csv", index=False)

titanic = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")
predictors_1 = ["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedC", "EmbarkedQ", "EmbarkedS", "TitleCode", "FamilySize", "FamilyId", "CabinCode", "SibSp", "Parch"]
predictors_2 = ["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedCode", "TitleCode", "FamilySize", "FamilyId", "CabinCode"]
predictors_3 = ["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedCode", "TitleCode", "FamilySize", "CabinCode"]

family_id_mapping = {}
clean_data(titanic)
clean_data(titanic_test)

create_submission(titanic, titanic_test, predictors_2)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

def find_best_features(data_frame):
	predictors = ["Pclass", "Gender", "AgeFill", "SibSp", "Parch", "Fare", "EmbarkedCode", "FamilySize", "TitleCode", "FamilyId",
	"CabinCode", "EmbarkedS", "EmbarkedQ", "EmbarkedC"]

	# Perform feature selection
	selector = SelectKBest(f_classif, k=5)
	selector.fit(data_frame[predictors], data_frame["Survived"])

	# Get the raw p-values for each feature, and transform from p-values into scores
	scores = -np.log10(selector.pvalues_)

	# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
	plt.bar(range(len(predictors)), scores)
	plt.xticks(range(len(predictors)), predictors, rotation='vertical')
	plt.show()

	predictors = ["Pclass", "Gender", "Fare", "TitleCode", "CabinCode"]

	alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)

	scores = cross_validation.cross_val_score(alg, titanic[predictors], y=titanic["Survived"], cv=3)
	print(scores.mean())

#find_best_features(titanic)

def classificators(data_frame, predictors_1, predictors_2):
	# gnb = GaussianNB()
	# scores_gnb = cross_validation.cross_val_score(gnb, data_frame[predictors_2], data_frame["Survived"], cv=3)
	# print(scores_gnb.mean())

	# clf_isotonic = CalibratedClassifierCV(gnb, cv=2, method='isotonic')
	# scores_clf_isotonic = cross_validation.cross_val_score(clf_isotonic, data_frame[predictors_2], data_frame["Survived"], cv=3)
	# print(scores_clf_isotonic.mean())

	# clf_sigmoid = CalibratedClassifierCV(gnb, cv=2, method='sigmoid')
	# scores_clf_sigmoid = cross_validation.cross_val_score(clf_sigmoid, data_frame[predictors_1], data_frame["Survived"], cv=3)
	# print(scores_clf_sigmoid.mean())
	print("Gradient Boosting Classifier")
	gbc = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
	scores_gbc = cross_validation.cross_val_score(gbc, data_frame[["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedCode", "TitleCode", "FamilySize", "FamilyId", "CabinCode"]], data_frame["Survived"], cv=10)
	print(scores_gbc.mean())
	gbc = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
	scores_gbc = cross_validation.cross_val_score(gbc, data_frame[["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedCode", "FamilySize", "TitleCode", "FamilyId"]], data_frame["Survived"], cv=10)
	print(scores_gbc.mean())
	gbc = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
	scores_gbc = cross_validation.cross_val_score(gbc, data_frame[["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedCode", "FamilySize", "TitleCode", "FamilyId", "CabinCode"]], data_frame["Survived"], cv=10)
	print(scores_gbc.mean())
	
	print("Random Forest Classifier")
	rfc = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
	scores_rfc = cross_validation.cross_val_score(rfc, data_frame[["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedCode", "TitleCode", "FamilySize", "FamilyId", "CabinCode"]], y=data_frame["Survived"], cv=10)
	print(scores_rfc.mean())
	rfc = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
	scores_rfc = cross_validation.cross_val_score(rfc, data_frame[["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedCode", "FamilySize", "TitleCode", "FamilyId"]], y=data_frame["Survived"], cv=10)
	print(scores_rfc.mean())
	rfc = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
	scores_rfc = cross_validation.cross_val_score(rfc, data_frame[["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedCode", "FamilySize", "TitleCode", "FamilyId", "CabinCode"]], y=data_frame["Survived"], cv=10)
	print(scores_rfc.mean())
	print("\n")

	rfc = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
	scores_rfc = cross_validation.cross_val_score(rfc, data_frame[["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedC", "EmbarkedQ", "EmbarkedS", "TitleCode", "FamilySize", "FamilyId", "CabinCode"]], y=data_frame["Survived"], cv=10)
	print(scores_rfc.mean())
	rfc = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
	scores_rfc = cross_validation.cross_val_score(rfc, data_frame[["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedC", "EmbarkedQ", "EmbarkedS", "FamilySize", "TitleCode", "FamilyId"]], y=data_frame["Survived"], cv=10)
	print(scores_rfc.mean())
	rfc = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
	scores_rfc = cross_validation.cross_val_score(rfc, data_frame[["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedC", "EmbarkedQ", "EmbarkedS", "FamilySize", "TitleCode", "FamilyId", "CabinCode"]], y=data_frame["Survived"], cv=10)
	print(scores_rfc.mean())

	print("Logistic Regression")
	lrc = LogisticRegression(random_state=1)
	scores_lrc = cross_validation.cross_val_score(lrc, data_frame[["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedCode", "TitleCode", "FamilySize", "FamilyId", "CabinCode"]], data_frame["Survived"], cv=10)
	print(scores_lrc.mean())
	lrc = LogisticRegression(random_state=1)
	scores_lrc = cross_validation.cross_val_score(lrc, data_frame[["Pclass", "Gender", "Fare", "FamilySize", "TitleCode", "AgeFill", "EmbarkedCode"]], data_frame["Survived"], cv=10)
	print(scores_lrc.mean())
	lrc = LogisticRegression(random_state=1)
	scores_lrc = cross_validation.cross_val_score(lrc, data_frame[["Pclass", "Gender", "Fare", "FamilySize", "TitleCode", "AgeFill", "EmbarkedCode", "CabinCode"]], data_frame["Survived"], cv=10)
	print(scores_lrc.mean())

	# svmc = svm.SVC(kernel='linear')
	# scores_svmc = cross_validation.cross_val_score(svmc, data_frame[predictors_1], data_frame["Survived"], cv=3)
	# print(scores_svmc.mean())

def loop_random_forests(data_frame, predictors):
	for i in range(2,15):
		rfc = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=i, min_samples_leaf=4)
		scores_rfc = cross_validation.cross_val_score(rfc, data_frame[predictors], y=data_frame["Survived"], cv=10)
		print(str(i), scores_rfc.mean())
p = ["Pclass", "Gender", "AgeFill", "Fare", "EmbarkedCode", "TitleCode", "FamilySize", "FamilyId"]
#loop_random_forests(titanic, p)

#classificators(titanic, predictors_1, predictors_2)



