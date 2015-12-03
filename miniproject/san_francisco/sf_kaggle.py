import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np

def model_comparison(train, validation, features):
	#Naive Bayes
	model = BernoulliNB()
	model.fit(training[features], training['crime'])
	predicted = np.array(model.predict_proba(validation[features]))
	print("BernoulliNB")
	print(log_loss(validation['crime'], predicted)) 
 
	#Logistic Regression for comparison
	model = LogisticRegression(C=.01)
	model.fit(training[features], training['crime'])
	predicted = np.array(model.predict_proba(validation[features]))
	print("LogisticRegression")
	print(log_loss(validation['crime'], predicted))

def look_through_data(train, test):
	#int columns
	print(train.describe())
	print(test.describe())

	#string columns
	print(train.select_dtypes(include=['object']))

def get_binarized_columns(data_frame):
	#get binarized weekdays, districts, and hours
	days = pd.get_dummies(data_frame.DayOfWeek)
	district = pd.get_dummies(data_frame.PdDistrict)
	hour = data_frame.Dates.dt.hour
	hour = pd.get_dummies(hour)
	return [days, district, hour] 

#load datasets parsing Dates columns from string to datetime
sf_train = pd.read_csv("train.csv", parse_dates = ['Dates'])
sf_test = pd.read_csv("test.csv", parse_dates = ['Dates'])

#convert crime labels to numbers
le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(sf_train.Category)
 
#build new array with binarized columns
train_data = pd.concat(get_binarized_columns(sf_train), axis=1)
train_data['crime']=crime
 
#repeat for test data
test_data = pd.concat(get_binarized_columns(sf_test), axis=1)
 
training, validation = train_test_split(train_data, train_size=.60)

#dummie features
features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
 'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN'] 

model = BernoulliNB()
model.fit(train_data[features], train_data['crime'])
predicted = model.predict_proba(test_data[features])
 
#store results
result=pd.DataFrame(predicted, columns=le_crime.classes_)
result.to_csv('results.csv', index = True, index_label = 'Id' )

