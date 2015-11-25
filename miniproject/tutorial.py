"""Tutorial functions
"""
def linear_regression_train(data_frame, predictors):
	# Initialize our algorithm class
	alg = LinearRegression()
	# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
	# We set random_state to ensure we get the same splits every time we run this.
	kf = KFold(data_frame.shape[0], n_folds=3, random_state=1)

	predictions = []
	for train, test in kf:
		# The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
		train_predictors = (data_frame[predictors].iloc[train,:])
		# The target we're using to train the algorithm.
		train_target = data_frame["Survived"].iloc[train]
		# Training the algorithm using the predictors and target.
		alg.fit(train_predictors, train_target)
		# We can now make predictions on the test fold
		test_predictions = alg.predict(data_frame[predictors].iloc[test,:])
		predictions.append(test_predictions)

	return predictions

def linear_regression_analysis(data_frame, predictions):
	# The predictions are in three separate numpy arrays.  Concatenate them into one.  
	# We concatenate them on axis 0, as they only have one axis.
	predictions = np.concatenate(predictions, axis=0)

	# Map predictions to outcomes (only possible outcomes are 1 and 0)
	predictions[predictions > .5] = 1
	predictions[predictions <=.5] = 0
	count = 0.0
	for i in range(0, len(predictions), 1):
		if predictions[i] == data_frame["Survived"][i]:
			count = count + 1

	accuracy = count/len(predictions)
	return accuracy


def logistic_regression_train(data_frame, predictors):
	# Initialize our algorithm
	alg = LogisticRegression(random_state=1)
	# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
	scores = cross_validation.cross_val_score(alg, data_frame[predictors], data_frame["Survived"], cv=3)
	# Take the mean of the scores (because we have one for each fold)
	return scores.mean()
