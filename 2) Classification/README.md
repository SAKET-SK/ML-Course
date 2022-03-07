Where does Classification Fit in the World of Machine Learning?


Machine Learning on a high level is made up of supervised and unsupervised learning.

Supervised Learning means that we will have labeled historical data that we will use to inform our model. We call the label or thing we’re trying to predict, the target. So in supervised learning, there is a known target for the historical data, and for unsupervised learning there is no known target.

Within supervised learning, there is Classification and Regression. Classification problems are where the target is a categorical value (often True or False, but can be multiple categories). Regression problems are where the target is a numerical value.

For example, predicting housing prices is a regression problem. It’s supervised, since we have historical data of the sales of houses in the past. It’s regression, because the housing price is a numerical value. Predicting if someone will default on their loan is a classification problem. Again, it’s supervised, since we have the historical data of whether past loanees defaulted, and it’s a classification problem because we are trying to predict if the loan is in one of two categories (default or not).
Logistic Regression, while it has regression in its name is an algorithm for solving classification problems, not regression problems.
Classification Terminology


Let’s look back at our Titanic dataset. Here again is the Pandas DataFrame of the data:contentImageThe Survived column is what we’re trying to predict. We call this the target. You can see that it’s a list of 1’s and 0’s. A 1 means that the passenger survived, and a 0 means the passenger didn’t survive.

The remaining columns are the information about the passenger that we can use to predict the target. We call each of these columns a feature. Features are the data we use to make our prediction.

While we know whether each passenger in the dataset survived, we’d like to be able to make predictions about additional passengers that we weren’t able to collect that data for. We’ll build a machine learning model to help us do this.
Sometimes you’ll hear features called predictors.
