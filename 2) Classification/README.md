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

Classification Graphically

We will eventually want to use all the features, but for simplicity let’s start with only two of the features (Fare and Age). Using two features enables us to visualize the data in a graph.

On the x-axis we have the passenger’s fare and on the y-axis their age. The yellow dots are passengers that survived and the purple dots are passengers that didn’t survive.
You can see that there are more yellow dots at the bottom of the graph than the top. This is because children were more likely to survive than adults, which fits our intuition. Similarly there are more yellow dots on the right of the graph, meaning people that paid more were more likely to survive.

The task of a linear model is to find the line that best separates the two classes, so that the yellow points are on one side and the purple points are on the other.

Here is an example of a good line. The line is used to make predictions about new passengers. If a passenger’s datapoint lies on the right side of the line, we’d predict that they survive. If on the left side, we’d predict that they didn’t survive.
The challenge of building the model will be determining what the best possible line is.
Equation for the Line


A line is defined by an equation in the following form:
0 = ax + by + c
The values a, b, and c are the coefficients. Any three values will define a unique line.

Let’s look at a specific example of a line where the coefficients are a=1, b=-1 and c=-30.
0 = (1)x + (-1)y + (-30)
The three coefficients are: 1, -1, -30

Recall that we’ve been plotting our data with x axis the Fare and y axis the Age of the passenger.

To draw a line from an equation, we need two points that are on the line.

We can see, for example, that the point (30, 0) lies right on the line (Fare 30, Age 0). If we plug it into the equation, it works out.
30 - 0 - 30 = 0
We can also see that the point (50, 20) is on the line (Fare 50, Age 20).
50 - 20 - 30 = 0
Here’s what our line looks like on the graph.contentImage
The coefficients of the line are what control where the line is.

Making a Prediction Based on the Line


Let’s again look at the same line.
0 = (1)x + (-1)y - 30 
contentImageIf we take a passenger’s data, we can use this equation to determine which side of the line they fall on. For example, let’s say we have a passenger whose Fare is 100 and Age is 20.

Let’s plug in these values to our equation:
(1)100 + (-1)20 - 30 = 100 - 20 - 30 = 50 
Since this value is positive, the point is on the right side of the line and we’d predict that the passenger survived.

Now let’s say a passenger had a Fare of 10 and their Age is 50. Let’s plug these values into the equation.
(1)10 + (-1)50 - 30 = -70 
Since this value is negative, the point is on the left side of the line and we’d predict that the passenger didn’t survive.

We can see these two points on the plot below.contentImage
Which side of the line a point lies on determines whether we think that passenger will survive or not.

What Makes a Good Line?


Let’s look at two different lines.

First we have the line we’ve been working with so far. Let’s call this Line 1.
0 = (1)x + (-1)y - 30 
contentImageNext we have another equation for a line. Let’s call this Line 2.
0 = (4)x + (5)y - 400 
contentImageIf we look at the two lines, we see that Line 1 has more yellow on the right and more purple points on the left. Line 2 doesn’t have many points to the right of it; most of both the purple and yellow dots are on the left. This makes Line 1 the preferred line as it does a better job of splitting the yellow and purple points.

We need to mathematically define this idea so that we can algorithmically find the best line.
Logistic Regression is a way of mathematically finding the best line.

Probability of Surviving


In order to determine the best possible line to split our data, we need to have a way of scoring the line. First, let’s look at a single datapoint.

Ideally, if the datapoint is a passenger who survived, it would be on the right side of the line and far from the line. If it’s a datapoint for a passenger who didn’t survive, it would be far from the line to the left. The further it is from the line, the more confident we are that it’s on the correct side of the line.

For each datapoint, we’ll have a score that’s a value between 0 and 1. We can think of it as the probability that the passenger survives. If the value is close to 0 that point would be far to the left of the line and that means we’re confident the passenger didn’t survive. If the value is close to 1 that point would be far to the right of the line and that means we’re confident the passenger did survive. A value of 0.5 means the point falls directly on the line and we are uncertain if the passenger survives.

The equation for calculating this score is below, though the intuition for it is far more important that the actual equation.

Recall that the equation for the line is in the form 0 = ax+by+c (x is the Fare, y is the Age, and a, b & c are the coefficients that we control). The number e is the mathematical constant, approximately 2.71828.

Logistic Regression gives not just a prediction (survived or not), but a probability (80% chance this person survived).

Likelihood


To calculate how good our line is, we need to score whether our predictions are correct. Ideally if we predict with a high probability that a passenger survives (meaning the datapoint is far to the right of the line), then that passenger actually survives.

So we’ll get rewarded when we predict something correctly and penalized if we predict something incorrectly.
Here’s the likelihood equation. Though again, the intuition is more important than the equation.

Here p is the predicted probability of surviving from the previous part.

The likelihood will be a value between 0 and 1. The higher the value, the better our line is.

Let’s look at a couple possibilities:
• If the predicted probability p is 0.25 and the passenger didn’t survive, we get a score of 0.75 (good).
• If the predicted probability p is 0.25 and the passenger survived, we get a score of 0.25 (bad).

We multiply all the individual scores for each datapoint together to get a score for our line. Thus we can compare different lines to determine the best one.

Let’s say for ease of computation that we have 4 datapoints.
We get the total score by multiplying the four scores together:
0.25 * 0.75 * 0.6 * 0.8 = 0.09
The value is always going to be really small since it is the likelihood that our model predicts everything perfectly. A perfect model would have a predicted probability of 1 for all positive cases and 0 for all negative cases.
The likelihood is how we score and compare possible choices of a best fit line.


What is Scikit-learn?


Now that we’ve built up the foundation of how Logistic Regression works, let’s dive into some code to build a model.

For this we’re going to introduce a new Python module called scikit-learn. Scikit-learn, often shortened to sklearn, is our scientific toolkit.

All of the basic machine learning algorithms are implemented in sklearn. We’ll see that with just a few lines of code we can build several different powerful models.

Note that scikit-learn is continually being updated. If you have a slightly different version of the module installed on your computer, everything will still work correctly, but you might see slightly different values than in the playground.
Scikit-learn is one of the best documented Python modules out there. You can find lots of code samples at scikit-learn.org
Prep Data with Pandas


Before we can use sklearn to build a model, we need to prep the data with Pandas. Let’s go back to our full dataset and review the Pandas commands.

Here’s a Pandas DataFrame of data with all the columns:contentImageFirst, we need to make all our columns numerical. Recall how to create the boolean column for Sex.
```
df['male'] = df['Sex'] == 'male'
```

Now, let’s take all the features and create a numpy array called X. We first select all the columns we are interested in and then use the values method to convert it to a numpy array.
```
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
```

Now let’s take the target (the Survived column) and store it in a variable y.
```
y = df['Survived'].values
```

It’s standard practice to call our 2d array of features X and our 1d array of target values y.

Build a Logistic Regression Model with Sklearn
We start by importing the Logistic Regression model:
```
from sklearn.linear_model import LogisticRegression 
```
All sklearn models are built as Python classes. We first instantiate the class.
```
model = LogisticRegression()
```
Now we can use our data that we previously prepared to train the model. The fit method is used for building the model. It takes two arguments: X (the features as a 2d numpy array) and y (the target as a 1d numpy array).

For simplicity, let’s first assume that we’re building a Logistic Regression model using just the Fare and Age columns. First we define X to be the feature matrix and y the target array.
```
X = df[['Fare', 'Age']].values
y = df['Survived'].values
```
Now we use the fit method to build the model.
```
model.fit(X, y)
```
Fitting the model means using the data to choose a line of best fit. We can see the coefficients with the coef_ and intercept_ attributes.
```
print(model.coef_, model.intercept_)
```
These values mean that the equation is as follows:
0 = 0.0161594x + -0.01549065y + -0.51037152
Here’s the line drawn on the graph. You can see it does a decent (but not great) job of splitting the yellow and purple points. We’ve handicapped ourselves a bit by only using 2 of our features, so in the next parts we’ll use all of the features.contentImage
It can be hard to remember the import statements for the different sklearn models. If you can’t remember, just look at the scikit-learn documentation.
