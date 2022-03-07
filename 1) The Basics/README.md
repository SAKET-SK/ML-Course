Machine Learning is a way of taking data and turning it into insights. We use computer power to analyze examples from the past to build a model that can predict the result for new examples.

We encounter machine learning models every day. For example, when Netflix recommends a show to you, they used a model based on what you and other users have watched to predict what you would like. When Amazon chooses a price for an item, they use a model based on how similar items have sold in the past. When your credit card company calls you because of suspicious activity, they use a model based on your past activity to recognize anomalous behavior.

Machine Learning can be used to create a chatbot, detect spam or image recognition.
One of the most common languages used by Machine Learning professionals is Python.

It's both very approachable and very powerful, so it's what we'll be using in this course. We assume a working knowledge of Python.

In this course we’ll use several Python packages that are helpful in solving Machine Learning problems. We’ll be using pandas, numpy, matplotlib and scikit-learn. Pandas is used for reading data and data manipulation, numpy is used for computations of numerical data, matplotlib is used for graphing data, and scikit-learn is used for machine learning models. Each of these packages is quite extensive, but we will review the functions we will be using. We will also review some basic statistics as it is the foundation of machine learning.
In Machine Learning, we talk about supervised and unsupervised learning. Supervised learning is when we have a known target based on past data (for example, predicting what price a house will sell for) and unsupervised learning is when there isn't a known past answer (for example, determining the topics discussed in restaurant reviews).

In this course we'll be focusing on supervised learning. Within supervised learning, there are classification and regression problems. Regression is predicting a numerical value (for example, predicting what price a house will sell for) and classification is predicting what class something belongs to (for example, predicting if a borrower will default on their loan).

We'll be focusing on classification problems. These are problems where we’re predicting which class something belongs to.

Our examples will include:
• Predicting who would survive the Titanic crash
• Determining a handwritten digit from an image
• Using biopsy data to classify if a lump is cancerous

We'll be using a number of popular techniques to tackle these problems. We'll get into each of them in more detail in upcoming modules:
• Logistic Regression
• Decision Trees
• Random Forests
• Neural Networks

At the end of this course, you'll be able to take a classification dataset and use Python to build several different models to determine the best model for the given problem.
Machine Learning can be used to solve a broad range of problems. This course will focus on supervised learning and classification.

Averages


When dealing with data, we often need to calculate some simple statistics.

Let's say we have a list of the ages of people in a class.
We have them in ascending order since it will be easier to do the calculations.
15, 16, 18, 19, 22, 24, 29, 30, 34 
The mean is the most commonly known average.

Add up all the values and divide by the number of values:
(15 + 16 + 18 + 19 + 22 + 24 + 29 + 30 + 34) / 9 =  207/9 = 23
The median is the value in the middle. In this case, since there are 9 values, the middle value is the 5th, which is 22.
In statistics, both the mean and the median are called averages. The layman’s average is the mean.

Percentiles


The median can also be thought of as the 50th percentile. This means that 50% of the data is less than the median and 50% of the data is greater than the median. This tells us where the middle of the data is, but we often want more of an understanding of the distribution of the data. We’ll often look at the 25th percentile and the 75th percentile.

The 25th percentile is the value that’s one quarter of the way through the data. This is the value where 25% of the data is less than it (and 75% of the data is greater than it).

Similarly, the 75th percentile is three quarters of the way through the data. This is the value where 75% of the data is less than it (and 25% of the data is greater than it).

If we look at our ages again:
15, 16, 18, 19, 22, 24, 29, 30, 34 
We have 9 values, so 25% of the data would be approximately 2 datapoints. So the 3rd datapoint is greater than 25% of the data. Thus, the 25th percentile is 18 (the 3rd datapoint).
Similarly, 75% of the data is approximately 6 datapoints. So the 7th datapoint is greater than 75% of the data. Thus, the 75th percentile is 29 (the 7th datapoint).

The full range of our data is between 15 and 34. The 25th and 75th percentiles tell us that half our data is between 18 and 29. This helps us gain understanding of how the data is distributed.
If there is an even number of datapoints, to find the median (or 50th percentile), you take the mean of the two values in the middle.

Standard Deviation & Variance


We can get a deeper understanding of the distribution of our data with the standard deviation and variance. The standard deviation and variance are measures of how dispersed or spread out the data is.

We measure how far each datapoint is from the mean.

Let's look at our group of ages again:
15, 16, 18, 19, 22, 24, 29, 30, 34
Recall that the mean is 23.

Let's calculate how far each value is from the mean. 15 is 8 away from the mean (since 23-15=8).

Here's a list of all these distances:
8, 7, 5, 4, 1, 1, 6, 7, 11
We square these values and add them together.contentImageWe divide this value by the total number of values and that gives us the variance.
362 / 9 = 40.22 
To get the standard deviation, we just take the square root of this number and get: 6.34

If our data is normally distributed like the graph below, 68% of the population is within one standard deviation of the mean. In the graph, we’ve highlighted the area within one standard deviation of the mean. You can see that the shaded area is about two thirds (more precisely 68%) of the total area under the curve. If we assume that our data is normally distributed, we can say that 68% of the data is within 1 standard deviation of the mean.contentImageIn our age example, while the ages are likely not exactly normally distributed, we assume that we are and say that approximately 68% of the population has an age within one standard deviation of the mean. Since the mean is 23 and the standard deviation is 6.34, we can say that approximately 68% of the ages in our population are between 16.66 (23 minus 6.34) and 29.34 (23 plus 6.34).
Even though data is never a perfect normal distribution, we can still use the standard deviation to gain insight about how the data is distributed.

Statistics with Python


We can calculate all of these operations with Python. We will use the Python package numpy. We will use numpy more later for manipulating arrays, but for now we will just use a few functions for statistical calculations: mean, median, percentile, std, var.

First we import the package. It is standard practice to nickname numpy as np
```
import numpy as np
```
Let’s initialize the variable data to have the list of ages.
```
data = [15, 16, 18, 19, 22, 24, 29, 30, 34]
```
Now we can use the numpy functions. For the mean, median, standard deviation and variance functions, we just pass in the data list. For the percentile function we pass the data list and the percentile (as a number between 0 and 100).

Numpy is a python library that allows fast and easy mathematical operations to be performed on arrays.

This course is in Python, one of the most commonly used languages for machine learning.

One of the reasons it is so popular is that there are numerous helpful python modules for working with data. The first we will be introducing is called Pandas.
Pandas is a Python module that helps us read and manipulate data. What's cool about pandas is that you can take in data and view it as a table that's human readable, but it can also be interpreted numerically so that you can do lots of computations with it.

We call the table of data a DataFrame.
Python will satisfy all of our Machine Learning needs. We’ll use the Pandas module for data manipulation.

Read in Your Data

We need to start by importing Pandas. It's standard practice to nickname it pd so that it's faster to type later on.
```
import pandas as pd 
```
We will be working with a dataset of Titanic passengers. For each passenger, we’ll have some data on them as well as whether or not they survived the crash.

Our data is stored as CSV (comma-separated values) file. The titanic.csv file is below. The first line is the header and then each subsequent line is the data for a single passenger.
Survived, Pclass, Sex, Age, Siblings/Spouses, Parents/Children, Fare
0, 3, male, 22.0, 1, 0, 7.25
1, 1, female, 38.0, 1, 0, 71.2833
1, 3, female, 26.0, 0, 0, 7.925
1, 1, female, 35.0, 1, 0, 53.1
We're going to pull the data into pandas so we can view it as a DataFrame.

The read_csv function takes a file in csv format and converts it to a Pandas DataFrame.
```
df = pd.read_csv('titanic.csv')
```
The object df is now our pandas dataframe with the Titanic dataset. Now we can use the head method to look at the data.
The head method returns the first 5 rows of the DataFrame.
```
print(df.head())
```

Summarize the Data

Usually our data is much too big for us to be able to display it all.
Looking at the first few rows is the first step to understanding our data, but then we want to look at some summary statistics.
In pandas, we can use the describe method. It returns a table of statistics about the columns.

```
print(df.describe())
```

Selecting a Single Column


We often will only want to deal with some of the columns that we have in our dataset. To select a single column, we use the square brackets and the column name.
```
col = df['Fare']
print(col)
```
The result is what we call a Pandas Series. A Pandas Series is a single column from a Pandas DataFrame.
A series is like a DataFrame, but it's just a single column.

Selecting Multiple Columns


We can also select multiple columns from our original DataFrame, creating a smaller DataFrame.
We're going to select just the Age, Sex, and Survived columns from our original DataFrame.

We put these values in a list as follows:
```
['Age', 'Sex', 'Survived']
```
Now we use that list inside of the bracket notation df[...] When printing a large DataFrame that’s too big to display, you can use the head method to print just the first 5 rows.
```
small_df = df[['Age', 'Sex', 'Survived']]
print(small_df.head()) 
```
When selecting a single column from a Pandas DataFrame, we use single square brackets. When selecting multiple columns, we use double square brackets.

Creating a Column


We often want our data in a slightly different format than it originally comes in. For example, our data has the sex of the passenger as a string ("male" or "female"). This is easy for a human to read, but when we do computations on our data later on, we’ll want it as boolean values (Trues and Falses).

We can easily create a new column in our DataFrame that is True if the passenger is male and False if they’re female.

Recall the syntax for selecting the Sex column:
```
df['Sex']
```
Now we want to create a column with this result. To create a new column, we use the same bracket syntax 
```
(df['male'])
```
and then assign this new value to it.
```
df['male'] = df['Sex'] == 'male'
```
What is Numpy?


Numpy is a Python package for manipulating lists and tables of numerical data. We can use it to do a lot of statistical calculations. We call the list or table of data a numpy array.

We often will take the data from our pandas DataFrame and put it in numpy arrays. Pandas DataFrames are great because we have the column names and other text data that makes it human readable. A DataFrame, while easy for a human to read, is not the ideal format for doing calculations. The numpy arrays are generally less human readable, but are in a format that enables the necessary computation.
Numpy is a Python module for doing calculations on tables of data. Pandas was actually built using Numpy as it’s foundation.
Converting from a Pandas Series to a Numpy Array


We often start with our data in a Pandas DataFrame, but then want to convert it to a numpy array. The values attribute does this for us.

Let's convert the Fare column to a numpy array.

First we recall that we can use the single bracket notation to get a pandas Series of the Fare column as follows.
```
df['Fare']
```
Then we use the values attribute to get the values as a numpy array.
```
df['Fare'].values
```
This is what the above array looks like:
```
array([ 7.25 , 71.2833,  7.925, 53.1, 8.05, 8.4583, …
```
The result is a 1-dimensional array. You can tell since there's only one set of brackets and it only expands across the page (not down as well).
Converting from a Pandas DataFrame to a Numpy Array


If we have a pandas DataFrame (instead of a Series as in the last part), we can still use the values attribute, but it returns a 2-dimensional numpy array.

Recall that we can create a smaller pandas DataFrame with the following syntax.
```
df[['Pclass', 'Fare', 'Age']]
```
Again, we apply the values attribute to get a numpy array.
```
df[['Pclass', 'Fare', 'Age']].values 
```
This is what the above array looks like:
```
array([[ 3.    ,  7.25  , 22.    ],
       [ 1.    , 71.2833, 38.    ],
       [ 3.    ,  7.925 , 26.    ],
                    ...           ,
       [ 3.    , 23.45  ,  7.    ],
       [ 1.    , 30.    , 26.    ],
       [ 3.    ,  7.75  , 32.    ]])
 ```
This is a 2-dimensional numpy array. You can tell because there’s two sets of brackets, and it expands both across the page and down.

Numpy Shape Attribute


We use the numpy shape attribute to determine the size of our numpy array. The size tells us how many rows and columns are in our data.

First, let's create a numpy array with the Pclass, Fare, and Age.
```
arr = df[['Pclass', 'Fare', 'Age']].values
```
If we look at the shape, we get the number of rows and the number of columns:
```
print(arr.shape) #(887, 3)
```
This result means we have 887 rows and 3 columns.
