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

