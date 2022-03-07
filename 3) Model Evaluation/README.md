Accuracy


In the previous module, we calculated how well our model performed using accuracy. Accuracy is the percent of predictions that are correct.

If you have 100 datapoints and predict 70 of them correctly and 30 incorrectly, the accuracy is 70%.

Accuracy is a very straightforward and easy to understand metric, however it’s not always the best one. For example, let’s say I have a model to predict whether a credit card charge is fraudulent. Of 10000 credit card chards, we have 9900 legitimate charges and 100 fraudulent charges. I could build a model that just predicts that every single charge is legitimate and it would get 9900/10000 (99%) of the predictions correct!

Accuracy is a good measure if our classes are evenly split, but is very misleading if we have imbalanced classes.
Always use caution with accuracy. You need to know the distribution of the classes to know how to interpret the value.
