# Machine Learning - What's in a Column?
# Getting a column from a numpy array.

# Task
# Given a csv file and a column name, print the elements in the given column.

# Input Format
# First line: filename of a csv file
# Second line: column name

# Output Format
# Numpy array

# Sample Input
# https://sololearn.com/uploads/files/one.csv
# a

# File one.csv contents:
# a,b
# 1,3
# 2,4

# Sample Output
# [1 2]
# Explanation
# The a is the header for the first column, which has values [1 2].

import pandas as pd
import numpy as np
filename = input()
column_name = input()
a=pd.read_csv(filename)
b=a[column_name]
print(np.array(b))
