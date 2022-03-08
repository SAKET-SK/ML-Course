# Machine Learning - Split to Achieve Gain


# Calculate Information Gain.

# Task
# Given a dataset and a split of the dataset, calculate the information gain using the gini impurity.

# The first line of the input is a list of the target values in the initial dataset. The second line is the target values of the left split and the third line is the target values of the right split.

# Round your result to 5 decimal places. You can use round(x, 5).

# Input Format
# Three lines of 1's and 0's separated by spaces

# Output Format
# Float (rounded to 5 decimal places)

# Sample Input
# 1 0 1 0 1 0
# 1 1 1
# 0 0 0

# Sample Output
# 0.5
# Explanation
# The initial set has 3 positive cases and 3 negative cases. Thus the gini impurity is 2*0.5*0.5=0.5.
# The left set has 3 positive cases and 0 negative cases. Thus the gini impurity is 2*1*0=0.
# The right set has 0 positive cases and 3 negative cases. Thus the gini impurity is 2*0*1=0.
# The information gain is 0.5-0-0=0.5

S = [int(x) for x in input().split()]
A = [int(x) for x in input().split()]
B = [int(x) for x in input().split()]

SL = len(S)
BL=len(B)
AL = len(A)


one = S.count(1)
zero = S.count(0)
giniP = one/(one+zero)
giniInit = 2*giniP*(1-giniP)


one = A.count(1)
zero = A.count(0)
giniP = one/(one+zero)
giniLeft = 2*giniP*(1-giniP)


one = B.count(1)
zero = B.count(0)
giniP = one/(one+zero)
giniRight= 2*giniP*(1-giniP)

IG= giniInit -(giniLeft*(AL/SL))-(giniRight*(BL/SL))
print (round(IG,5))
