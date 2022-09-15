#####################################################################
# Omar Abdelmotaleb
# I pledge my honor that I have abided by the Stevens Honor System.
# Part 3: summary.py
#####################################################################

import pandas as pd
from math import sqrt

# Takes in a string state to view statistical summary of COVID cases
state = str(input())

# Read the CSV
df = pd.read_csv("us-states.csv")
# Filter by the state input
cases = df[df.state == state]
# Filter the column of cases out to use
state_cases = cases["cases"].tolist()

mean = sum(state_cases)/len(state_cases)

# Computes the variance and takes the square root to get standard deviation.
# x is the data, u is the population mean, n is the population size
def sd(x, u, n):
    top = 0
    for i in x:
        top += ((i - u) ** 2)
    return sqrt(top / n)

stdev = sd(state_cases, mean, len(state_cases))


print("Minimum: " + str( min(state_cases) ))
print("Maximum: " + str( max(state_cases) ))
print("Mean: " + str( mean ))
print("Standard Deviation: " + str( stdev ))