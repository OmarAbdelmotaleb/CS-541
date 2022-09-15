#####################################################################
# Omar Abdelmotaleb
# I pledge my honor that I have abided by the Stevens Honor System.
# Part 1: twin_prime.py
#####################################################################

import math

# Takes an integer x as input and computes the maximum twin prime
# between 1 and x.

# Given instructions to use command line, this is limited by integer values.
x = int(input())

# Assuming input is exclusive in range
while x < 6:
    print("Cannot give twin primes for given input. Please try again.")
    x = int(input())

def is_prime(n):
    if n < 2:
        return False

    checks = list(range(2,math.floor(math.sqrt(n))+1))
    for i in checks:
        if n % i == 0:
            return False

    return True

x = x-1
while x > 2:
    if is_prime(x) and is_prime(x-2):
        print("Twin Primes (" + str(x-2) + ", " + str(x) + ")")
        break
    x -= 1   