# Enter your code here. Read input from STDIN. Print output to STDOUT

N = int(input().strip())
X = [int(x) for x in input().strip().split()]

mean = sum(X) / N
variance = sum([((x - mean) ** 2) for x in X]) / N
stddev = variance ** 0.5

print("{0:0.1f}".format(stddev))
