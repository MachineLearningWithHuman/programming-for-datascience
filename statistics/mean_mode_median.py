#problem: mean, median, and mode

import numpy as np
from scipy import stats

N = int(input())
X = list(map(int, input().split()))
print("mean:",np.mean(X))
print("median",np.median(X))
print("mode",int(stats.mode(X)[0]))
