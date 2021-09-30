import numpy as np
import matplotlib.pyplot as mp

# 三维数组，nums[0][0] = [10,10,10]
x = np.zeros((2, 2, 3), dtype=np.int32)
x[0, 0] = 10
print(x)

mp.figure(1)
mp.figure(2)
mp.show()
