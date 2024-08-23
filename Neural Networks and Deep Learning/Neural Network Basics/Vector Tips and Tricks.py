import numpy as np
import math

#dont use
a = np.random.randn(5)

#use
a = np.random.randn(5, 1)

a=np.array([[2,1],[1,3]])

print(a * a)

a.shape=(4,3)

a = np.random.randn(4, 3) #4 rows 3 columns 
b = np.random.randn(4, 1) #4 rows 1 column

c = a.T + b


a = np.random.randn(3, 3) #4 rows 3 columns 
b = np.random.randn(3, 1) #4 rows 1 column

c = a * b

print(c)

math.ex
