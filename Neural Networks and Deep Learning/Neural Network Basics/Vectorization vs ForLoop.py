import numpy as np 
import time 
a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(f"Vectorized version: {1000*(toc-tic)} ms")

c = 0
tic = time.time()
for i in range(1000000):
  c += a[i]*b[i]
toc = time.time()

print(f"For loop version: {1000*(toc-tic)} ms")

#Vectorization examples
v = [1, 2, 3, 4]
u = np.exp(v)
u = np.log(v)
u = np.abs(v)
u = np.maximum(v, 0)

