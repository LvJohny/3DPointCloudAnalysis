import numpy as np
import  random
a = np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]])

print(a.mean(0))
print(a.shape[1])
b = np.array([[1,2,3]])
print(b.shape[0])

c = a[0,:]
print(a.shape == (3,))
print(c.shape[0])
print(c.mean(0))
approximate = 'random'

if (approximate == 'centroid'):
    print("centroid")
elif (approximate == 'random'):
    print("random")
else:
    print("error")

print(random.randint(0,a.shape[0]))