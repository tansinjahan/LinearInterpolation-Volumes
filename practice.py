import numpy as np

a = np.array(5)
b = np.arange(9.0).reshape((3, 3))

print("this is a ", a)
print("This is b", b)

mul = np.dot(b, a)
print("This is mul", mul)