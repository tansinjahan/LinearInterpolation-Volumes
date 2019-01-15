import numpy as np
import matplotlib.pyplot as plt

out = np.array([[1,5],[5,5]])
print("This is the shape of out", out.shape)

out_1 = np.array([[1,5],[4,5]])


plt.plot(out, out_1)
plt.show()

