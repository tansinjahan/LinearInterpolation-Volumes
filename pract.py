import numpy as np

out = np.array([[[1,0,0], [0,2,0], [1,1,0]],
                [[1,0,0], [0,2,0], [1,1,0]],
                [[1,0,0], [0,2,0], [1,1,0]]])
print("This is the shape of out", out.shape)
z,x,y = out.nonzero()
print("This is z", type(z),z.shape,z)
print("This is x", type(x),x.shape,x)
print("This is y", type(y),y.shape,y)