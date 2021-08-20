import numpy as np
import matplotlib.pyplot as plt
import numpy.random

xs = np.arrange(0.0,10.0,0.5,dtype=np.float64)
ys = np.sin(xs)
zs = np.cos(xs)
ys2 = np.sin(xs)+1.0
zs2 = np.cos(xs)+1.0

print(ys)

plt.plot(ys,zs,xs)
plt.title('My plot!')
plt.show()