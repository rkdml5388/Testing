import numpy as np
import matplotlib.pyplot as plt
avg = 0.0
var = 1.0
sigma=np.sqrt(var)
s=np.random.normal(avg,var,10000)
plt.hist(s,40,normed=True)
x=np.arange(-10,10,0.1)
G=np.sqrt(1/(2*np.pi*var**2))*np.exp(-(x-avg)**2/(2*var**2))
plt.plot(x,G)
plt.show()