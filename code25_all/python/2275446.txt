import numpy as np
import scipy
from scipy import *
import matplotlib.pyplot as plt

#array
aa = []
for x in range(44):
    aa.append([])
    for z in range(44):
        aa[x].append(3*sin(x/3.0)+2*cos(z/3.0))

b = aa
plt.imshow(b)
plt.show()

time = 0
dt = 0.1
while(time<3):
    b = sin(aa)
    time += dt

