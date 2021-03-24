#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math           #задание1
def l(x,y):
    len_r = math.sqrt(x**x + y**y)
    print(len_r)
x=5
y=20
l(x,y)


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline                    #задание3')
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 2*np.pi, 0.01)          
r = 4                                    

plt.plot(r*np.sin(t), r*np.cos(t), lw=3) 
plt.axis('equal')                        


# In[7]:


import numpy as np
from matplotlib import pyplot as plt
from math import pi

x=1.     
y=0.5    
a=2.    
b=1.5   
t = np.linspace(0, 2*pi, 100)
plt.plot( x+a*np.cos(t) , y+b*np.sin(t) )
plt.show()


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import decimal
import numpy as np
import numpy.ma as ma

xmin = -20
xmax = 20
dx = 0.1

#xlist = np.around(np.arange(xmin, xmax, dx), decimals=4)
xlist = ma.array(np.around(np.arange(xmin, xmax, dx), decimals=4), mask = [0])
ylist = 1 / xlist

plt.plot(xlist, ylist)
plt.show()


# In[31]:


import numpy as np                           #задание5.1
import itertools
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt3d = plt.figure().gca(projection='3d')
xx, zz = np.meshgrid(range(10), range(10))
yy =0.5
for _ in itertools.repeat(None, 2):
    plt3d.plot_surface(xx, yy, zz)
    yy=yy+.1

plt.show()


# In[32]:


from mpl_toolkits.mplot3d import Axes3D      #задание 5.2
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=plt.figaspect(1))  
ax = fig.add_subplot(111, projection='3d')

coefs = (1, 2, 2)  

rx, ry, rz = 1/np.sqrt(coefs)

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = rx * np.outer(np.cos(u), np.sin(v))    #эллипсоид
y = ry * np.outer(np.sin(u), np.sin(v))
z = rz * np.outer(np.ones_like(u), np.cos(v))

ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')

max_radius = max(rx, ry, rz)
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)

Z = np.sqrt(4.*(X**2 + Y**2)/1. + 1)

xcolors = X - min(X.flat)
xcolors = xcolors/max(xcolors.flat)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.hot(xcolors),
    linewidth=1)

plt.show()


# In[ ]:




