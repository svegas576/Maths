#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(1,6,400)
plt.plot(x,np.cos(x), marker='o')
plt.plot(x,np.cos(2*x), marker='o')


# In[ ]:




