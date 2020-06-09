#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[14]:


data = pd.read_csv('/Users/shruti/Downloads/1.01. Simple linear regression.csv')


# In[15]:


data


# In[16]:


data.describe()


# In[17]:


y = data['GPA']
x1 = data['SAT']


# In[18]:


plt.scatter(x1,y)
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()


# In[19]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[20]:


plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1, yhat, lw=4, c='orange', label='regression line')
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()


# In[ ]:




