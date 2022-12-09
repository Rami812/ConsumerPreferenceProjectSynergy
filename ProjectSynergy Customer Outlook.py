#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[4]:


df=pd.read_csv("GroceryPreferenceResponses.csv")#The file name does not need to actually end with .csv to run the file


# In[5]:


model=linear_model.LinearRegression()
model.fit(df[["Ideal Look","Nutricious Value"]],df[["PurchaseLikelihood"]])


# In[6]:


model.coef_


# In[ ]:


##Look into dispalying trend with python

