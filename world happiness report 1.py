#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_validate
from sklearn.model_selection  import cross_val_score as CVS
from sklearn.linear_model import LogisticRegression
get_ipython().run_line_magic('matplotlib', 'inline')
from numpy.random import randn
df=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv")
df


# In[3]:


df.columns


# In[4]:


df.isnull().sum()


# In[5]:


# Data=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv')
# Data.head()
def highlight_max(s):    
    is_max = s == s.max()
    return ['background-color: green' if v else '' for v in is_max]
df.style.apply(highlight_max)


# In[6]:


sns.pairplot(df)


# In[7]:


fig, axes = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(20,10))

sns.barplot(x='Economy (GDP per Capita)',y='Country',data=df.nlargest(20,'Economy (GDP per Capita)'),ax=axes[0,0],palette="Blues_d")

sns.barplot(x='Happiness Score' ,y='Country',data=df.nlargest(20,'Happiness Score'),ax=axes[0,1],palette="YlGn")

sns.barplot(x='Health (Life Expectancy)' ,y='Country',data=df.nlargest(20,'Health (Life Expectancy)'),ax=axes[1,0],palette='OrRd')

sns.barplot(x='Freedom' ,y='Country',data=df.nlargest(20,'Freedom'),ax=axes[1,1],palette='YlOrBr')


# In[8]:


fig, axes = plt.subplots(nrows=1, ncols=2,constrained_layout=True,figsize=(20,10))

sns.barplot(x='Generosity' ,y='Country',data=df.nlargest(30,'Generosity'),ax=axes[0],palette='Spectral')
sns.barplot(x='Trust (Government Corruption)' ,y='Country',data=df.nlargest(30,'Trust (Government Corruption)'),
            ax=axes[1],palette='RdYlGn')


# In[9]:


print('max:',df['Happiness Score'].max())
print('min:',df['Happiness Score'].min())
add=df['Happiness Score'].max()-df['Happiness Score'].min()
grp=round(add/3,3)
print('range difference:',(grp))


# In[10]:


low=df['Happiness Score'].min()+grp
mid=low+grp

print('upper range of Low grp',low)
print('upper range of Mid grp',mid)
print('upper range of High grp','max:',df['Happiness Score'].max())


# In[11]:


df.info()


# In[ ]:





# In[ ]:




