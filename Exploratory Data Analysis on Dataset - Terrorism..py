#!/usr/bin/env python
# coding: utf-8

# # NAME -: KRUTIKA SUKDEV JAVRE 

# # project name :-iris flower classification ML projects

# # Hey in this project i did the analysis of iris flower dataset using python. how the lengths and widths of each species varies in this project.by using the ML algorithme prediction of length and width is also shown......

# # loading the libraries

# In[207]:



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # importing the "iris flower" dataset

# In[208]:


from sklearn.datasets import load_iris


# In[209]:


iris = load_iris()


# In[210]:


iris


# In[211]:


df.head()


# In[212]:


df.tail()


# In[213]:


iris.data.shape


# In[214]:


iris.target.shape


# In[215]:


iris.feature_names


# In[216]:


iris.target_names


# In[217]:


print(iris.DESCR)


# # convert data into dataframe

# In[221]:


df=pd.DataFrame(iris.data, columns = iris.feature_names)


# In[222]:


df.head()


# In[223]:


df.tail()


# In[224]:


df['target']=iris.target


# In[225]:


df.head()


# In[226]:


df['target']=iris.target


# In[227]:


df.head()


# In[228]:


df.dtypes


# In[229]:


df.describe()


# In[230]:


print(df.groupby('target').size())


# # data visualization

# In[231]:


df.plot(kind='box',subplots=True,  layout=(3,2),figsize=(8,12));


# In[232]:


df.hist(figsize=(12,12))
plt.show()


# In[233]:


df.corr


# In[234]:


sns.heatmap(df.corr(),annot=True, cmap = 'Wistia')


# In[235]:


sns.pairplot(df)


# In[236]:


sns.catplot(x='sepal length (cm)', y = 'sepal width (cm)', palette = 'coolwarm',hue='target',data=df)


# In[ ]:




