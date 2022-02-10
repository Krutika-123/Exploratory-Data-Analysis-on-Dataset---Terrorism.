#!/usr/bin/env python
# coding: utf-8

# ## Project Name :- Exploratory Data Analysis on Dataset - Terrorism (LGM task-2.1)
# #### Hey in this project I did the analysis and try to find out the hot zone of terrorism using python.
# ___

#  **Author :**    KRUTIKA SUKDEV JAVRE
# 
# 
#  ---

# # Exploratory Data Analysis
# Exploratory Data Analysis (EDA) refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.
# ___

# In[2]:


# Importing all the important Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading Dataset
# The dataset is available at https://bit.ly/2TK5Xn5
# 
# The Global Terrorism dataset consists of 181691 records with 135 columns.
# ___

# In[3]:


df=pd.read_csv('Terrorfinal.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[10]:


df.columns


# In[11]:


df.shape


# In[4]:


# Check Missing and Null Values
print(df.isnull().sum())
print("Total number of null values =", df.isnull().sum().sum())


# In[13]:


# Dataset Types
df.dtypes


# In[15]:


# Dataset Statistical Description
df.describe()


# ## Data Pre-processing
# In data pre-processing, renaming of column names is carried out for better accecibility of the columns.
# 
# A new dataset is created using only required columns from the original dataset. The shape of new dataset is 181691 rows and 19 columns.
# ***

# In[5]:


df.rename(columns={'iyear':'Year', 'imonth':'Month', 'iday':'Day',
                   'extended':'Extended', 'country_txt':'Country',
                   'provstate':'state', 'region_txt':'Region',
                   'attacktype1_txt':'AttackType', 'target1':'Target',
                   'nkill':'Killed', 'nwound':'Wounded', 'summary':'Summary',
                   'gname':'Group', 'targtype1_txt':'Target_type',
                   'weaptype1_txt':'Weapon_type', 'motive':'Motive'}, inplace=True)


# In[6]:


new_df = df[['Year','Month','Extended','Day','Country','state','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]


# In[24]:


new_df.shape


# In[25]:


new_df.columns


# ## Correlation Analysis
# Pandas dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. Both NA and null values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored.
# ___

# In[29]:


# Correlation Analysis
corr_mat = new_df.corr()
corr_mat


# In[30]:


plt.figure(figsize=(8,6))
sns.heatmap(corr_mat, annot=True)
plt.title('Correlation Analysis')
plt.savefig('correlation.png')
plt.show()


# ## Covariance
# Pandas dataframe.cov() is used to compute the pairwise covariance among the series of a DataFrame. The returned data frame is the covariance matrix of the columns of the DataFrame.
# 
# Both NA and null values are automatically excluded from the calculation. A threshold can be set for the minimum number of observations for each value created. Comparisons with observations below this threshold will be returned as NaN.

# In[31]:


# Covariance Analysis
cov_mat = new_df.cov()
cov_mat


# In[32]:


fig,axes = plt.subplots(1,1,figsize=(8,6))
sns.heatmap(cov_mat, annot=True)
plt.title('Covariance')
plt.savefig('covariance.png')
plt.show()


# ## Visualizing Data
# **Pie chart :** Pie Chart is a circular statistical plot that can display only one series of data. The area of the chart is the total percentage of the given data. The area of slices of the pie represents the percentage of the parts of the data.
# 
# **Heatmap :** Heat map is used to find out the correlation between different features in the dataset. High positive or negative value shows that the features have high correlation.
# 
# **Histogram :** A histogram shows the frequency on the vertical axis and the horizontal axis is another dimension. Usually it has bins, where every bin has a minimum and maximum value. Each bin also has a frequency between x and infinite.
# 
# **CountPlot :** A count plot can be thought of as a histogram across a categorical, instead of quantitative, variable.
# 
# **Choropleth Map :** A Choropleth Map is a map composed of colored polygons. It is used to represent spatial variations of a quantity. This page documents how to build outline choropleth maps, but you can also build choropleth tile maps using our Mapbox trace types.

# In[33]:


new_df.hist(figsize=(20,10))


# In[7]:


new_df.Year.value_counts()


# In[35]:


# Increase in Attacks in years
Year = new_df.Year.value_counts().to_dict()
rate = ((Year[2017] - Year[1970]) / Year[2017]) * 100
print(Year[1970],'attacks happened in 1970')
print(Year[2017],'attacks happened in 2017')
print('Total number of attacks from 1970 has increased by',np.round(rate,0),'% till 2017')


# ## Top 10 Countries with most attacks

# In[36]:


# Top 10 Countries with most attacks
print('Top 10 Countries with most attacks')
print('-----------------------------------')
print(new_df['Country'].value_counts().head(10))


# In[10]:


new_df['Country'].value_counts()[:10].values


# In[37]:


#Top Countries affected by Terror Attacks
plt.subplots(figsize=(18,10))
sns.barplot(new_df['Country'].value_counts()[:10].index, new_df['Country'].value_counts()[:10].values,palette='rocket')
plt.xlabel('Countries')
plt.ylabel('Count')
plt.xticks(rotation= 90)
plt.title('Top Countries Affected')
plt.savefig('top10_Countries.png')
plt.show()


# ## Top 10 Cities with most attacks

# In[38]:


print('Top 10 Cities with the most attacks')
print('-----------------------------------')
print(new_df['city'].value_counts().head(10))


# In[39]:


#Top Cities affected by Terror Attacks
plt.subplots(figsize=(18,10))
sns.barplot(new_df['city'].value_counts()[:10].index, new_df['city'].value_counts()[:10].values,palette='rocket')
plt.xlabel('Cities')
plt.ylabel('Count')
plt.xticks(rotation= 90)
plt.title('Top Cities Affected')
plt.savefig('top10_Cities.png')
plt.show()


# ## Top 10 Regions with most attacks

# In[40]:


print('Top 10 Regions with the most attacks')
print('-----------------------------------')
print(new_df['Region'].value_counts())


# In[41]:


plt.subplots(figsize=(15,5))
sns.countplot('Region', data=new_df, palette='inferno', order=new_df['Region'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Regions')
plt.title('Number Of Terrorist Activities By Region')
plt.savefig('top_Regions.png')
plt.show()


# ### Terrorist Activies by Region in Each Year

# In[42]:


pd.crosstab(new_df.Year, new_df.Region).plot(kind='area', figsize=(20,10))
plt.ylabel('Number of Attacks')
plt.title('Terrorist Activities by Region in each Year')
plt.savefig('yearly_RegionWiseIncrease.png')
plt.show()


# In[43]:


data_after = new_df[new_df['Year']>=2001]
fig,ax = plt.subplots(figsize=(15,10), nrows=2, ncols=1)
ax[0] = pd.crosstab(new_df.Year, new_df.Region).plot(ax=ax[0])
ax[0].set_title('Change in Regions per Year')
ax[0].legend(loc='center left',bbox_to_anchor = (1,0.5))
ax[0].vlines(x=2001,ymin=0,ymax=7000,colors='red',linestyles='--')

pd.crosstab(data_after.Year, data_after.Region).plot.bar(stacked=True, ax=ax[1])
ax[1].set_title('After Declaration of War on Terror (2001-2017)')
ax[1].legend(loc='center left',bbox_to_anchor = (1,0.5))
plt.savefig('yearly_RegionwiseChange.png')
plt.show()


# In[44]:


fig,axes = plt.subplots(figsize=(16,11),nrows=1,ncols=2)
sns.barplot(x = new_df['Country'].value_counts()[:20].values, y = new_df['Country'].value_counts()[:20].index, 
            ax=axes[0],palette='magma');
axes[0].set_title('Terrorist Attacks per Country')
sns.barplot(x=new_df['Region'].value_counts().values, y=new_df['Region'].value_counts().index,
            ax=axes[1],palette='magma_r')
axes[1].set_title('Terrorist Attacks per Region')
fig.tight_layout()
plt.savefig('terroristAttacks_Countries_Regions.png')
plt.show()


# In[45]:


print('Country with Highest Terrorist Attacks:', new_df['Country'].value_counts().index[0])
print('Regions with Highest Terrorist Attacks:', new_df['Region'].value_counts().index[0])
print('Maximum people killed in an attack are:', new_df['Killed'].max(),'that took place in', new_df.loc[df['Killed'].idxmax()].Country)


# ### Years with most attacks

# In[46]:


print('Year with the most attacks')
print('-----------------------------------')
print(new_df['Year'].value_counts().head(10))


# In[47]:


print("Year with the most attacks:", new_df['Year'].value_counts().idxmax())


# In[48]:


plt.subplots(figsize=(20,10))
sns.countplot('Year', data=new_df,palette='RdYlGn_r', edgecolor=sns.color_palette("YlOrBr", 10))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.savefig('terroristActivities_Years.png')
plt.show()


# #### Months with most attacks

# In[49]:


print('Month with the most attacks')
print('-----------------------------------')
print(new_df['Month'].value_counts().head(12))


# **Top 10 Groups with most attacks**

# In[50]:


print("Top 10 Group with the most attacks")
print('-----------------------------------')
print(new_df['Group'].value_counts().head(10))


# In[51]:


print("Group with the most attacks:",new_df['Group'].value_counts().index[1])


# In[52]:


#Top Groups with the most Terror Attacks
plt.subplots(figsize=(18,10))
sns.barplot(new_df['Group'].value_counts()[:10].index, new_df['Group'].value_counts()[:10], palette='rocket')
plt.title('Groups with the most Terror Attacks')
plt.xlabel('Groups')
plt.ylabel('Count')
plt.xticks(rotation= 90)
plt.savefig('top10_TerroristGroups.png')
plt.show()


# **Most Attack Types**

# In[53]:


print("Most Attack Types")
print('-----------------------------------')
print(new_df['AttackType'].value_counts())


# In[54]:


print("Most Attack Types")
print('-----------------------------------')
print(new_df['AttackType'].value_counts())


# **Most used Weapon Types**

# In[55]:


print("Most used Weapon Types")
print('-----------------------------------')
print(new_df['Weapon_type'].value_counts())


# In[56]:


plt.figure(figsize=(16,8))
sns.countplot(new_df['Weapon_type'], order=new_df['Weapon_type'].value_counts().index,
              palette='hot')
plt.xticks(rotation=90)
plt.xlabel('Weapon')
plt.title('Type of Weapons')
plt.savefig('top_WeaponTypes.png')
plt.show()


# **Most Target Types**

# In[57]:


print("Most Target Types")
print('-----------------------------------')
print(new_df['Target_type'].value_counts())


# In[58]:


print("Most Target Types")
print('-----------------------------------')
print(new_df['Target_type'].value_counts())


# In[59]:


df_region=pd.crosstab(new_df.Year, new_df.Target_type)
df_region.plot(color=sns.color_palette('Set2',12))
fig=plt.gcf()
fig.set_size_inches(16,10)
plt.title('Change of Target Types over Years')
plt.savefig('yearly_TargetTypeChange.png')
plt.show()


# In[60]:


fig,axes = plt.subplots(figsize=(16,11), nrows=1, ncols=2)
sns.barplot(y=new_df['Group'].value_counts()[1:12].index, 
            x=new_df['Group'].value_counts()[1:12].values, 
            ax=axes[0], palette='cubehelix')
axes[0].set_title('Most Active Terrorist Organizations')

#Most affected targets
sns.barplot(y=df['Target_type'].value_counts().index,
            x=df['Target_type'].value_counts().values,
            ax=axes[1], palette='tab10')
axes[1].set_title('Most Affected Targets')
fig.tight_layout()
plt.show()


# In[61]:


fig,ax = plt.subplots(figsize=(18,7),nrows=1,ncols=2)
ax[0]=new_df[new_df['Extended']==1].groupby('AttackType').count()['Extended'].sort_values().plot.barh(color='green',ax=ax[0])
ax[1]=new_df[new_df['Extended']==0].groupby('AttackType').count()['Extended'].sort_values().plot.barh(color='purple',ax=ax[1])
ax[0].set_title('Number of Extended Attacks')
ax[0].set_ylabel('Method')
ax[1].set_title('Number of Unextended Attacks')
ax[1].set_ylabel('Method')
plt.tight_layout()
plt.show()


# # THANK YOU
