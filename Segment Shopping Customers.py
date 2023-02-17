#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd  #data manipulation library
import seaborn as sns #visualization library, for statistical visualization
import matplotlib.pyplot as plt #another visualization libray
from sklearn.cluster import KMeans #Clustering library from sklearn
import warnings 
warnings.filterwarnings('ignore') #ignoring warnings


# In[2]:


df = pd.read_csv("C:/Users/diami/OneDrive/Documents/Mall_Customers.csv")


# In[3]:


df.head()


# In[ ]:





# # Univariate Analysis

# In[4]:


df.describe()


# In[5]:


df.columns


# In[6]:


#creating a for loop to create histograms for each numerical variables, avoiding repetition of code

columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[7]:


#sns.kdeplot(df['Annual Income (k$)'], shade = True, hue = df['Gender'] ); #kdeplot allow us to bring different parameter like shade and hue(break by dimension)


# In[8]:


columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(df[i], shade = True, hue = df['Gender'] )


# In[9]:


columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data = df, x = 'Gender', y = df[i] )


# In[10]:


df['Gender'].value_counts(normalize = True) #56% female and 44% male


# # Bivariate Analysis

# In[11]:


# Scatterplot
columns = ['Annual Income (k$)', 'Age']
for i in columns:
    plt.figure()
    sns.scatterplot(data = df, x = df[i] , y = 'Spending Score (1-100)');


# In[ ]:


dff = df.drop('CustomerID', axis =1 )
sns.pairplot(dff, hue = 'Gender')


# In[13]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[14]:


#correlation using table
df.corr()


# In[15]:


#correlation using heatmap
sns.heatmap(dff.corr(), annot=True, cmap ='coolwarm')


# # Clustering - Univariate , Bivariate 

# In[16]:


clustering1 = KMeans(n_clusters = 3) #n_clusters determine using elbow method down below


# In[17]:


clustering1.fit(df[['Annual Income (k$)']])


# In[18]:


clustering1.labels_


# In[19]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[20]:


#how many of our customer fall in each cluster
df['Income Cluster'].value_counts()


# In[21]:


#how many clusters do we actually need , Doing the elbow method for number of clusters
clustering1.inertia_


# In[22]:


inertia_scores=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)
inertia_scores


# In[23]:


plt.plot(range(1,11), inertia_scores) #looks like the elbow start in 3, so n_clusters = 3


# In[24]:


df.columns


# In[25]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[26]:


#Bivariate Clustering


# In[27]:


#clustering using Annual Income and Spending Score


# In[28]:


clustering2 = KMeans( n_clusters = 5) 
clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df['Spending and Income Cluster'] = clustering2.labels_
df.head()


# In[29]:


inertia_scores2=[]
for i in range(1,11):
    kmeans2 = KMeans(n_clusters = i)
    kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11), inertia_scores2) #looks like the elbow start in 5, so n_clusters = 5


# In[30]:


centers = pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x', 'y']


# In[31]:


plt.figure(figsize = (10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data = df, x = 'Annual Income (k$)', y= 'Spending Score (1-100)', hue = 'Spending and Income Cluster', palette = 'tab10')


# In[32]:


#Looks like cluster 3 is the ideal cluster with high income and high spending score
# From the crosstab we can see that 


# In[33]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[34]:


df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean() #we can see the average age, income and spending score of each income


# In[35]:


#clustering using Age and Spending Score


# In[36]:


clustering3 = KMeans(n_clusters =4) 
clustering3.fit(df[['Age', 'Spending Score (1-100)']])
df['Spending and Age Cluster'] = clustering3.labels_
df.head()


# In[37]:


inertia_scores3=[]
for i in range(1,11):
    kmeans3 = KMeans(n_clusters = i)
    kmeans3.fit(df[['Age', 'Spending Score (1-100)']])
    inertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11), inertia_scores3) #looks like the elbow start in 4, so n_clusters = 4


# In[38]:


centers2 = pd.DataFrame(clustering3.cluster_centers_)
centers2.columns = ['x', 'y']


# In[39]:


plt.figure(figsize = (10,8))
plt.scatter(x=centers2['x'],y=centers2['y'],s=100,c='black',marker='*')
sns.scatterplot(data = df, x = 'Age', y= 'Spending Score (1-100)', hue = 'Spending and Age Cluster', palette = 'tab10');


# In[ ]:




