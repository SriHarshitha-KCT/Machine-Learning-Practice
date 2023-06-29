#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyforest


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer = load_breast_cancer()


# In[4]:


df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[5]:


df.head()


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[8]:


scaler=StandardScaler()


# In[15]:


scaled_data=scaler.fit_transform(df)


# In[16]:


from sklearn.decomposition import PCA


# In[17]:


pca=PCA(n_components=3)


# In[22]:


pca_data=pca.fit_transform(scaled_data)


# In[23]:


pca_data.shape


# In[26]:


data=pd.DataFrame(pca_data)


# In[27]:


data


# In[30]:


x=data
y=cancer["target"]


# In[34]:


from sklearn.linear_model import LogisticRegression


# In[35]:


model=LogisticRegression()


# In[39]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.80)


# In[40]:


model.fit(x_train,y_train)


# In[41]:


y_pred=model.predict(x_test)


# In[42]:


y_pred


# In[43]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)


# In[44]:


score


# In[ ]:




