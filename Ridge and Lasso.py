#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pyforest
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')


# In[14]:


dataset=load_boston()


# In[16]:


dataset


# In[25]:


data=pd.DataFrame(dataset.data)


# In[26]:


data.head()


# In[27]:


data.columns=dataset.feature_names


# In[29]:


X=data.iloc[:,:-1] ## independent features
y=data.iloc[:,-1]


# In[32]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)


# In[33]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[34]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[36]:


prediction_lasso=lasso_regressor.predict(X_test)
prediction_ridge=ridge_regressor.predict(X_test)


# In[ ]:




