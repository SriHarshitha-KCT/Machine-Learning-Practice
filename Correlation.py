#!/usr/bin/env python
# coding: utf-8

# In[38]:


import warnings
warnings.filterwarnings('ignore')

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[39]:


cars = pd.read_csv(R"C:\Users\Sri_Harshitha\Downloads\archive\CarPrice_Assignment.csv")
cars.head()


# In[40]:


#Splitting company name from CarName column
CompanyName = cars['CarName'].apply(lambda x : x.split(' ')[0])
cars.insert(3,"CompanyName",CompanyName)
cars.drop(['CarName'],axis=1,inplace=True)
cars.head()


# In[41]:


cars.CompanyName = cars.CompanyName.str.lower()

def replace_name(a,b):
    cars.CompanyName.replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

cars.CompanyName.unique()


# In[42]:


cars['fueleconomy'] = (0.55 * cars['citympg']) + (0.45 * cars['highwaympg'])


# In[43]:


cars['price'] = cars['price'].astype('int')
temp = cars.copy()
table = temp.groupby(['CompanyName'])['price'].mean()
temp = temp.merge(table.reset_index(), how='left',on='CompanyName')
bins = [0,10000,20000,40000]
cars_bin=['Budget','Medium','Highend']
cars['carsrange'] = pd.cut(temp['price_y'],bins,right=False,labels=cars_bin)
cars.head()


# In[44]:


cars_lr = cars[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 
                    'fueleconomy', 'carlength','carwidth', 'carsrange']]
def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df
# Applying the function to the cars_lr

cars_lr = dummies('fueltype',cars_lr)
cars_lr = dummies('aspiration',cars_lr)
cars_lr = dummies('carbody',cars_lr)
cars_lr = dummies('drivewheel',cars_lr)
cars_lr = dummies('enginetype',cars_lr)
cars_lr = dummies('cylindernumber',cars_lr)
cars_lr = dummies('carsrange',cars_lr)


# In[45]:


cars_lr.head()


# In[46]:


from sklearn.preprocessing import StandardScaler
scaled_data=StandardScaler().fit_transform(cars_lr)


# In[47]:


data=pd.DataFrame(scaled_data,columns=cars_lr.columns)


# In[48]:


plt.figure(figsize = (25,25))
sns.heatmap(data.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


x=data.iloc[:,1:]
y=data.iloc[:,0]


# In[51]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.80)


# In[53]:


x_train


# In[55]:


y_train


# In[58]:


from sklearn.linear_model import LinearRegression


# In[59]:


model=LinearRegression()


# In[60]:


model.fit(x_train,y_train)


# In[61]:


y_pred=model.predict(x_test)


# In[62]:


y_pred


# In[63]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print('R-squared score:', r2)


# In[ ]:




