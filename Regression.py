#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing useful libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


#Reading data
#from crime_dataset_headers import *
#data_url='https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'
#crime=pd.read_csv(data_url, header=None, mames=headers, na_values='?', index_col='communityname')


# In[8]:


#Change path to the file when needed.
df=pd.read_csv('/Users/lam/PredictiveAnalytics/EngineeringTest/POS_data_2018.csv')


# In[57]:


df.head()


# In[60]:


df['invoice_closed']=pd.to_datetime(df['invoice_closed'])
df['invoice_opened']=pd.to_datetime(df['invoice_opened'])
df['time']=pd.to_datetime(df['time'])


# In[61]:


df.head()


# In[62]:


feature_names=['invoice_closed','invoice_opened','guests',
               'invoice','cancellation','ticket','table','time']

target_name='article_number'


# In[63]:


feature_names


# In[64]:


target_name


# In[65]:


X=df[feature_names]
y=df[target_name]


# In[66]:


X.head()


# In[67]:


y.head()


# In[68]:


#Split the data into: trainning and testing (cross-validation)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=123)


# In[69]:


# Import the estimator object (model)
from sklearn.linear_model import LinearRegression


# In[70]:


#Create an instance of the estimator
linear_regression_model=LinearRegression()


# In[71]:


#Use the trainning data to train the estimator
linear_regression_model.fit(X_train,y_train)


# In[ ]:


# Evaluate the model
from sklearn.metrics import mean_squared_error
# Get the predictions of the model for the data it has not seen (testing)
y_pred_test=linear_regression_model.predict(x_test)
#All the metrics compare in some way how close are the predicted vs. the actual values
error_metric=mean_squared_error(y_pred=y_pred_test,y_true=y_test)
print('The Mean Squqre Error of this model is: ',error_metric)


# In[ ]:


fig, ax=plt.subplots()
ax.scatter(y_test,y_pred_test)
ax.plot(y_test,y_test,color='red')
ax.set_xlabel('Testing target values')
ax.set_ylabel('Predicted target values')
ax.set_title('Predicted vs. Actual values')


# In[ ]:


#Make predictions
from collections import OrderedDict
new_data=OrderedDict([('Time'],2020-11-29)
                     ])
#.values.reshape(1,-1) because it must be 2-dim, because we passed only one new observation
new_data=pd.Series(new_data).values.reshape(1,-1)
#Use the model to make predictions
linear_regression_model.predict(new_data)

