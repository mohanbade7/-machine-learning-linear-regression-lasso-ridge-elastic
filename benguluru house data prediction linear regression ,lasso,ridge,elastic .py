#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


pd.read_csv("Bengaluru_House_Data.csv")


# In[4]:


df=pd.read_csv("Bengaluru_House_Data.csv")


# In[5]:


df.isnull().sum()


# In[6]:


df.fillna(method='bfill')


# In[7]:


df.isnull().sum()


# In[8]:


df=df.fillna(method='bfill')


# In[9]:


df.isnull().sum()


# In[10]:


df=df.fillna(method='ffill')


# In[11]:


df.isnull().sum()


# In[12]:


df.columns


# In[13]:


from sklearn.preprocessing import LabelEncoder


ordinal_data=['availability','size','total_sqft']
nominal_data=['area_type','location','society']

model=LabelEncoder()
for col in ordinal_data:
    model.fit(df[col])
    df[col]=model.transform(df[col])


# In[14]:


df


# In[16]:


dff=pd.get_dummies(df[nominal_data])


# In[17]:


dff


# In[18]:


df.columns


# In[19]:


dff[['availability','size','bath','balcony','price']]=df[['availability','size','bath','balcony','price']]


# In[20]:


dff


# In[21]:




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x=dff.drop(['price'],axis=1)
y=dff['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:", mean_squared_error(test_pred,y_test))


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

x=dff.drop(['price'],axis=1)
y=dff['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error
import math

x=dff.drop(['price'],axis=1)
y=dff['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Linear regression******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))


model=Lasso(alpha=2)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Lasso Linear regression******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))


# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error
import math

x=dff.drop(['price'],axis=1)
y=dff['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Linear regression******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))


model=Lasso(alpha=2)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Lasso Linear regression******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))

model=Ridge(alpha=2)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Ridge Linear regression******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))



# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error
import math

x=dff.drop(['price'],axis=1)
y=dff['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Linear regression******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))


model=Lasso(alpha=2)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Lasso Regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))

model=Ridge(alpha=2)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Ridge regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import mean_squared_error
import math

x=dff.drop(['price'],axis=1)
y=dff['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Linear regression******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))


model=Lasso(alpha=2)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Lasso Regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))

model=Ridge(alpha=2)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Ridge regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))

model=ElasticNet(alpha=2)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Elastic regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))


# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
import math

x=dff.drop(['price'],axis=1)
y=dff['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Linear regression******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))

print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=Lasso(alpha=2)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Lasso Regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=Ridge(alpha=2)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Ridge regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=ElasticNet(alpha=2)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Elastic regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))



# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
import math

x=dff.drop(['price'],axis=1)
y=dff['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Linear regression******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))

print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=Lasso(alpha=1)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Lasso Regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=Ridge(alpha=1)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Ridge regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=ElasticNet(alpha=1)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Elastic regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))



# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
import math

x=dff.drop(['price'],axis=1)
y=dff['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Linear regression******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))

print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=Lasso(alpha=0.7)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Lasso Regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=Ridge(alpha=0.7)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Ridge regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=ElasticNet(alpha=0.7)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Elastic regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))



# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
import math

x=dff.drop(['price'],axis=1)
y=dff['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Linear regression******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))

print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=Lasso(alpha=0.9)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Lasso Regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=Ridge(alpha=0.9)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Ridge regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=ElasticNet(alpha=0.9)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Elastic regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))



# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
import math

x=dff.drop(['price'],axis=1)
y=dff['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Linear regression******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))

print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=Lasso(alpha=2.5)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Lasso Regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=Ridge(alpha=2.5)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Ridge regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))

model=ElasticNet(alpha=2.5)
model.fit(x_train,y_train)
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

print("******* Elastic regularisation******")
print("train mean squared error:", mean_squared_error(train_pred,y_train))
print("test mean squared error:",  round(mean_squared_error(test_pred,y_test),2))
print("train root mean squared error:", math.sqrt(mean_squared_error(train_pred,y_train)))
print("test root mean squared error:", math.sqrt(mean_squared_error(test_pred,y_test)))
print("train r2:",r2_score(train_pred,y_train))
print("test r2:",r2_score(test_pred,y_test))



# In[ ]:




