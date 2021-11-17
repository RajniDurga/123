#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Author-Rajni Durga 
GRIP NOVEMBER'21 THE SPARKS FOUNDATION 


# In[ ]:


TASK-1 Prediction using Supervised ML 


# In[45]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[46]:


data=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[47]:


data.head(15)


# In[48]:


data.shape


# In[49]:


data.describe()


# In[50]:


plt.scatter(data['Hours'],data['Scores'])
plt.title('Hours vs Percentage')
plt.xlabel('Studied Hours')
plt.ylabel('Scores')
plt.show()


# In[38]:


x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[39]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=0)


# In[40]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[18]:


line=regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line,color='orange');
plt.show()


# In[19]:


y_pred=regressor.predict(x_test)
print(y_pred)


# In[21]:


plt.scatter(x_train,y_train,color='pink')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage of marks')
plt.show()


# In[37]:


data=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
data


# In[41]:


data=np.array(9.25)
data=data.reshape(-1,1)
pred=regressor.predict(data)
print("If the student studies for 9.25 hours/day,the score is {}.".format(pred))


# In[27]:


from sklearn import metrics
print('Mean Absolue Error:',metrics.mean_absolute_error(y_test,y_pred))


# In[29]:


from sklearn.metrics import r2_score
print("The R-Square of the model is:",r2_score(y_test,y_pred))


# In[ ]:




