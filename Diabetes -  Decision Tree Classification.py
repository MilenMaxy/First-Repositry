#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries in use
import pandas as pd # Import pandas library
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics # Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

# Any more needed libraries will be mentioned below


# In[2]:


#loading dataset
df=pd.read_csv("diabetes_dataset.csv")
df.head()


# In[3]:


#feature variables
x=df.drop(['Outcome'], axis=1)
x


# In[4]:


#target variable
y=df.Outcome
y


# In[5]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# In[6]:


# Create Decision Tree classifer object
model = DecisionTreeClassifier()

# Train Decision Tree Classifer
model = model.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = model.predict(x_test)


# In[7]:


#Evaluation using Accuracy score
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


# In[8]:


confusion_matrix(y_test,y_pred)


# In[9]:


print("Accuracy:",((82+27)/154))


# In[10]:


#Evaluation using Classification report
print(classification_report(y_test,y_pred))


# In[11]:


#checking prediction value
model.predict([[6,148,72,35,0,33.6,0.627,50]])


# In[21]:


#Import modules for Visualizing Decision trees
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
#import pydotplus


# In[17]:


features=x.columns
features


# In[18]:


dot_data = StringIO()
export_graphviz(model, out_file=dot_data,filled=True, rounded=True,special_characters=True,feature_names = features,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes_set.png')
Image(graph.create_png())


# In[19]:


# Create Decision Tree classifer object
model = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
model = model.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = model.predict(x_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


# The classification rate increased to 79.87%, which is better accuracy than the previous model.
# 
# 

# In[20]:


#Better Decision Tree Visualisation
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,filled=True, rounded=True,special_characters=True, feature_names = features,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes_set.png')
Image(graph.create_png())


# In[ ]:




