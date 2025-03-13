#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data handling libraries
import pandas as pd #Used for data manipulation and analysis, especially handling tabular data.
import numpy as np #Supports numerical computations, arrays, and mathematical functions.

# data visualization libraries
import matplotlib.pyplot as plt #A plotting library for creating visualizations such as line plots, histograms, and scatter plots.
import seaborn as sns #Built on top of Matplotlib, it provides enhanced visualizations like heatmaps, violin plots, and pair plots.

from sklearn.model_selection import train_test_split #train_test_split: Splits the dataset into training and testing sets, commonly used in supervised learning.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# # loading the dataset

# In[2]:


df = pd.read_csv('creditcard.csv')


# In[3]:


df


# In[4]:


df.head() # displays initial 5 rows


# # Exploratory Data Analysis (EDA)

# In[5]:


# Check for missing values
df.isnull().sum()


# In[6]:


# Summary statistics
df.describe()


# In[7]:


# Distribution of the target variable (Class)
sns.countplot(x='Class', data=df)
plt.title('Distribution of Fraud and Non-Fraud Transactions') 
plt.show()
# here 0 is for non fraud trasactions and 1 is for fraud transactions


# In[8]:


# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Matrix')
plt.show()


# # data preprocessing

# In[9]:


# Separate the features and the target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# # Model Training:

# In[10]:


# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)


# # Model Evaluation:

# In[11]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[12]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[13]:


# Classification report
print(classification_report(y_test, y_pred))


# In[14]:


# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# # Feature Importance:

# In[15]:


# Get feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance')
plt.show()


# In[ ]:




