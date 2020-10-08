#!/usr/bin/env python
# coding: utf-8

# ## SVM - classification

# In[1]:


from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)


# In[3]:


clf.predict([[2., 2.]])


# In[4]:


# get support vectors
clf.support_vectors_


# In[5]:


# get indices of support vectors
clf.support_


# In[6]:


# get number of support vectors for each class
clf.n_support_


# ## Multi-class classification

# In[7]:


X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y)


# In[11]:


dec = clf.decision_function([[1]])
dec.shape[1]


# In[12]:


clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1]


# In[13]:


lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y)


# In[14]:


dec = lin_clf.decision_function([[1]])
dec.shape[1]

