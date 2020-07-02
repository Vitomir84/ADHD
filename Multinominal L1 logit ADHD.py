#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report


# In[19]:


df = pd.read_excel("C:/Users/vitomir.jovanovic.COP/Desktop/ADHD/baza za multinominal/standardizovana baza rekodovan ishod.xlsx")
df.columns


# In[20]:


df['dijagnoza']=df['dijagnoza'].map({'ADHD':1, 'No diagnosis':0, 'Dyslexia':2})


# In[21]:


X=df.drop(['dijagnoza', 'Unnamed: 0'], axis=1)
y=df['dijagnoza']


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=22)


# In[81]:


#kreiramo multinomialni logit model sa lasso regularizacijom (l1). Solver 'saga' pruza ovu mogucnost da se oba parametra primene istovremeno, C je optimizacija penala (manje vrednosti jaca regularizacija)
lm = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.4, fit_intercept=True, intercept_scaling=1, random_state=22, solver='saga', max_iter=200, multi_class='multinomial')


# In[82]:


lm.fit(X_train,y_train)


# In[83]:


predictions = lm.predict(X_test)


# In[84]:


print(classification_report(y_test,predictions))


# In[85]:


stopa_greske=[]
optimizacija = [0.01, 0.05, 0.09, 0.15, 0.2, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

for i in optimizacija:
    lmc = LogisticRegression(C=i)
    lmc.fit(X_train,y_train)
    predictions = lmc.predict(X_test)
    stopa_greske.append(np.mean(predictions!=y_test))


# In[86]:



fig = plt.figure()

# dodajemo axes u figuri
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # velicina osa
x=optimizacija
# plotujemo ose
axes.plot(x, stopa_greske, 'b')
axes.set_xlabel('Parametar optimizacije l') # Notice the use of set_ to begin methods
axes.set_ylabel('Stopa greske')
axes.set_title('Stopa greske vs Parametar optimizacije l')


# In[ ]:


#u model treba staviti c=0.01 kao najbolji parametar optimizacije

