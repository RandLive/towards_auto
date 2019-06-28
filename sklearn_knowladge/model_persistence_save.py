# -*- coding: utf-8 -*-
"""
sklearn
"""

from sklearn import svm
from sklearn import datasets
clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)  

# In[save and load model]
from joblib import dump, load
dump(clf, 'clf.joblib') 
clf = load('clf.joblib') 
clf.predict(X[0:1])

print(y[0])