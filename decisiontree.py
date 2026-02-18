import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
X = np.array([[30],[40],[50],[60],[20],[10],[70]],dtype=float)
y = np.array([0,1,1,1,0,0,1],dtype=float)
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X,y)
X_marks=np.array([[20]],dtype=float)
print(classifier.predict(X_marks))
