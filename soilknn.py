import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
sns.set()

soildata=pd.read_csv('C:/Users/Karthik/Desktop/bacup/rgbmean.csv')

soildata.head(5)
classes=soildata['Category'].unique()

import pandas as pd
y=['Category']
print(y)
y_target=(pd.Categorical(y).codes)
print(y_target)

soildata.shape

SOIL_DATA=np.array(soildata)
print(type(SOIL_DATA))

X=SOIL_DATA[:,0:4-1]
Y=SOIL_DATA[:,-1]

print(X.shape)
print(Y.shape)

pd.DataFrame(X,columns=['Red','Green','Blue'])

X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state=42,test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)

print(classes)

aa=confusion_matrix(y_test,y_pred,labels=classes)
print(aa)

test=[11.898,43.799,58.761]
test=np.expand_dims(test,0)
print(test.shape)

knn.predict(test) #array(['C2'], dtype=object)
knn.score(X_test,y_test)*100 #95.0

file = open("knn_model.pkl","wb")
pickle.dump(knn,file)
file = open("knn_model.pkl","rb")
model = pickle.load(file)
prediction = model.predict(X_test)
print(accuracy_score(y_test,prediction))
