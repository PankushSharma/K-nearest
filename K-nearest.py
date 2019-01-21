#K-Nearest program
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = datasets.load_iris()
x = iris.data[:,0:4]
y = iris.target
x_train,x_test, y_train, y_test = train_test_split(x, y,test_size=0.33,stratify=y)
y_train.shape,y_test.shape
kn=[1,3,5,7,9,11,13,15,17,19,21]
a=[]
for k in kn:
    dummy=[]
    for x in x_test:
        eqdist = np.power(x_train-x,2)
        sort=np.argsort(eqdist,axis=0)[:k]
        #print(sort)
        y1=y_train[sort]
        #print(y1)
        number,count = np.unique(y1,return_counts=True)
        #print("number ",number)
        #print(count)
        dummy=np.append(dummy,np.argmax(count))
        #print("count",dummy)
    a.append((sum(dummy==y_test)/len(y_test))*100)
print("Accuracy of test data set at diffrent Nearest neighbours")    
print(a)
