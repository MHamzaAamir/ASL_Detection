import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

newdata = []
newlabels = []
for i in range(len(data)):
    if (len(data[i])) != 84:
        newdata.append(data[i])
        newlabels.append(labels[i])

newdata = np.array(newdata)
newlabels = np.array(newlabels)

x_train,x_test,y_train,y_test = train_test_split(newdata,newlabels,test_size=0.2,stratify=newlabels)

model = RandomForestClassifier()

model.fit(newdata,newlabels)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict,y_test)

print(score)


f = open('model.pickle', 'wb')
pickle.dump({'model': model}, f)
f.close()