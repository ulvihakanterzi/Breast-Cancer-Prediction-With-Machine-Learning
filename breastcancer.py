import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

kanser= pd.read_csv('breast-cancer.csv')

y = kanser[['diagnosis']]
x = kanser.drop(columns=['diagnosis', 'id'], axis=1)

x_train, x_test, y_train, y_test, = train_test_split(x,y,train_size=0.8, random_state=5)


tree = DecisionTreeClassifier()

model= tree.fit(x_train,y_train)
a = model.score(x_test,y_test)

print("Model Başarı Oranı = %" , a*100)