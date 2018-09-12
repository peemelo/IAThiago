import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
np.set_printoptions(threshold=np.nan)

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]




df = pd.read_csv("precos_casa_california.csv")
dfHead = df.head()
dfDescribe = df.describe()
features = list(df.columns[0:])
data = list(df.lines[0:])
x= df[features]
y_name = df.columns[-1]
y = df[y_name]
df[y_name].unique()

#retirando outliers
dfNew = reject_outliers(df)

'''y_numerico = y.replace({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})


x_train, x_test, y_train, y_test = train_test_split(x, y_numerico, test_size=0.3, random_state=7)

model = DecisionTreeClassifier()
model = model.fit(x_train, y_train)

score = model.score(x_test, y_test)

export_graphviz(model, out_file='iris.dot')

joblib.dump(model, 'modelo_iris_classificacao.pkl')'''