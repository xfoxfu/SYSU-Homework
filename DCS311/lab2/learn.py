import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

col_names = ['buying','maint','dorrs','persons','lug_boot','safety','Label']
pima = pd.read_csv("./lab2_dataset/DecisionTree验收数据.csv", header=None, names=col_names)
pima.head()

feature_cols = ['buying','maint','dorrs','persons','lug_boot','safety']
X = pima[feature_cols] # Features
y = pima['Label'] # Target variable

Xh = pd.get_dummies(pima[feature_cols])

clf = DecisionTreeClassifier()
clf = clf.fit(Xh[0:15],y[0:15])

yt = clf.predict(Xh[15:18])

for i in range(15,18):
    pima['Label'][i] = yt[i-15]

print(pima[15:18])

# print(tree.export_graphviz(clf))

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
      filled=True, rounded=True,
      special_characters=True,feature_names = list(Xh),class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('cart-comp.pdf')
Image(graph.create_png())
