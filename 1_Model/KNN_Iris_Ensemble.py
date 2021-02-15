import pandas as pd

data = pd.read_csv("bezdekIris.data")

columns = ['Sepel Length','Sepel Width','Petal Length','Petal Width','Flower']

data.columns=columns

X = data[['Sepel Length','Sepel Width','Petal Length','Petal Width']]
Y = data[['Flower']]
