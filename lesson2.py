import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm


df = pd.read_csv("/home/orkhan/Downloads/data_exercise.csv")


x = df[['input_1', 'input_2','input_3','input_4']]
y = df['output']

regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)

print('Coefficients: \n', regr.coef_)

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x)

print_model = model.summary()


print(print_model)