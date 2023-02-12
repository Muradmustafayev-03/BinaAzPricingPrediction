from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util import format_df

df = format_df(pd.read_csv('data/modified_train_binaaz.csv'))

pivot = int(len(df.index) * 0.8)

x_train = df[[col for col in df.keys() if col != 'price']][:pivot]
x_test = df[[col for col in df.keys() if col != 'price']][pivot:]
y_train = df['price'][:pivot]
y_test = df['price'][pivot:]

reg1 = RandomForestRegressor(n_estimators=100, max_depth=180)
reg2 = GradientBoostingRegressor(n_estimators=190, max_depth=5)

er = VotingRegressor([('lf', reg1), ('gb', reg2)], weights=[0.7, 0.3])
er.fit(x_train, y_train)

print(er.score(x_test, y_test))
print(np.sqrt(mean_squared_error(y_test, er.predict(x_test))))


plt.scatter(x_test.index, y_test, marker='x')
plt.scatter(x_test.index, er.predict(x_test), marker='x')
plt.show()

