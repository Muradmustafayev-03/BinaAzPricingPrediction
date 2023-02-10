from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from util import format_df


df = format_df(pd.read_csv('data/modified_train_binaaz.csv'))

pivot = int(len(df.index) * 0.8)

x_train = df[['floor', 'floors', 'area', 'new_building', 'longitude', 'latitude']][:pivot]
x_test = df[['floor', 'floors', 'area', 'new_building', 'longitude', 'latitude']][pivot:]
y_train = df['price'][:pivot]
y_test = df['price'][pivot:]

reg = RandomForestRegressor()
reg.fit(x_train, y_train)

print(reg.score(x_test, y_test))
print(mean_squared_error(y_test, reg.predict(x_test)))
