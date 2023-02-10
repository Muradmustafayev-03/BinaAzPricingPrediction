from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from util import format_df


df = format_df(pd.read_csv('data/modified_train_binaaz.csv', index_col=False))

x_train = df[['floors', 'area', 'rooms', 'new_building', 'longitude', 'latitude']][:18000]
x_test = df[['floors', 'area', 'rooms', 'new_building', 'longitude', 'latitude']][18000:]
y_train = df['price'][:18000]
y_test = df['price'][18000:]

reg = RandomForestRegressor()
reg.fit(x_train, y_train)

print(reg.score(x_test, y_test))
print(mean_squared_error(y_test, reg.predict(x_test)))
