#import libraries
import pandas as pd
import numpy as np
import pickle   

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('OnlineNewsPopularity.csv')
print(df.head())

  # fixes leading-space column names
df.columns = df.columns.str.strip()  

#drop url column
df = df.drop(columns=["url"]) 

#feature engineering
df['log_shares'] = np.log1p(df['shares'])

#splitting x and y
X = df.drop(columns=["shares", "log_shares"])
y = df["log_shares"]

feature_columns = list(X.columns) 

#train test split
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(
        n_estimators=150,
        random_state=42,
    )),
])
model_pipeline.fit(X_train, y_train) 
y_pred = model_pipeline.predict(X_test)
print("Gradient Boosting (final model)")
print("R2 Score :", r2_score(y_test, y_pred))
print("RMSE     :", mean_squared_error(y_test, y_pred) ** 0.5)
print("MAE      :", mean_absolute_error(y_test, y_pred))

with open("model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)
 
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)
 
print(f"\nSaved model.pkl (Pipeline w/ scaler) and feature_columns.pkl "
      f"({len(feature_columns)} columns)")