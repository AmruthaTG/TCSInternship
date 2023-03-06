# Importing necessary libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Loading dataset
data = pd.read_excel('salaryen.xlsx')
### Splitting into target and features
X = data.drop('salary',axis=1)
y = data['salary']
# split into train and test set


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# model training XGBoost
from xgboost import XGBClassifier
xgmodel= XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.5,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, gamma=0.0, gpu_id=-1, grow_policy='depthwise',
              importance_type=None, interaction_constraints='',
              learning_rate=0.15, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=3, max_leaves=0, min_child_weight=5,
              missing=1, monotone_constraints='()', n_estimators=100,
              n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1)
xgmodel.fit(X_train, y_train)
y_pred_xg = xgmodel.predict(X_test)


# finding accuracy
xg_acc = accuracy_score(y_test, y_pred_xg)
# Finding accuracy
print("Accuracy score", xg_acc)
#Saving model using pickle
pickle.dump(xgmodel, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))




