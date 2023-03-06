import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
#Loading the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('index.html')
from flask import Flask

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.ensemble import RandomForestClassifier

data = pd.read_excel('salaryOne.xlsx')
### Splitting into target and features
X = data.drop('salary',axis=1)
y = data['salary']

feature_descriptions = {
    "age": "Age of the Employee",
    "workclass": "Working class to which the employee belongs to",
    "education": "educational qualification of the employee",
    "marital-status": "Marital Status of employee",
    "occupation": "Occupation of Employee",
    "sex": "Male or Female",
    "hours-per-week": "Working Hours of the employee",
    }
### Split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=50, max_depth=5)
rf_model.fit(X_train, y_train)
explainer = ClassifierExplainer(rf_model, X_test, y_test,
                                cats=['workclass', 'education', 'marital-status', 'occupation', 'sex', 'hours-per-week'],
                                descriptions=feature_descriptions,
                                labels=['<=50k', '>50k'], 
                                target = "salary", 
                                )
db = ExplainerDashboard(explainer, 
                        title="HR Salary Dashboard", 
                        shap_interaction=False,
                        )

db = ExplainerDashboard(explainer, title="HR Salary Dashboard", server=app, url_base_pathname="/dashboard/")

@app.route('/dashboard')
def return_dashboard():
    return db.app.index()

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #features entered by user are collected and passed to model created for prediction
    int_features = [float(x) for x in request.form.values()]
    print('request.form.values()')
    print(np.array(int_features))
    final_features = [np.array(int_features)]

    output =[]

    final_features = {'age': final_features[0][0], 'workclass' : final_features[0][1], 'education':final_features[0][2], 'marital-status':final_features[0][3],'occupation':final_features[0][4], 'sex':final_features[0][5], 'hours-per-week':final_features[0][6], }
    final_features = pd.DataFrame(data=final_features, index=[0])
    prediction = model.predict(final_features)
    print(prediction)
    if (prediction[0]==0):
        output = 'Salary<=50k'
    elif(prediction[0]==1):
        output = 'Salary>50k'

    output = f'The Expected Salary of the Employee:{output}'
    # the predicted value is returned to the html
    return render_template('index.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    app.run()
