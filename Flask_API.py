from flask import Flask, jsonify, request
import joblib
import numpy as np
import json
import pandas as pd
import shap
import sys

### Append current directory to find customer transformer classes when loading model
sys.path.append('./')

#### Initializing Flask app
app = Flask(__name__)


#### Imports
path = 'Objects/'
optimal_threshold = joblib.load(path + 'optimal_threshold.bz2')
model = joblib.load(path + 'model.bz2')
X_train = joblib.load(path + 'X_train.bz2')
X_test = joblib.load(path + 'X_test.bz2')
y_train = joblib.load(path + 'y_train.bz2')
explainer = joblib.load(path + 'explainer.bz2')


#### End point definition for predictions: probability & decision
@app.route('/predictions', methods = ['GET', 'POST'])
def predictions():
    if request.method == 'POST':

     data = request.get_json()
     selected_id = int(data['index'])


     row = X_test.loc[selected_id,:]

     ### Transform row to dataframe ie into model input
     row = pd.DataFrame(row.values.reshape(1,-1), columns = row.index)

     ### Compute proba for row and apply optimal threshold
     probability = model.predict_proba(row)[:,1][0].round(decimals = 2)
     prediction = 'Default' if probability >= optimal_threshold else 'Solvency'

    return jsonify({'probability' : probability, 'prediction' : prediction})


#### End point for getting Shapley values for given index
@app.route('/shapley', methods = ['GET', 'POST'])
def shapley():
    if request.method == 'POST':
        data = request.get_json()
        selected_id = int(data['index'])
        row = X_test.loc[selected_id,:]

        ###Converts series to dataframe
        row = pd.DataFrame(row.values.reshape(1,-1), columns = row.index)
        
        ### Applying model preprocessing steps to build Shap explainer input
        preprocessed_row = model.named_steps['preprocessor'].transform(row)
        preprocessed_row = model.named_steps['features_generator'].transform(preprocessed_row)

        ### Keeping actual values to display on shap force plot
        row_actual_values = preprocessed_row.loc[:,model.named_steps['feature_selector'].get_support()]

        preprocessed_row = model.named_steps['imputer'].transform(preprocessed_row)
        preprocessed_row = model.named_steps['scaler'].transform(preprocessed_row)
        preprocessed_row = model.named_steps['feature_selector'].transform(preprocessed_row)

        shapley_values = explainer.shap_values(preprocessed_row)

    return jsonify({'explainer_expected_value' : explainer.expected_value, 'shap_values' : shapley_values.flatten().tolist(), 
    'feature_values': row_actual_values.squeeze().tolist(), 
    'feature_names' : row_actual_values.columns.tolist()})



#### End point for comparing individual with solvable & non solvable populations on single feature 
@app.route('/groups comparison', methods = ['GET', 'POST'])
def groups():
    if request.method == 'POST':
        data = request.get_json()
        selected_id = int(data['index'])
        feature = data['feature']

        ### Preprocess X_train
        train_data = model.named_steps['preprocessor'].transform(X_train)
        train_data = model.named_steps['features_generator'].transform(train_data)

        ### Reduce to final features
        train_data = train_data.loc[:,model.named_steps['feature_selector'].get_support()]

        ### Filter on selected column
        train_data = train_data[feature]

        ### Feature and target values concatenation for train set
        train_data = pd.DataFrame(np.column_stack([train_data, y_train]), columns = [feature, 'TARGET'], index = train_data.index)
    
        ### Row of interest
        row = X_test.loc[selected_id,:]
        
        ### Transforming to dataframe for processing
        row = pd.DataFrame(row.values.reshape(1,-1), columns = row.index, index = [selected_id])

        row = model.named_steps['preprocessor'].transform(row)
        row = model.named_steps['features_generator'].transform(row)
        row = row.loc[:, model.named_steps['feature_selector'].get_support()]

        ### Reduce row to selected feature value
        row_value = row.loc[:,feature].squeeze()

        ### Set selected_id as target value for row to add another class to target column : row.shape = (1,2)       
        row = pd.DataFrame([[row_value, selected_id]], columns = [feature, 'TARGET'], index = [selected_id])

        ### Concatenate row with train_data
        concat_data = pd.concat([train_data, row])
        

    return jsonify({'index' : concat_data.index.tolist(), 'columns' : concat_data.columns.tolist(), 'values' : [row.tolist() for _, row in concat_data.iterrows()]})

    

if __name__ == "__main__":
    app.run(debug=True)



 
 
 
 