import joblib
import numpy as np
import pandas as pd
import shap
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import time
import re
import requests
import json
import sys
sys.path.append('./')



### Page configuration and title
st.set_page_config(page_title= 'Decision dashboard', page_icon=None, layout='wide', initial_sidebar_state='auto')
st.title('Decision dashboard')


#### Progress bar
my_bar = st.progress(0)
for percent_complete in range(100):
 time.sleep(0.1)
 my_bar.progress(percent_complete + 1)



#### Importing
@st.cache
def importing():
 
 path = 'Objects/'
 index_ = joblib.load(path + 'X_test.bz2').index
 coef = joblib.load(path + 'coef.bz2').flatten()
 descriptions = pd.read_csv('Data/HomeCredit_columns_description.csv', encoding = 'unicode_escape')
 features = joblib.load(path + 'model_features.bz2')

 return index_, coef, descriptions, features

### Indices of loans, feature coefficients, descriptions & features of the model
index_, coef, descriptions, features = importing()


st.sidebar.markdown("### " + "Which loan would you like to investigate?")

#### Select loan
selected_id = st.sidebar.selectbox(
    label="Select loan ID", options= index_
)


#### Request API at corresponding endpoint
url = 'http://127.0.0.1:5000/predictions'
r = requests.post(url, json = {'index' : str(selected_id)})

r = r.json()
probability = r['probability']
prediction = r['prediction']

### Initialization of two columns within application interface
col_1, col_2 = st.beta_columns(2)
	

#### Headers
col_1.header('Probability of default')
col_2.header('Prediction')


#### Markdowns of predictions received from API
col_1.markdown('### ' + str(probability))
col_2.markdown('### ' + str(prediction))



#### Most important features
st.sidebar.markdown('### ' + 'Most important features')

#### Initializing container within sidebar
container = st.sidebar.beta_container()

### Palette with two colors : one for each coefficient sign, positive or negative
palette = sns.color_palette(palette = 'icefire', n_colors = 2)


list_of_colors = [(palette[0] if coef_ >= 0 else palette[1]) for coef_ in coef]
sorted_features, sorted_coef, sorted_colors = zip(*sorted(zip(features, np.abs(coef), list_of_colors), key = lambda k : k[1]))


#### Plot most important features
with container:
 fig = plt.figure(figsize= (9,18))
 ax = fig.add_subplot(111)
 ax.barh(y = np.arange(0,10,1), width = sorted_coef[-10:], color = sorted_colors[-10:])
 ax.set_yticks(np.arange(0,10,1))
 ax.set_yticklabels(sorted_features[-10:], fontsize = 40)
 st.pyplot(fig)







#### Prediction explanation with Shapley values
st.header('Prediction explanation')


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


#### Request API for Shapley values of the client of interest
url = 'http://127.0.0.1:5000/shapley'
r = requests.post(url, json = {'index' : str(selected_id)})

r = r.json()
explainer_expected_value = r['explainer_expected_value']
shap_values = r['shap_values']
feature_values = r['feature_values']
feature_names = r['feature_names']

st_shap(shap.force_plot(
    base_value = explainer_expected_value, shap_values = np.array(shap_values), features = np.array(feature_values),
    feature_names= feature_names))



#### Initializing columns for individual vs groups comparison on single variable

col_3, col_4 = st.beta_columns([2,1])

col_4.markdown("#### " + "What variable would you like to investigate?")


### Select variable - features sorted by abs coef value
selected_var = col_4.selectbox(
    label="Select variable", options= sorted_features[::-1]
)

### Variables description text 
col_4.subheader('Variable description')


###Variable description retrieved in descriptions dataframe

if selected_var not in descriptions['Row'].values:
 string = selected_var[::-1]
 string = re.split('\w_', string, 1)[1]
 string = string[::-1]
 if string in descriptions['Row'].values:
  col_4.markdown('## ' + descriptions[descriptions['Row'] == string]['Description'].values[0])
else:
 col_4.markdown('## ' + descriptions[descriptions['Row'] == selected_var]['Description'].values[0])




#### Request API giving client & feature expecting to receive a dataframe containing two columns: feature of interest & target
url = 'http://127.0.0.1:5000/groups comparison'
r = requests.post(url, json = {'index' : str(selected_id), 'feature' : selected_var})

r = r.json()
values = r['values']
columns = r['columns']
index = r['index']

df = pd.DataFrame(np.array(values), columns = columns, index = index)


#### Barplot
with col_3:
 st.header('Groups comparison')
 fig = plt.figure(figsize = (9,6))
 g = fig.add_subplot(1, 1, 1)
 g = sns.barplot(x = 'TARGET', y = selected_var, data = df)
 g.set_xticks(g.get_xticks())
 g.set_xticklabels(['No', 'Yes', 'ID'], fontsize = 8)
 g.set_xlabel('Default', fontsize = 18)
 g.set_ylabel('')
 g.set_title(selected_var, fontsize = 12)
 plt.yticks(fontsize = 8)
 st.pyplot(fig)

















