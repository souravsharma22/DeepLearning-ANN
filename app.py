import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import tensorflow as tf
import pickle


# loading model
model = tf.keras.models.load_model('model.h5')

with open('ohe_gender.pkl','rb') as file:
    ohe_geo = pickle.load(file)

with open('LabelEncoder_gender.pkl','rb') as file:
    label_encoder = pickle.load(file)

with open('Scaler.pkl','rb') as file:
    Scaler = pickle.load(file)

#streamlit app creation
st.title("Customer churn Prediction")

geography = st.selectbox('Geography',ohe_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('CreditScore')
estimated_salary = st.number_input('EstimatedSalary')
tenure = st.slider('Tenure',0,10)
No_of_Prodect = st.slider('NumOfProducts',1,4)
HasCrCard = st.selectbox('HasCrCard',[0,1])
IsActiveMember	=st.selectbox('IsActiveMember',[0,1])

input_data = pd.DataFrame({
    "CreditScore":[credit_score],
    "Gender":[label_encoder.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts": [No_of_Prodect],
    "HasCrCard": [HasCrCard],
    "IsActiveMember": [IsActiveMember],
    "EstimatedSalary": [estimated_salary]
})

geo_data = ohe_geo.transform([[geography]]).toarray()
geo_data_df = pd.DataFrame(geo_data,columns=ohe_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_data_df],axis=1)

Scaled_input= Scaler.transform(input_data)

prediction =model.predict(Scaled_input)

prediction_proba = prediction[0][0]

st.write(f"Churn Probability :{prediction_proba:.2f}")
if prediction_proba>0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")