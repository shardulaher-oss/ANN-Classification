import tensorflow as tf 
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle

models=tf.keras.models.load_model('models.h5')
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('one_hot_encoder_geo.pkl','rb') as file:
    one_hot_encoder_geo=pickle.load(file)
with open('scalar.pkl','rb') as file:
    scalar=pickle.load(file)

    #Sttreamlit app

st.title('Customer Churn Prediction')

#user input

geography=st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,90)
balance=st.number_input('Balance')
credit_Score=st.number_input('CreditScore')
estimated_salary=st.number_input('EstimatedSalary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('NumberOfProducts',1,4)
has_cr_card=st.selectbox("HasCreditCard",[0,1])
is_active_memeber=st.selectbox('IsActiveMember',[0,1])

# Build input DataFrame using the original dataset feature names
input_data = pd.DataFrame({
    'CreditScore': [credit_Score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_memeber],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode geography and concat (columns will be like 'Geography_France')
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled=scalar.transform(input_data)

prediction=models.predict(input_data_scaled)

st.write(f"value :{prediction}")

pred_prob=prediction[0][0]
if pred_prob>0.5:
    st.write("The Customer is churned")
else:
    st.write("The customer is not churned")
