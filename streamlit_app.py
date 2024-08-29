import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

st.write("""
# Player Clustering Prediction App
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    minutes_played = st.sidebar.slider('Minutes Played', 0, 9510, 5000)
    current_value = st.sidebar.slider('Value', 0, 180000000, 90000000)
    age = st.sidebar.slider('Age', 15, 43, 32)
    data = {'Minutes Played': minutes_played,
            'Value': current_value,
            'Age': age}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

try:
    clf = joblib.load('kmeans_model.joblib')
    scaler = joblib.load('kmean_model/scaler.joblib')
except Exception as e:
    st.error(f"Error loading models: {e}")

def predict(data):    
    try:
        data_scaled = scaler.transform(data)  # Scale the input data
        return clf.predict(data_scaled)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

if st.button('Predict Player Cluster'):
    result = predict(df)
    if result is not None:
        st.text(f'The player belongs to cluster: {result[0]}')
