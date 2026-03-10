import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st

# Model Training
np.random.seed(42)
df = pd.DataFrame({
    'powerplay_score': np.random.randint(20, 120, 1000),
    'powerplay_wickets': np.random.randint(0, 6, 1000),
    'result': np.random.randint(0, 2, 1000)
})

X = df[['powerplay_score', 'powerplay_wickets']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# UI
st.title("IPL Match Win Predictor")

score = st.slider("Powerplay Score", 0, 120, 50)
wickets = st.slider("Wickets Lost", 0, 6, 1)

if st.button("Predict"):
    input_data = pd.DataFrame({'powerplay_score': [score], 'powerplay_wickets': [wickets]})
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    
    if pred == 1:
        st.write("Result: WIN")
        st.write(f"Confidence: {prob[1]:.2%}")
    else:
        st.write("Result: LOSS")
        st.write(f"Confidence: {prob[0]:.2%}")
    
    st.write(f"Model Accuracy: {accuracy:.2%}")
