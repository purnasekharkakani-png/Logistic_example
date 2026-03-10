import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st

# Data Loading & Model Training
def load_model():
    np.random.seed(42)
    df = pd.DataFrame({
        'powerplay_score': np.random.randint(20, 120, 1000),
        'powerplay_wickets': np.random.randint(0, 6, 1000),
        'result': np.random.randint(0, 2, 1000)
    })
    
    X = df[['powerplay_score', 'powerplay_wickets']]
    y = df['result']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return model, accuracy

model, accuracy = load_model()

# UI Layout
st.title("IPL Powerplay Win Predictor")
st.write("Predict match result based on powerplay performance")

powerplay_score = st.slider(
    "Powerplay Score",
    min_value=0,
    max_value=120,
    value=50
)

powerplay_wickets = st.slider(
    "Powerplay Wickets Lost",
    min_value=0,
    max_value=6,
    value=1
)

if st.button("Predict Result"):
    test_data = pd.DataFrame(
        [[powerplay_score, powerplay_wickets]],
        columns=['powerplay_score', 'powerplay_wickets']
    )
    
    prediction = model.predict(test_data)[0]
    probability = model.predict_proba(test_data)[0]
    
    if prediction == 1:
        st.success(f"Prediction: Team will WIN (Confidence: {probability[1]:.1%})")
    else:
        st.error(f"Prediction: Team will LOSE (Confidence: {probability[0]:.1%})")

st.write(f"Model Accuracy: {accuracy:.2%}")
