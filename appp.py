import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8],
    'Salary': [25000, 30000, 35000, 45000, 50000, 60000, 65000, 70000]
}


df = pd.DataFrame(data)


X = df[['Experience']]
y = df['Salary']


model = DecisionTreeRegressor()
model.fit(X, y)


st.set_page_config(
    page_title="Decision Tree Regression",
    page_icon="📈",
    layout="centered"
)


st.title("📈 Decision Tree Regression App")

st.markdown("---")

st.write("Predict Salary based on Years of Experience")

experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)


if st.button("Predict Salary"):

    input_data = pd.DataFrame(
        [[experience]],
        columns=['Experience']
    )

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Salary: ₹ {prediction:,.2f}")

st.markdown("---")

st.caption("Built using Streamlit and Decision Tree Regressor")