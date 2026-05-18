import streamlit as st
import pandas as pd
import joblib


model = joblib.load("model.pkl")
modelColumns = joblib.load("modelColumns.pkl")

data = pd.read_csv("TireProductionMalfunctions.csv")

st.set_page_config(
    page_title="Tire Machine Failure Predictor",
    layout="centered"
)

st.title("Machine Failure Predictor")

categoricalColumns = ['machine_name', 'malfunction_component']
targetColumn = 'failed_in_next_7_days'
ignoredColumns = ['date', targetColumn]

numericColumns = [
    col for col in data.columns
    if col not in categoricalColumns + ignoredColumns
]

st.sidebar.markdown("""

---

### How the model was trained
- Dataset: Tire production machine logs
- Target: Failure in next 7 days (binary classification)
- Preprocessing:
  - One-hot encoding for categorical variables
- Model:
  - Random Forest Classifier
- Evaluation:
  - Accuracy + classification report
""")
st.header("About the app")
st.subheader("How did we train the model?")
st.write("""
         Using a big data set and some python libraries, we trained a model that tries to predict
         with high precision how likely a machine, in a tire factory,
         is to fail in the next 7 days.

         The data set is considerably large (20 thousand lines) so we have a solid foundation and information to get a precise result.
         """
        ) 
st.subheader("How to use?")
st.write("""
            1. Select the machine and component parameters
            2. Adjust values (slider or manual input) containing the machine age, how many days since the last malfunction and how many hours of work the machine has.
            3. Click **Predict Failure Risk**
            4. View result and probability
         """)

st.header("Machine Information")

userInput = {}

for col in categoricalColumns:
    options = data[col].unique()

    userInput[col] = st.selectbox(
        col.replace("_", " ").title(),
        options
    )

for col in numericColumns:

    minValue = float(data[col].min())
    maxValue = float(data[col].max())
    meanValue = float(data[col].mean())


    input_name = col.replace("_", " ")
    mode = st.radio(f"{input_name} input mode", ["Slider", "Manual"], key=col)

    if mode == "Slider":
        userInput[col] = st.slider(
            col.replace("_", " ").title(),
            min_value=minValue,
            max_value=maxValue,
            value=meanValue
        )
    else:
        userInput[col] = st.number_input(
            col.replace("_", " ").title(),
            min_value=minValue,
            max_value=maxValue,
            value=meanValue
        )


if st.button("Predict Failure Risk"):

    inputDF = pd.DataFrame([userInput])

    inputEncoded = pd.get_dummies(
        inputDF,
        columns=categoricalColumns
    )

    inputEncoded = inputEncoded.reindex(
        columns=modelColumns,
        fill_value=0
    )

    prediction = model.predict(inputEncoded)[0]

    probability = model.predict_proba(inputEncoded)[0][1]

    st.header("Prediction Result")

    if prediction == 1:
        st.error(
            f"High Failure Risk Detected ({probability:.2%} probability)"
        )
    else:
        st.success(
            f"Low Failure Risk ({probability:.2%} probability)"
        )

    st.progress(float(probability))

    st.write(f"Failure Probability: {probability:.2%}")

    st.header("Most Important Features")

    importanceDF = pd.DataFrame({
        'Feature': modelColumns,
        'Importance': model.feature_importances_
    })

    importanceDF = importanceDF.sort_values(
        by='Importance',
        ascending=False
    ).head(10)

    st.bar_chart(
        importanceDF.set_index('Feature')
    )
