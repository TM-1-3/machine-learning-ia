import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv("TireProductionMalfunctions.csv") # Loading the Dataset

inputData = data.drop(columns=['date','failed_in_next_7_days']) # Data to Train the Model

target = data['failed_in_next_7_days'] # 

inputDataEncoded = pd.get_dummies(inputData, columns=['machine_name','malfunction_component']) # Converts Text Data into Numerical Values

joblib.dump(list(inputDataEncoded.columns), 'modelColumns.pkl') # Save Column Names for Web App

dataTrain, dataTest, targetTrain, targetTest = train_test_split(inputDataEncoded, target, test_size=0.2, random_state=67) # Split Data for Training and Testing

model = RandomForestClassifier(n_estimators=100, random_state=67, class_weight='balanced') # Build Model

model.fit(dataTrain, targetTrain) # Train Model

print("MODEL TESTING RESULTS")
targetPrediction = model.predict(dataTest)
print(f"\nOverall Accuracy: {accuracy_score(targetTest, targetPrediction) * 100:.2f}%\n")
print("Detailed Report:")
print(classification_report(targetTest, targetPrediction))

joblib.dump(model, 'model.pkl')