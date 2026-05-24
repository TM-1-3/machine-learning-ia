import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("TireProductionMalfunctions.csv") 

inputData = data.drop(columns=['date','failed_in_next_7_days']) 

target = data['failed_in_next_7_days']  

inputDataEncoded = pd.get_dummies(inputData, columns=['machine_name','last_malfunction_component']) 

joblib.dump(list(inputDataEncoded.columns), 'modelColumns.pkl') 

dataTrain, dataTest, targetTrain, targetTest = train_test_split(inputDataEncoded, target, test_size=0.2, random_state=67) 

print("--- 1. Random Forest ---")
rfModel = RandomForestClassifier(n_estimators=100, random_state=67, class_weight='balanced')
rfModel.fit(dataTrain, targetTrain)
rfPrediction = rfModel.predict(dataTest)
print(f"Accuracy: {accuracy_score(targetTest, rfPrediction) * 100:.2f}%\n")
joblib.dump(rfModel, 'rfModel.pkl')

print("--- 2. Decision Tree ---")
dtModel = DecisionTreeClassifier(random_state=67, class_weight='balanced', max_depth=5, min_samples_leaf=15)
dtModel.fit(dataTrain, targetTrain)
dtPrediction = dtModel.predict(dataTest)
print(f"Accuracy: {accuracy_score(targetTest, dtPrediction) * 100:.2f}%\n")
joblib.dump(dtModel, 'dtModel.pkl')

print("--- 3. Gradient Boosting ---")
gbModel = GradientBoostingClassifier(random_state=67)
gbModel.fit(dataTrain, targetTrain)
gbPrediction = gbModel.predict(dataTest)
print(f"Accuracy: {accuracy_score(targetTest, gbPrediction) * 100:.2f}%\n")
joblib.dump(gbModel, 'gbModel.pkl')
