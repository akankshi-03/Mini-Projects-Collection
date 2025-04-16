# Lung Cancer Risk Prediction using Decision Tree Classifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("lung_cancer_data.csv")  # Add dataset in the same folder

# Separate features and labels
X = data.drop('Risk', axis=1)
y = data['Risk']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print("Lung Cancer Risk Prediction Report:
")
print(classification_report(y_test, predictions))
