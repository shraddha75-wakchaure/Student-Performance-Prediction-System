import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("c:/Users/Shraddha Wakchaure/AppData/Local/Packages/5319275A.WhatsAppDesktop_cv1g1gvanyjgm/LocalState/sessions/2FBD92F5348CA5717A826F505B721A67453D8415/transfers/2026-16/student_data.csv")

# Select features
data = data[['studytime', 'absences', 'G1', 'G2', 'G3']]

# Define X and y
X = data[['studytime', 'absences', 'G1', 'G2']]
y = data['G3']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")