# Import required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

# Features (first 196 columns) and target (LABEL column)
X = df.iloc[:, :-1]
y = df["LABEL"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


pipeline_knn = Pipeline([
    ("scaler", StandardScaler()),        
    ("knn", KNeighborsClassifier(n_neighbors=3))  
])

# Train the pipeline
pipeline_knn.fit(X_train, y_train)

# Predictions
y_pred = pipeline_knn.predict(X_test)

# Evaluation
print("Pipeline (KNN) Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
