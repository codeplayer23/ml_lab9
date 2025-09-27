# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

# Features (first 196 columns) and target (LABEL column)
X = df.iloc[:, :-1]
y = df["LABEL"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Base models
base_models = [
    ('knn', KNeighborsClassifier(n_neighbors=3)),
    ('svm', SVC(probability=True, kernel='rbf')),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('ada', AdaBoostClassifier(random_state=42)),
    ('nb', GaussianNB()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
]

# Meta model 
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# Stacking classifier
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,                
    n_jobs=-1
)

# Train stacking classifier
stack_model.fit(X_train, y_train)

# Predictions
y_pred = stack_model.predict(X_test)

# Evaluation
print("Stacking Classifier Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
