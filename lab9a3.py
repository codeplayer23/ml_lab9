import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer

# Load dataset
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

# Features (all except last col), Target (last col)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define pipeline (scaler + classifier)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=3))
])

# Train pipeline
pipeline.fit(X_train, y_train)
print("Pipeline Accuracy:", accuracy_score(y_test, pipeline.predict(X_test)))

# Initialize LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=df.columns[:-1],
    class_names=list(map(str, set(y))),
    mode="classification"
)

# Pick one test instance to explain
i = 10  # Example index
exp = explainer.explain_instance(
    X_test[i],
    pipeline.predict_proba,   
    num_features=10
)

print(exp.as_list())

# Visualize explanation
exp.show_in_notebook(show_table=True)
