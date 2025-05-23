from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the Breast Cancer dataset and convert it into a DataFrame
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Explore the dataset
print(df.head())
print(df.info())

from sklearn.model_selection import train_test_split

# Fill missing values (if any)
df.fillna(df.median(), inplace=True)

# Separate features and target label
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)  # Adjust max_iter to ensure convergence
log_reg.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred_log = log_reg.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Accuracy: {accuracy_log * 100:.2f}%")

from sklearn.tree import DecisionTreeClassifier

# Train the Decision Tree model
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred_tree = tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy_tree * 100:.2f}%")


# Train the SVM model
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")

from sklearn.metrics import precision_score, recall_score, f1_score

# Logistic Regression Metrics
precision_log = precision_score(y_test, y_pred_log, average='weighted')
recall_log = recall_score(y_test, y_pred_log, average='weighted')
f1_log = f1_score(y_test, y_pred_log, average='weighted')
print(f"Logistic Regression - Precision: {precision_log:.2f}, Recall: {recall_log:.2f}, F1 Score: {f1_log:.2f}")