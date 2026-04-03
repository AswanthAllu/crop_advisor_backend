import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

filename = "crop_recommandation_dataset.csv"
df = pd.read_csv(filename)
print(f"Data loaded: {df.shape[0]} rows.")

feature_cols = ['temperature', 'humidity', 'ph', 'rainfall']
target_col = 'label'

X_raw = df[feature_cols]
y = df[target_col]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_raw)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
# main algo cascaded
class RF_DT_SVM_Cascade:
    def __init__(self):
        self.stage1 = RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=42, n_jobs=-1, class_weight='balanced'
        )
        self.stage2 = DecisionTreeClassifier(
            max_depth=None, min_samples_split=5, min_samples_leaf=2,
            random_state=42, class_weight='balanced'
        )
        self.stage3 = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, C=1000, gamma='scale', random_state=42))
        ])

    def fit(self, X, y):
        self.stage1.fit(X, y)
        s1_probs = self.stage1.predict_proba(X)
        X_s2 = np.column_stack((X, s1_probs))
        
        self.stage2.fit(X_s2, y)
        s2_probs = self.stage2.predict_proba(X_s2)
        X_s3 = np.column_stack((X, s2_probs))
        
        self.stage3.fit(X_s3, y)

    def predict(self, X):
        s1_probs = self.stage1.predict_proba(X)
        X_s2 = np.column_stack((X, s1_probs))
        s2_probs = self.stage2.predict_proba(X_s2)
        X_s3 = np.column_stack((X, s2_probs))
        return self.stage3.predict(X_s3)

cascade = RF_DT_SVM_Cascade()
cascade.fit(X_train, y_train)

y_pred_encoded = cascade.predict(X_test)
y_test_names = le.inverse_transform(y_test)
y_pred_names = le.inverse_transform(y_pred_encoded)

acc = accuracy_score(y_test, y_pred_encoded)
print("-" * 60)
print(f"Final Cascade Model Accuracy: {acc*100:.2f}%")

precisions = precision_score(y_test, y_pred_encoded, average=None)
recalls = recall_score(y_test, y_pred_encoded, average=None)
f1s = f1_score(y_test, y_pred_encoded, average=None)
classes = le.classes_

print("\nClass-wise Metrics:")
for i, cls in enumerate(classes):
    print(f"{cls}: Precision={precisions[i]:.3f}, Recall={recalls[i]:.3f}, F1={f1s[i]:.3f}")

prec_macro = precision_score(y_test, y_pred_encoded, average='macro')
rec_macro = recall_score(y_test, y_pred_encoded, average='macro')
f1_macro = f1_score(y_test, y_pred_encoded, average='macro')
print("\nMacro Average Metrics:")
print(f"Precision={prec_macro:.3f}, Recall={rec_macro:.3f}, F1={f1_macro:.3f}")

cm = confusion_matrix(y_test, y_pred_encoded)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test_names, y_pred_names))

artifacts = {
    "model": cascade,
    "encoder": le,
    "poly": poly
}
joblib.dump(artifacts, "crop_model.pkl")
print("Model saved as 'crop_model.pkl'")

def predict_custom(temp, humid, ph, rain):
    raw_data = pd.DataFrame([[temp, humid, ph, rain]], columns=feature_cols)
    poly_data = poly.transform(raw_data)
    pred_code = cascade.predict(poly_data)[0]
    return le.inverse_transform([pred_code])[0]

print(f"Test Prediction: {predict_custom(26.5, 81.4, 6.2, 264.6)}")