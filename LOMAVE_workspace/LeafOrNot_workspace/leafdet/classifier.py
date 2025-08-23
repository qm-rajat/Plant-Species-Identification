
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def build_model(n_estimators=200, random_state=42):
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("rf", RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight="balanced"))
    ])
    return clf

def fit_and_save(X, y, save_path="leafdet/models/leaf_rf.joblib", test_size=0.2, random_state=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = build_model()
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    report = classification_report(yte, yhat, output_dict=True)
    joblib.dump(model, save_path)
    return model, report
