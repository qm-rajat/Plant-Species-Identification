
import json, sys
import joblib
from .extract_dataset_features import features_for_image

def predict(image_path, model_path="leafdet/models/leaf_rf.joblib"):
    feats = features_for_image(image_path)
    keys = sorted(feats.keys())
    X = [[feats[k] for k in keys]]
    model = joblib.load(model_path)
    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
    yhat = model.predict(X)[0]
    return {"label": int(yhat), "prob_leaf": float(prob) if prob is not None else None, "features": {k:feats[k] for k in keys}}

if __name__ == "__main__":
    img_path = sys.argv[1]
    out = predict(img_path)
    print(json.dumps(out, indent=2))
