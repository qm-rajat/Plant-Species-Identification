import os, json
from .extract_dataset_features import build_matrix_from_folders
from .classifier import fit_and_save

def train(
    leaf_dir="C:/Users/rajat/Downloads/LOMAVE_workspace/data/leaf",
    nonleaf_dir="C:/Users/rajat/Downloads/LOMAVE_workspace/data/non_leaf",
    model_out="leafdet/models/leaf_rf.joblib"
):
    X, y, files, feat_names = build_matrix_from_folders(leaf_dir, nonleaf_dir)
    if len(X) == 0:
        raise RuntimeError("No training data found.")
    model, report = fit_and_save(X, y, save_path=model_out)
    return {"report": report, "n_samples": int(len(y)), "features": feat_names}

if __name__ == "__main__":
    info = train()
    print(json.dumps(info, indent=2))
