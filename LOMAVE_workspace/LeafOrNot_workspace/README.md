
# LeafOrNot — Leaf vs Non-Leaf Detector

Two paths:
1) **Quick Heuristic** (no training): color + shape rules for fast screening.
2) **Trainable ML** (scikit-learn): hand-crafted features (color/shape/texture) → RandomForest.

## Install
```
pip install numpy opencv-python scikit-image scikit-learn joblib matplotlib
```

## Quick Heuristic (no training)
```
python demo_quick.py /path/to/image.jpg
# -> {"input": "...", "label": "leaf"|"non-leaf", "score": 0.0-1.0, diagnostics:{...}}
```

## Train a model
Put images into:
```
data/leaf/
data/non_leaf/
```
Then:
```
python -m leafdet.train
```

## Predict with trained model
```
python -m leafdet.predict /path/to/image.jpg
```

### Features used (ML path)
- **Shape:** area_ratio, solidity, aspect_ratio, circularity
- **Color:** HSV (H,S,V) means/stds, LAB (a*, b*)
- **Texture:** LBP histogram, GLCM stats

## Notes
- Heuristic thresholds tuned for typical green leaves on simple backgrounds. Adjust in `leafdet/quick_rule.py`.
- For robust performance across domains, prefer the ML classifier (provide a few hundred samples per class).
