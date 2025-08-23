
# LOMAVE (Leaf Oriented Multi-scale Adaptive Vein Enhancement)

## Setup
Install dependencies:
```
pip install numpy opencv-python scikit-image scipy matplotlib networkx
```

## Run
Place leaf images in `data/input/`. Optionally, put ground-truth binary edge maps in `data/ground_truth/` with the same filenames.

```
python -m lomave  # (no entry point; use main.py)
python main.py --input_dir data/input --output_dir outputs/images --ground_truth_dir data/ground_truth
```

## Outputs
- Enhanced map, binary mask, skeleton, weight map
- Baseline edges: Sobel, Prewitt, Canny, LoG
- `outputs/metrics/results.json` with features and metrics (if GT provided)
