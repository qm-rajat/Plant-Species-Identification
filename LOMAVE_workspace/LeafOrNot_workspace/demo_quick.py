import json, sys, os
import cv2
from leafdet.quick_rule import quick_leaf_rule

def run(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    label, score, diag = quick_leaf_rule(img)
    return {"input": os.path.basename(image_path), "label": label, "score": score, "diagnostics": diag}, img

if __name__ == "__main__":
    img_path = r"data\input\3a211a87-cac9-4393-84ca-51e240e1b043___RS_HL 0210.JPG"  # Use your image path here
    result, img = run(img_path)
    print(json.dumps(result, indent=2))
    cv2.imshow(f"Result: {result['label']} (Score: {result['score']})", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
