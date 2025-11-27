import json
import os
from pathlib import Path

def verify():
    base_dir = Path("c:/Users/aalam23/Documents/My Code/VLM")
    subset_dir = base_dir / "bdd_subset"
    images_dir = subset_dir / "images"
    labels_path = subset_dir / "labels.json"

    print("Verifying subset...")
    
    # Verify images
    if not images_dir.exists():
        print("Error: Images directory missing.")
        return
    
    image_count = len(list(images_dir.glob("*")))
    print(f"Images count: {image_count}")
    
    if image_count != 200:
        print("Error: Expected 200 images.")
    else:
        print("Images count is correct.")

    # Verify labels
    if not labels_path.exists():
        print("Error: Labels file missing.")
        return
        
    with open(labels_path, 'r') as f:
        labels = json.load(f)
        
    label_count = len(labels)
    print(f"Labels count: {label_count}")
    
    if label_count != 200:
        print("Error: Expected 200 labels.")
    else:
        print("Labels count is correct.")
        
    # Verify correspondence
    print("Verifying correspondence...")
    image_names = {p.name for p in images_dir.glob("*")}
    label_names = {l['name'] for l in labels}
    
    if image_names == label_names:
        print("All labels match image filenames.")
    else:
        print("Mismatch between labels and images.")
        missing_images = label_names - image_names
        missing_labels = image_names - label_names
        if missing_images:
            print(f"Missing images for labels: {len(missing_images)}")
        if missing_labels:
            print(f"Missing labels for images: {len(missing_labels)}")

if __name__ == "__main__":
    verify()
