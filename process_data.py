import json
import os
import random
import shutil
from pathlib import Path

def main():
    # Define paths
    base_dir = Path("c:/Users/aalam23/Documents/My Code/VLM")
    train_dir = base_dir / "archive (1)" / "train"
    annotations_path = train_dir / "annotations" / "bdd100k_labels_images_train.json"
    images_dir = train_dir / "images"
    
    output_dir = base_dir / "bdd_subset"
    output_images_dir = output_dir / "images"
    output_labels_path = output_dir / "labels.json"

    # 1. Load the training annotation JSON
    print(f"Loading annotations from {annotations_path}...")
    if not annotations_path.exists():
        print(f"Error: Annotation file not found at {annotations_path}")
        return

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"Loaded {len(annotations)} annotations.")

    # 2. Verify valid images exist in the train image directory
    print(f"Verifying images in {images_dir}...")
    if not images_dir.exists():
        print(f"Error: Image directory not found at {images_dir}")
        return

    valid_annotations = []
    # Check a few to ensure paths are correct, or check all if feasible. 
    # Checking all might be slow if there are 100k images. 
    # Let's filter the list to only those where images exist.
    
    # Optimization: Just check existence during sampling or filter first?
    # Let's filter first to ensure we sample from valid ones.
    # To save time, we can just check if the image file exists for the sampled ones, 
    # and if not, resample. But filtering is safer.
    # However, checking 70k files might take a moment. 
    # Let's try to sample first, then verify, and resample if needed.
    
    # Actually, the requirement says "Verify valid images exist".
    # I'll check if the directory is not empty.
    if not any(images_dir.iterdir()):
         print("Error: Image directory is empty.")
         return

    # 3. Randomly sample 200 entries
    print("Sampling 200 entries...")
    # We need to make sure we get 200 valid ones.
    sampled_annotations = []
    
    # Shuffle the annotations to pick random ones
    random.shuffle(annotations)
    
    count = 0
    for ann in annotations:
        image_name = ann['name']
        image_path = images_dir / image_name
        
        if image_path.exists():
            sampled_annotations.append(ann)
            count += 1
        
        if count >= 200:
            break
            
    if len(sampled_annotations) < 200:
        print(f"Warning: Only found {len(sampled_annotations)} valid images.")
    
    # 4. Copy those images to a new bdd_subset/images folder
    print(f"Copying images to {output_images_dir}...")
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    for ann in sampled_annotations:
        image_name = ann['name']
        src_path = images_dir / image_name
        dst_path = output_images_dir / image_name
        shutil.copy2(src_path, dst_path)
        
    # 5. Save the corresponding annotations to bdd_subset/labels.json
    print(f"Saving annotations to {output_labels_path}...")
    with open(output_labels_path, 'w') as f:
        json.dump(sampled_annotations, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
