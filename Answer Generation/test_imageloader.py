import os
import json
from PIL import Image

def test_image_loading(input_file, image_dir):
    """Test if images can be loaded from the specified directory"""
    print(f"Testing image loading from {image_dir}")
    
    # Load JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Count variables
    total_instances = len(data)
    instances_with_positive_facts = 0
    images_found = 0
    images_not_found = 0
    
    # Track the first few not found for debugging
    missing_images = []
    
    for item in data:
        instance = item.get("data", item)
        
        if "img_posFacts" in instance and len(instance["img_posFacts"]) > 0:
            instances_with_positive_facts += 1
            
            for pos_fact in instance["img_posFacts"]:
                image_id = pos_fact.get("image_id")
                if image_id:
                    image_path = os.path.join(image_dir, f"{image_id}.png")
                    
                    if os.path.exists(image_path):
                        # Try to open the image to verify it's valid
                        try:
                            img = Image.open(image_path)
                            img.verify()  # Verify it's a valid image file
                            images_found += 1
                        except Exception as e:
                            print(f"Error with image {image_path}: {e}")
                            images_not_found += 1
                            if len(missing_images) < 5:
                                missing_images.append((image_id, str(e)))
                    else:
                        images_not_found += 1
                        if len(missing_images) < 5:
                            missing_images.append((image_id, "File not found"))
    
    # Print results
    print(f"Total instances: {total_instances}")
    print(f"Instances with positive facts: {instances_with_positive_facts}")
    print(f"Images found: {images_found}")
    print(f"Images not found: {images_not_found}")
    
    if missing_images:
        print("\nSample of missing images:")
        for img_id, error in missing_images:
            print(f"  - Image ID {img_id}: {error}")
    
    # Check alternative file extensions if PNG not found
    if images_not_found > 0:
        print("\nChecking for alternative extensions...")
        extensions = ['.jpg', '.jpeg', '.webp']
        
        for ext in extensions:
            found_with_ext = 0
            
            for img_id, _ in missing_images:
                alt_path = os.path.join(image_dir, f"{img_id}{ext}")
                if os.path.exists(alt_path):
                    found_with_ext += 1
            
            if found_with_ext > 0:
                print(f"Found {found_with_ext} images with extension: {ext}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python test_images.py <input_file> <image_dir>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    image_dir = sys.argv[2]
    
    test_image_loading(input_file, image_dir)