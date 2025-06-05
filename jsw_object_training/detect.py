from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def detect_cement_bags(image_path, output_dir='detection_results'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trained model
    model = YOLO('best.pt')
    
    # Process the image
    results = model(image_path)
    
    # Get the first result (since we're processing one image)
    result = results[0]
    
    # Get the annotated image
    annotated_img = result.plot()
    
    # Save the result
    output_path = os.path.join(output_dir, f'detected_{Path(image_path).name}')
    cv2.imwrite(output_path, annotated_img)
    
    # Print detection information
    print(f"\nDetection Results for {image_path}:")
    print(f"Number of cement bags detected: {len(result.boxes)}")
    for i, box in enumerate(result.boxes):
        conf = float(box.conf)
        print(f"Bag {i+1}: Confidence = {conf:.2f}")
    
    print(f"\nResult saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage
    image_path = input("Enter the path to the image you want to analyze: ")
    if os.path.exists(image_path):
        detect_cement_bags(image_path)
    else:
        print(f"Error: Image file not found at {image_path}") 