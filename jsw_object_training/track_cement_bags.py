from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tracking.log'),
        logging.StreamHandler()
    ]
)

class CementBagTracker:
    def __init__(self, model_path="best_cement_bags.pt", conf_threshold=0.5, iou_threshold=0.5):
        """Initialize the cement bag tracker."""
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logging.info(f"Loaded model from {model_path}")
        
    def process_frame(self, frame):
        """Process a single frame and return detections."""
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=True,
            verbose=False
        )
        
        return results[0] if results else None
    
    def draw_detections(self, frame, results):
        """Draw detections on the frame."""
        if results is None:
            return frame
            
        annotated_frame = results.plot()
        return annotated_frame
    
    def process_video(self, video_path, output_path=None):
        """Process a video file and save the results."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            results = self.process_frame(frame)
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, results)
            
            # Write frame if output is specified
            if writer:
                writer.write(annotated_frame)
                
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                logging.info(f"Processed {frame_count} frames")
                
        # Cleanup
        cap.release()
        if writer:
            writer.release()
            
        logging.info(f"Finished processing video. Total frames: {frame_count}")
        
    def process_image(self, image_path, output_path=None):
        """Process a single image and save the result."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Process frame
        results = self.process_frame(frame)
        
        # Draw detections
        annotated_frame = self.draw_detections(frame, results)
        
        # Save result if output path is provided
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            logging.info(f"Saved result to {output_path}")
            
        return annotated_frame

def main():
    # Example usage
    tracker = CementBagTracker()
    
    # Process a video
    video_path = "path/to/your/video.mp4"
    output_path = f"tracked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    tracker.process_video(video_path, output_path)
    
    # Process an image
    image_path = "path/to/your/image.jpg"
    output_path = f"tracked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    tracker.process_image(image_path, output_path)

if __name__ == "__main__":
    main() 