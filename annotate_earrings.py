"""
Earring-Specific Annotation Tool
Detects and annotates earrings in all images with bounding boxes.
Specifically designed for earring product images.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
from typing import List, Dict, Tuple, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class ObjectDetector:
    """General object detector using YOLO"""
    
    def __init__(self, method='smart', conf_threshold=0.25):
        """
        Initialize earring detector
        
        Args:
            method: Detection method ('yolo', 'opencv', 'smart')
            conf_threshold: Confidence threshold for YOLO
        """
        self.method = method
        self.conf_threshold = conf_threshold
        self.yolo_model = None
        
        if method in ['yolo', 'smart'] and YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                print("YOLO model loaded for general object detection")
            except:
                print("Warning: YOLO model not available, using OpenCV")
                self.method = 'opencv'
    
    def detect_objects_in_image(self, image_path: str) -> List[Dict]:
        """
        Detect objects in a single image

        Returns list of detections with format:
        [{
            'bbox': [x, y, width, height],
            'confidence': float,
            'class': str,
            'method': str
        }]
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        height, width = image.shape[:2]
        detections = []
        
        if self.method == 'yolo' and self.yolo_model:
            # Try YOLO first
            detections = self._detect_with_yolo(image, image_path)
        
        if self.method == 'smart':
            # Smart detection: try YOLO first, fallback to OpenCV
            if self.yolo_model:
                detections = self._detect_with_yolo(image, image_path)
            
            # If no detections or confidence too low, use OpenCV
            if not detections or (detections and detections[0].get('confidence', 0) < 0.3):
                cv_detections = self._detect_earring_opencv(image)
                if cv_detections:
                    # Prefer OpenCV detection if YOLO failed
                    detections = cv_detections
        
        elif self.method == 'opencv' or not detections:
            # Use OpenCV-based detection
            detections = self._detect_earring_opencv(image)
        
        # Ensure at least one detection (main subject fallback)
        if not detections:
            # Fallback: annotate the center region of the image
            # For product images, earring is typically centered
            center_x, center_y = width // 2, height // 2
            bbox_size = min(width, height) * 0.6  # 60% of smaller dimension
            
            detections = [{
                'bbox': [
                    center_x - bbox_size // 2,
                    center_y - bbox_size // 2,
                    bbox_size,
                    bbox_size
                ],
                'confidence': 1.0,
                'class': 'earring',
                'method': 'center_fallback'
            }]
        
        # Ensure bbox doesn't exceed image bounds
        for det in detections:
            bbox = det['bbox']
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(width - bbox[0], bbox[2])
            bbox[3] = min(height - bbox[1], bbox[3])
        
        return detections
    
    def _detect_with_yolo(self, image, image_path: str) -> List[Dict]:
        """Detect using YOLO model"""
        detections = []
        
        try:
            results = self.yolo_model(str(image_path), conf=self.conf_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = result.names[cls_id]
                    
                    # Accept jewelry-related classes or general objects
                    # YOLO doesn't have "earring" class, so we accept any detection
                    # and classify as "earring" since these are earring product images
                    bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    
                    detections.append({
                        'bbox': bbox,
                        'confidence': conf,
                        'class': cls_name,
                        'method': 'yolo'
                    })
        except Exception as e:
            print(f"YOLO detection error: {e}")
        
        return detections
    
    def _detect_earring_opencv(self, image) -> List[Dict]:
        """
        Detect ALL earrings using OpenCV-based methods
        Designed for product images with multiple earrings (different colors)
        """
        height, width = image.shape[:2]
        image_area = width * height
        detections = []
        
        # Method 1: Color-based segmentation to detect different colored earrings
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for different color ranges (jewelry colors)
        color_ranges = [
            # Gold/Yellow tones
            (np.array([15, 50, 50]), np.array([30, 255, 255])),
            # Silver/Gray tones  
            (np.array([0, 0, 100]), np.array([180, 50, 255])),
            # Blue tones (blue stones)
            (np.array([100, 50, 50]), np.array([130, 255, 255])),
            # Red/Pink tones (red/pink stones)
            (np.array([160, 50, 50]), np.array([180, 255, 255])),
            # Green tones (green stones)
            (np.array([40, 50, 50]), np.array([80, 255, 255])),
        ]
        
        all_contours = []
        
        # Try color-based detection
        for color_range in color_ranges:
            mask = cv2.inRange(hsv, color_range[0], color_range[1])
            
            # Remove noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        
        # Method 2: Edge-based detection (for non-color-based earrings)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Combine edge methods
        combined = cv2.bitwise_or(adaptive_thresh, edges)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        edge_contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(edge_contours)
        
        # Method 3: Background subtraction (earrings vs background)
        # Use threshold to separate foreground objects
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(thresh_contours)
        
        # Filter and process all contours
        valid_detections = []
        min_area = image_area * 0.02  # Minimum 2% of image
        max_area = image_area * 0.7   # Maximum 70% of image
        
        for contour in all_contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter by reasonable aspect ratio for earrings
                if 0.25 < aspect_ratio < 4.0:
                    # Calculate confidence based on area
                    confidence = min(0.9, (area / image_area) * 3)
                    
                    # Check for overlap with existing detections (non-maximum suppression)
                    overlap = False
                    for existing in valid_detections:
                        existing_bbox = existing['bbox']
                        iou = self._calculate_iou([x, y, w, h], existing_bbox)
                        if iou > 0.3:  # If overlap > 30%, skip
                            overlap = True
                            break
                    
                    if not overlap:
                        valid_detections.append({
                            'bbox': [float(x), float(y), float(w), float(h)],
                            'confidence': confidence,
                            'class': 'earring',
                            'method': 'opencv_multi',
                            'area': area
                        })
        
        # Sort by confidence/area
        valid_detections.sort(key=lambda x: x.get('area', 0), reverse=True)
        
        if valid_detections:
            # Return top detections (up to 8 earrings per image)
            return valid_detections[:8]
        
        # Fallback: If no detections, use grid-based approach for multiple earrings
        # Check if image might have multiple earrings in grid layout
        # Split image into 4 quadrants and try to detect in each
        detections_fallback = []
        for i in range(2):
            for j in range(2):
                x_start = i * width // 2
                y_start = j * height // 2
                x_end = (i + 1) * width // 2
                y_end = (j + 1) * height // 2
                
                roi = image[y_start:y_end, x_start:x_end]
                if roi.size > 0:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in roi_contours:
                        area = cv2.contourArea(contour)
                        roi_area = (x_end - x_start) * (y_end - y_start)
                        
                        if area > roi_area * 0.1:  # At least 10% of quadrant
                            x, y, w, h = cv2.boundingRect(contour)
                            # Adjust coordinates to full image
                            x += x_start
                            y += y_start
                            
                            detections_fallback.append({
                                'bbox': [float(x), float(y), float(w), float(h)],
                                'confidence': 0.6,
                                'class': 'earring',
                                'method': 'opencv_grid'
                            })
        
        if detections_fallback:
            return detections_fallback[:4]  # Max 4 from grid
        
        # Final fallback: center-based single detection
        center_x, center_y = width // 2, height // 2
        bbox_size = min(width, height) * 0.4
        
        return [{
            'bbox': [
                center_x - bbox_size // 2,
                center_y - bbox_size // 2,
                bbox_size,
                bbox_size
            ],
            'confidence': 0.5,
            'class': 'earring',
            'method': 'opencv_center'
        }]
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def draw_annotation_on_image(self, image_path: str, detections: List[Dict],
                                 output_path: Optional[str] = None) -> Image.Image:
        """Draw bounding boxes on image with different colors for multiple earrings"""
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Different colors for multiple earrings
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 165, 0),    # Orange
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 255, 0),    # Yellow
            (128, 0, 128)     # Purple
        ]
        
        for idx, det in enumerate(detections):
            bbox = det['bbox']
            x, y, w, h = bbox
            conf = det.get('confidence', 0.0)
            
            # Use different color for each earring
            color = colors[idx % len(colors)]
            
            # Draw rectangle with thicker line
            draw.rectangle([x, y, x + w, y + h], outline=color, width=4)
            
            # Draw label with class name
            label = f"{det.get('class', 'unknown')} ({conf:.2f})"
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Get text size
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw background rectangle for text
            draw.rectangle(
                [x, y - text_height - 6, x + text_width + 8, y],
                fill=color
            )
            draw.text(
                (x + 4, y - text_height - 3),
                label,
                fill=(255, 255, 255),
                font=font
            )
        
        if output_path:
            img.save(output_path)
        
        return img


def annotate_all_earrings(images_dir="images",
                          output_annotated_dir="annotated_images",
                          method='smart',
                          conf_threshold=0.25):
    """
    Annotate jewelry objects in ALL images directly from image files

    Args:
        images_dir: Directory with image subfolders
        output_annotated_dir: Directory to save annotated images
        method: Detection method ('yolo', 'opencv', 'smart')
        conf_threshold: Confidence threshold for YOLO
    """
    images_dir = Path(images_dir)
    output_annotated_dir = Path(output_annotated_dir)
    output_annotated_dir.mkdir(exist_ok=True)

    # Initialize detector
    detector = ObjectDetector(method=method, conf_threshold=conf_threshold)

    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.rglob(ext))

    if not image_files:
        print(f"No image files found in {images_dir}")
        return

    # Process annotations
    annotations = []
    images_data = []
    annotation_id = 1
    image_id = 1
    processed = 0
    skipped = 0

    print("=" * 80)
    print("Jewelry Annotation Tool - Processing ALL Images")
    print("=" * 80)
    print(f"Method: {method}, Confidence threshold: {conf_threshold}")
    print(f"Total images found: {len(image_files)}\n")

    # Process each image
    for image_path in image_files:
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            skipped += 1
            continue

        # Get relative path for folder name
        try:
            relative_path = image_path.relative_to(images_dir)
            folder_name = str(relative_path.parent) if relative_path.parent != Path('.') else ''
        except ValueError:
            folder_name = ''

        # Detect objects
        detections = detector.detect_objects_in_image(str(image_path))

        # Draw annotations on image
        output_filename = f"{folder_name}_{image_path.name}" if folder_name else image_path.name
        output_image_path = output_annotated_dir / output_filename
        detector.draw_annotation_on_image(
            str(image_path),
            detections,
            str(output_image_path)
        )

        # Add image to COCO format
        img_info = {
            "id": image_id,
            "file_name": str(image_path.name),
            "folder_name": folder_name,
            "width": 0,  # Will be updated if we can read image
            "height": 0
        }

        # Try to get image dimensions
        try:
            with Image.open(image_path) as img:
                img_info["width"] = img.width
                img_info["height"] = img.height
        except:
            pass

        images_data.append(img_info)

        # Add annotations to COCO format
        for det in detections:
            bbox = det['bbox']
            x, y, w, h = bbox

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # jewelry category
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "confidence": det.get('confidence', 0.0),
                "class": det.get('class', 'unknown'),
                "method": det.get('method', 'unknown')
            }
            annotations.append(annotation)
            annotation_id += 1

        processed += 1
        image_id += 1

        if processed % 50 == 0:
            print(f"Processed: {processed}/{len(image_files)} images...")

    # Create COCO data structure
    coco_data = {
        "images": images_data,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "jewelry",
                "supercategory": "object"
            }
        ]
    }

    # Save annotations
    output_annotations = f"annotations_jewelry_{method}.json"
    with open(output_annotations, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Annotation Complete!")
    print(f"  - Processed: {processed} images")
    print(f"  - Skipped: {skipped} images")
    print(f"  - Total annotations: {len(annotations)} objects")
    print(f"  - Annotated images saved to: {output_annotated_dir}")
    print(f"  - Annotations saved to: {output_annotations}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Annotate Jewelry in All Images')
    parser.add_argument('--images-dir', default='images',
                       help='Directory containing image subfolders')
    parser.add_argument('--output-dir', default='annotated_images',
                       help='Directory to save annotated images')
    parser.add_argument('--method', default='smart',
                       choices=['yolo', 'opencv', 'smart'],
                       help='Detection method (smart tries YOLO then OpenCV)')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold for YOLO')

    args = parser.parse_args()

    annotate_all_earrings(
        images_dir=args.images_dir,
        output_annotated_dir=args.output_dir,
        method=args.method,
        conf_threshold=args.conf_threshold
    )

