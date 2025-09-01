# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import os
import sys
import time
import glob
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
import PIL.Image
from dataclasses import dataclass, asdict
from tqdm import tqdm


from nanoowl.owl_predictor import OwlPredictor, OwlDecodeOutput
from nanoowl.owl_drawing import draw_owl_output


@dataclass
class DetectionMetrics:
    """Container for object detection metrics"""
    precision: float
    recall: float
    f1_score: float
    map_50: float
    map_50_95: float
    avg_precision_per_class: Dict[str, float]
    total_predictions: int
    total_ground_truth: int
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_time: float
    avg_time_per_image: float
    fps: float
    preprocessing_time: float
    inference_time: float
    postprocessing_time: float
    memory_usage_mb: float


@dataclass
class BenchmarkResult:
    """Container for complete benchmark results"""
    detection_metrics: DetectionMetrics
    performance_metrics: PerformanceMetrics
    config: Dict[str, Any]


class GroundTruthAnnotation:
    """Ground truth annotation for an image"""
    def __init__(self, image_path: str, boxes: List[List[float]], labels: List[str], scores: Optional[List[float]] = None):
        self.image_path = image_path
        self.boxes = boxes  # List of [x1, y1, x2, y2] in pixel coordinates
        self.labels = labels
        self.scores = scores or [1.0] * len(boxes)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """Calculate Average Precision (AP) using the 11-point interpolation method"""
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]
    
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


class ObjectDetectionEvaluator:
    """Evaluator for object detection metrics"""
    
    def __init__(self, iou_threshold: float = 0.5):  # Standard IoU threshold for object detection
        self.iou_threshold = iou_threshold
        
    def evaluate_predictions(self, 
                           predictions: List[OwlDecodeOutput], 
                           ground_truths: List[GroundTruthAnnotation],
                           class_names: List[str],
                           image_sizes: List[Tuple[int, int]]) -> DetectionMetrics:
        """Evaluate predictions against ground truth annotations"""
        
        all_pred_boxes = []
        all_pred_labels = []
        all_pred_scores = []
        all_gt_boxes = []
        all_gt_labels = []
        
        # Collect all predictions and ground truths
        for i, (pred, gt, img_size) in enumerate(zip(predictions, ground_truths, image_sizes)):
            if pred is not None:
                # Get boxes - they are already in pixel coordinates
                boxes = pred.boxes.detach().cpu().numpy()
                
                # Ensure boxes are within image bounds and handle negative coordinates
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_size[0])
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_size[1])
                
                # Map model predictions to ground truth format
                pred_labels = [class_names[label_idx] for label_idx in pred.labels.detach().cpu().numpy()]
                # Remove articles for matching with ground truth
                labels = []
                for label in pred_labels:
                    if label.startswith('a '):
                        labels.append(label[2:])  # Remove "a "
                    elif label.startswith('an '):
                        labels.append(label[3:])  # Remove "an "
                    else:
                        labels.append(label)
                scores = pred.scores.detach().cpu().numpy()
                
                # Filter out boxes that are too small or have negative coordinates
                valid_boxes = []
                valid_labels = []
                valid_scores = []
                
                for i, box in enumerate(boxes):
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    # More lenient filtering - only remove obviously invalid boxes
                    if width > 1 and height > 1 and box[0] >= 0 and box[1] >= 0:
                        valid_boxes.append(box)
                        valid_labels.append(labels[i])
                        valid_scores.append(scores[i])
                
                if valid_boxes:
                    all_pred_boxes.extend(valid_boxes)
                    all_pred_labels.extend(valid_labels)
                    all_pred_scores.extend(valid_scores)
            
            # Ground truth
            all_gt_boxes.extend(gt.boxes)
            all_gt_labels.extend(gt.labels)
        
        # Calculate metrics
        return self._calculate_metrics(
            all_pred_boxes, all_pred_labels, all_pred_scores,
            all_gt_boxes, all_gt_labels, class_names
        )
    
    def _calculate_metrics(self, 
                          pred_boxes: List[List[float]], 
                          pred_labels: List[str], 
                          pred_scores: List[float],
                          gt_boxes: List[List[float]], 
                          gt_labels: List[str],
                          class_names: List[str]) -> DetectionMetrics:
        """Calculate detection metrics"""
        
        # Initialize counters
        tp = fp = fn = 0
        class_aps = {}
        
        # For each class, calculate AP
        for class_name in class_names:
            # Get predictions and ground truths for this class
            class_pred_boxes = [box for box, label in zip(pred_boxes, pred_labels) if label == class_name]
            class_pred_scores = [score for score, label in zip(pred_scores, pred_labels) if label == class_name]
            class_gt_boxes = [box for box, label in zip(gt_boxes, gt_labels) if label == class_name]
            
            if len(class_gt_boxes) == 0:
                class_aps[class_name] = 0.0
                continue
            
            if len(class_pred_boxes) == 0:
                class_aps[class_name] = 0.0
                fn += len(class_gt_boxes)
                continue
            
            # Sort predictions by confidence
            sorted_indices = np.argsort(class_pred_scores)[::-1]
            class_pred_boxes = [class_pred_boxes[i] for i in sorted_indices]
            class_pred_scores = [class_pred_scores[i] for i in sorted_indices]
            
            # Match predictions to ground truths
            gt_matched = [False] * len(class_gt_boxes)
            precisions = []
            recalls = []
            class_tp = class_fp = 0
            
            for pred_box in class_pred_boxes:
                # Find best matching ground truth
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(class_gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Check if match is good enough
                if best_iou >= self.iou_threshold:
                    class_tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    class_fp += 1
                
                # Calculate precision and recall
                precision = class_tp / (class_tp + class_fp)
                recall = class_tp / len(class_gt_boxes)
                precisions.append(precision)
                recalls.append(recall)
            
            tp += class_tp
            fp += class_fp
            fn += len(class_gt_boxes) - class_tp
            
            # Calculate AP for this class
            if len(precisions) > 0:
                class_aps[class_name] = calculate_ap(np.array(precisions), np.array(recalls))
            else:
                class_aps[class_name] = 0.0
        
        # Overall metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        map_50 = np.mean(list(class_aps.values())) if class_aps else 0.0
        
        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            map_50=map_50,
            map_50_95=map_50,  # Simplified - would need multiple IoU thresholds for true mAP@0.5:0.95
            avg_precision_per_class=class_aps,
            total_predictions=len(pred_boxes),
            total_ground_truth=len(gt_boxes),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn
        )


class BatchBenchmark:
    """Main class for batch processing and benchmarking"""
    
    def __init__(self, 
                 model_name: str = "google/owlvit-base-patch32",
                 image_encoder_engine: Optional[str] = None,
                 device: str = "cuda"):
        
        self.predictor = OwlPredictor(
            model_name=model_name,
            image_encoder_engine=image_encoder_engine,
            device=device
        )
        self.device = device
        self.evaluator = ObjectDetectionEvaluator()
    
    def load_dataset(self, dataset_path: str) -> Tuple[List[str], List[GroundTruthAnnotation]]:
        """Load dataset from directory or annotation file"""
        
        dataset_path = Path(dataset_path)
        
        if dataset_path.is_dir():
            # Load images from directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_paths = []
            
            for ext in image_extensions:
                image_paths.extend(list(dataset_path.glob(f"*{ext}")))
                image_paths.extend(list(dataset_path.glob(f"*{ext.upper()}")))
            
            # Create dummy ground truth (for performance testing only)
            ground_truths = []
            for img_path in image_paths:
                ground_truths.append(GroundTruthAnnotation(
                    image_path=str(img_path),
                    boxes=[],  # Empty for performance testing
                    labels=[]
                ))
            
            return [str(p) for p in image_paths], ground_truths
        
        elif dataset_path.suffix == '.json':
            # Load from JSON annotation file
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            image_paths = []
            ground_truths = []
            
            for item in data:
                image_paths.append(item['image_path'])
                ground_truths.append(GroundTruthAnnotation(
                    image_path=item['image_path'],
                    boxes=item.get('boxes', []),
                    labels=item.get('labels', []),
                    scores=item.get('scores', [])
                ))
            
            return image_paths, ground_truths
        
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    def run_benchmark(self, 
                     dataset_path: str,
                     prompts: List[str],
                     threshold: float = 0.5,
                     output_dir: Optional[str] = None,
                     save_visualizations: bool = False,
                     max_images: Optional[int] = None,
                     warmup_runs: int = 5) -> BenchmarkResult:
        """Run complete benchmark on dataset"""
        
        print(f"Loading dataset from {dataset_path}...")
        image_paths, ground_truths = self.load_dataset(dataset_path)
        
        print(f"Found {len(image_paths)} images in dataset")
        
        if max_images:
            if max_images < len(image_paths):
                print(f"Limiting to {max_images} images")
                image_paths = image_paths[:max_images]
                ground_truths = ground_truths[:max_images]
            else:
                print(f"Dataset has {len(image_paths)} images, which is less than requested {max_images}")
        
        print(f"Loaded {len(image_paths)} images")
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Encode text once
        print("Encoding text prompts...")
        text_encodings = self.predictor.encode_text(prompts)
        
        # Warmup runs
        if warmup_runs > 0:
            print(f"Running {warmup_runs} warmup iterations...")
            sample_image = PIL.Image.open(image_paths[0])
            for _ in range(warmup_runs):
                _ = self.predictor.predict(
                    image=sample_image,
                    text=prompts,
                    text_encodings=text_encodings,
                    threshold=threshold
                )
            torch.cuda.synchronize()
        
        # Run benchmark
        print("Running benchmark...")
        predictions = []
        image_sizes = []
        
        # Performance tracking
        total_start_time = time.time()
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        
        # Memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        for i, (image_path, gt) in enumerate(tqdm(zip(image_paths, ground_truths), total=len(image_paths))):
            try:
                # Load image
                prep_start = time.time()
                image = PIL.Image.open(image_path)
                image_sizes.append((image.width, image.height))
                prep_end = time.time()
                preprocessing_times.append(prep_end - prep_start)
                
                # Run prediction
                inf_start = time.time()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
                output = self.predictor.predict(
                    image=image,
                    text=prompts,
                    text_encodings=text_encodings,
                    threshold=threshold
                )
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inf_end = time.time()
                inference_times.append(inf_end - inf_start)
                
                # Postprocessing (visualization if needed)
                post_start = time.time()
                if save_visualizations and output_dir:
                    image_array = np.array(image)
                    image_array = draw_owl_output(image_array, output, text=prompts, draw_text=True)
                    output_image = PIL.Image.fromarray(image_array)
                    output_path = os.path.join(output_dir, f"result_{i:04d}.jpg")
                    output_image.save(output_path)
                
                post_end = time.time()
                postprocessing_times.append(post_end - post_start)
                
                predictions.append(output)
                
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                predictions.append(None)
                image_sizes.append((0, 0))
                preprocessing_times.append(0)
                inference_times.append(0)
                postprocessing_times.append(0)
        
        total_end_time = time.time()
        
        # Calculate performance metrics
        total_time = total_end_time - total_start_time
        avg_time_per_image = total_time / len(image_paths)
        fps = len(image_paths) / total_time
        
        avg_preprocessing_time = np.mean(preprocessing_times)
        avg_inference_time = np.mean(inference_times)
        avg_postprocessing_time = np.mean(postprocessing_times)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage_mb = (peak_memory - initial_memory) / 1024 / 1024
        else:
            memory_usage_mb = 0
        
        performance_metrics = PerformanceMetrics(
            total_time=total_time,
            avg_time_per_image=avg_time_per_image,
            fps=fps,
            preprocessing_time=avg_preprocessing_time,
            inference_time=avg_inference_time,
            postprocessing_time=avg_postprocessing_time,
            memory_usage_mb=memory_usage_mb
        )
        
        # Calculate detection metrics (only if ground truth is available)
        detection_metrics = None
        if any(len(gt.boxes) > 0 for gt in ground_truths):
            print("Calculating detection metrics...")
            
            # Use class names without articles for evaluation
            if args.use_pascal_voc_prompts:
                class_names = get_pascal_voc_class_names()
            else:
                class_names = prompts
            detection_metrics = self.evaluator.evaluate_predictions(
                predictions, ground_truths, class_names, image_sizes
            )
        else:
            print("No ground truth annotations found, skipping detection metrics")
            detection_metrics = DetectionMetrics(
                precision=0.0, recall=0.0, f1_score=0.0, map_50=0.0, map_50_95=0.0,
                avg_precision_per_class={}, total_predictions=0, total_ground_truth=0,
                true_positives=0, false_positives=0, false_negatives=0
            )
        
        # Create config
        config = {
            "model_name": self.predictor.model.name_or_path if hasattr(self.predictor.model, 'name_or_path') else "unknown",
            "prompts": prompts,
            "threshold": threshold,
            "num_images": len(image_paths),
            "device": self.device,
            "image_encoder_engine": "TensorRT" if self.predictor.image_encoder_engine else "PyTorch"
        }
        
        return BenchmarkResult(
            detection_metrics=detection_metrics,
            performance_metrics=performance_metrics,
            config=config
        )
    
    def save_results(self, results: BenchmarkResult, output_path: str):
        """Save benchmark results to JSON file"""
        results_dict = {
            "detection_metrics": asdict(results.detection_metrics),
            "performance_metrics": asdict(results.performance_metrics),
            "config": results.config
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def print_results(self, results: BenchmarkResult):
        """Print benchmark results to console"""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        
        print("\nConfiguration:")
        for key, value in results.config.items():
            print(f"  {key}: {value}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Total time: {results.performance_metrics.total_time:.2f} seconds")
        print(f"  Average time per image: {results.performance_metrics.avg_time_per_image:.3f} seconds")
        print(f"  FPS: {results.performance_metrics.fps:.2f}")
        print(f"  Preprocessing time: {results.performance_metrics.preprocessing_time:.3f} seconds")
        print(f"  Inference time: {results.performance_metrics.inference_time:.3f} seconds")
        print(f"  Postprocessing time: {results.performance_metrics.postprocessing_time:.3f} seconds")
        print(f"  Memory usage: {results.performance_metrics.memory_usage_mb:.1f} MB")
        
        if results.detection_metrics.total_ground_truth > 0:
            print(f"\nDetection Metrics:")
            print(f"  Precision: {results.detection_metrics.precision:.3f}")
            print(f"  Recall: {results.detection_metrics.recall:.3f}")
            print(f"  F1-Score: {results.detection_metrics.f1_score:.3f}")
            print(f"  mAP@0.5: {results.detection_metrics.map_50:.3f}")
            print(f"  True Positives: {results.detection_metrics.true_positives}")
            print(f"  False Positives: {results.detection_metrics.false_positives}")
            print(f"  False Negatives: {results.detection_metrics.false_negatives}")
            
            if results.detection_metrics.avg_precision_per_class:
                print(f"\n  Per-class Average Precision:")
                for class_name, ap in results.detection_metrics.avg_precision_per_class.items():
                    print(f"    {class_name}: {ap:.3f}")


def create_sample_dataset(output_dir: str, num_images: int = 10):
    """Create a sample dataset with dummy annotations for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample annotation file
    annotations = []
    for i in range(num_images):
        # Use existing sample images or create dummy entries
        image_path = f"../assets/owl_glove_small.jpg"  # Use existing sample image
        annotations.append({
            "image_path": image_path,
            "boxes": [[100, 100, 200, 200], [300, 150, 400, 250]],  # Sample bounding boxes
            "labels": ["an owl", "a glove"],  # Sample labels
            "scores": [1.0, 1.0]
        })
    
    # Save annotations
    annotation_path = os.path.join(output_dir, "annotations.json")
    with open(annotation_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Sample dataset created at {output_dir}")
    print(f"Annotations saved to {annotation_path}")
    
    return annotation_path


def extract_pascal_voc_dataset(zip_path: str = "/data/pascal.zip", output_dir: str = "./datasets/pascal_voc_2012", max_images: int = None):
    """Extract and process Pascal VOC 2012 dataset from zip file"""
    try:
        import zipfile
        
        # Check if annotations already exist
        annotations_file = os.path.join(output_dir, "pascal_voc_annotations.json")
        if os.path.exists(annotations_file):
            print(f"Pascal VOC annotations already exist at {annotations_file}")
            return annotations_file
        
        print(f"Extracting Pascal VOC dataset from {zip_path}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        print("Extraction complete. Processing annotations...")
        
        # Look for the VOC directories (this dataset has a different structure)
        voc_dir = None
        for root, dirs, files in os.walk(output_dir):
            if 'VOC2012_train_val' in dirs:
                # Navigate to the nested VOC2012_train_val directory
                nested_dir = os.path.join(root, 'VOC2012_train_val', 'VOC2012_train_val')
                if os.path.exists(nested_dir):
                    voc_dir = nested_dir
                    break
        
        if not voc_dir:
            print("Error: VOC2012_train_val directory not found in extracted files")
            print("Available directories:")
            for root, dirs, files in os.walk(output_dir):
                for dir_name in dirs:
                    print(f"  - {os.path.join(root, dir_name)}")
            return None
        
        annotations_file = os.path.join(output_dir, "pascal_voc_annotations.json")
        num_annotations = create_pascal_voc_annotations(voc_dir, annotations_file, None)  # Process all images
        
        if num_annotations > 0:
            print(f"Successfully processed {num_annotations} annotations")
            return annotations_file
        else:
            print("Error: No annotations were processed")
            return None
            
    except Exception as e:
        print(f"Error extracting Pascal VOC dataset: {e}")
        return None


def create_pascal_voc_annotations(voc_dir: str, output_file: str, max_images: int = None):
    """Create NanoOWL annotations from extracted Pascal VOC dataset"""
    
    # This dataset has a different structure - annotations and images are directly in the voc_dir
    annotations_dir = os.path.join(voc_dir, "Annotations")
    images_dir = os.path.join(voc_dir, "JPEGImages")
    
    if not os.path.exists(annotations_dir):
        print(f"Annotations directory not found: {annotations_dir}")
        return 0
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return 0
    
    # Pascal VOC 20 classes
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # Find all XML annotation files
    xml_files = []
    for ext in ['*.xml']:
        xml_files.extend(glob.glob(os.path.join(annotations_dir, ext)))
    
    if max_images:
        xml_files = xml_files[:max_images]
    
    print(f"Processing {len(xml_files)} annotation files...")
    
    annotations = []
    
    for xml_file in tqdm(xml_files, desc="Processing annotations"):
        try:
            annotation = parse_voc_annotation(xml_file)
            image_path = os.path.join(images_dir, annotation['filename'])
            
            # Check if image exists
            if not os.path.exists(image_path):
                continue
            
            # Convert to NanoOWL format
            boxes = []
            labels = []
            
            for obj in annotation['objects']:
                if obj['name'] in voc_classes:
                    boxes.append(obj['bbox'])
                    labels.append(obj['name'])
            
            if boxes:  # Only include images with valid objects
                annotations.append({
                    'image_path': image_path,
                    'boxes': boxes,
                    'labels': labels,
                    'scores': [1.0] * len(boxes)
                })
        
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue
    
    # Save annotations
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    return len(annotations)


def parse_voc_annotation(xml_path: str):
    """Parse Pascal VOC XML annotation file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image info
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Get objects
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        
        # VOC uses 1-based indexing, convert to 0-based
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def download_pascal_voc_dataset(output_dir: str = "./datasets/pascal_voc_2012", max_images: int = None):
    """Download Pascal VOC 2012 dataset using the download script"""
    try:
        import subprocess
        import sys
        
        print("Downloading Pascal VOC 2012 dataset...")
        cmd = [sys.executable, "download_pascal_voc.py", "--output_dir", output_dir]
        if max_images:
            cmd.extend(["--max_images", str(max_images)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Extract annotations file path from output
            annotations_file = os.path.join(output_dir, "pascal_voc_annotations.json")
            if os.path.exists(annotations_file):
                return annotations_file
            else:
                print("Error: Annotations file not found after download")
                return None
        else:
            print(f"Error downloading dataset: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error downloading Pascal VOC dataset: {e}")
        return None


def get_pascal_voc_prompts():
    """Get Pascal VOC class names as prompts"""
    return [
        'an aeroplane', 'a bicycle', 'a bird', 'a boat', 'a bottle',
        'a bus', 'a car', 'a cat', 'a chair', 'a cow',
        'a diningtable', 'a dog', 'a horse', 'a motorbike', 'a person',
        'a pottedplant', 'a sheep', 'a sofa', 'a train', 'a tvmonitor'
    ]


def get_pascal_voc_class_names():
    """Get Pascal VOC class names without articles (for ground truth matching)"""
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch benchmark for NanoOWL")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Path to dataset directory or annotation JSON file")
    parser.add_argument("--prompts", type=str, default="[an owl, a glove]",
                       help="Detection prompts (comma-separated)")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Detection threshold")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32",
                       help="Model name")
    parser.add_argument("--image_encoder_engine", type=str, default=None,
                       help="Path to TensorRT engine file")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--save_visualizations", action="store_true",
                       help="Save visualization images")
    parser.add_argument("--max_images", type=int, default=None,
                       help="Maximum number of images to process")
    parser.add_argument("--warmup_runs", type=int, default=5,
                       help="Number of warmup runs")
    parser.add_argument("--create_sample_dataset", action="store_true",
                       help="Create a sample dataset for testing")
    parser.add_argument("--download_pascal_voc", action="store_true",
                       help="Download Pascal VOC 2012 dataset")
    parser.add_argument("--extract_pascal_voc", action="store_true",
                       help="Extract Pascal VOC 2012 dataset from /data/pascal.zip")
    parser.add_argument("--use_pascal_voc_prompts", action="store_true",
                       help="Use Pascal VOC class names as prompts")
    
    args = parser.parse_args()
    
    # Parse prompts
    if args.use_pascal_voc_prompts:
        prompts = get_pascal_voc_prompts()
        print(f"Using Pascal VOC prompts: {len(prompts)} classes")
    else:
        prompts = args.prompts.strip("[]()").split(',')
        prompts = [p.strip().strip('"\'') for p in prompts]
    
    # Handle dataset preparation
    if args.extract_pascal_voc:
        print("Extracting Pascal VOC 2012 dataset from /data/pascal.zip...")
        dataset_path = extract_pascal_voc_dataset("/data/pascal.zip", args.dataset, args.max_images)
        if dataset_path is None:
            print("Failed to extract Pascal VOC dataset. Exiting.")
            sys.exit(1)
    elif args.download_pascal_voc:
        print("Downloading Pascal VOC 2012 dataset...")
        dataset_path = download_pascal_voc_dataset(args.dataset, args.max_images)
        if dataset_path is None:
            print("Failed to download Pascal VOC dataset. Exiting.")
            sys.exit(1)
    elif args.create_sample_dataset:
        dataset_path = create_sample_dataset(args.dataset, 10)
    else:
        dataset_path = args.dataset
    
    # Initialize benchmark
    benchmark = BatchBenchmark(
        model_name=args.model,
        image_encoder_engine=args.image_encoder_engine
    )
    
    # Run benchmark
    results = benchmark.run_benchmark(
        dataset_path=dataset_path,
        prompts=prompts,
        threshold=args.threshold,
        output_dir=args.output_dir,
        save_visualizations=args.save_visualizations,
        max_images=args.max_images,
        warmup_runs=args.warmup_runs
    )
    
    # Print and save results
    benchmark.print_results(results)
    
    results_file = os.path.join(args.output_dir, "benchmark_results.json")
    benchmark.save_results(results, results_file)
