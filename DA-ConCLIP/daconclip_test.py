import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing import List, Tuple, Dict, Optional
import os
import pandas as pd
from matplotlib import colors as mcolors
from prompts import prompts_india, prompts_germany, prompts_china
from categories import road_sign_categories

class RoadSignClassifier:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", top_k: int = 10, batch_size: int = 32, use_finetuned: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        ft_file = "finetuned_multidomain_clip_germany_art.pth"
        if use_finetuned and os.path.exists(ft_file):
            print("\n>>> Loading fine-tuned weights from finetuned model", ft_file ,"\n")
            self.model.load_state_dict(torch.load(ft_file, map_location=self.device), strict=False)
        else:
            print("\n>>> Using pretrained CLIP model without fine-tuning\n")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.top_k = top_k
        self.batch_size = batch_size
        self.categories = road_sign_categories
        self.prompt_variations = prompts_germany

        # Initialize text features
        self._initialize_text_features()

    def _initialize_text_features(self):
        """Precompute text embeddings for all prompts"""
        all_prompts = [p for variations in self.prompt_variations for p in variations]
        self.prompt_to_category = []
        for cat_idx, variations in enumerate(self.prompt_variations):
            self.prompt_to_category.extend([cat_idx] * len(variations))
        
        text_inputs = self.processor(
            text=all_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:

        target_size = 512

        # Resize to target size
        if min(image.size) < target_size:
            image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)

        return image

    def classify(self, image: Image.Image, return_all: bool = False) -> List[Tuple[str, float]]:
        """Classify a single traffic sign image"""
        inputs = self.processor(
            text=None,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            # Get image features
            image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ self.text_features.T
            
            # Convert to probabilities
            prompt_probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Aggregate by category
            category_probs = np.zeros(len(self.categories))
            for prompt_idx, prob in enumerate(prompt_probs):
                category_idx = self.prompt_to_category[prompt_idx]
                category_probs[category_idx] += prob
            
            # Normalize
            category_probs /= category_probs.sum()
            
            if return_all:
                # Return all categories sorted by probability
                sorted_indices = np.argsort(-category_probs)
                return [(self.categories[i], float(category_probs[i])) for i in sorted_indices]
            else:
                # Return top-k only
                top_k_indices = np.argpartition(category_probs, -self.top_k)[-self.top_k:]
                top_k_indices = top_k_indices[np.argsort(-category_probs[top_k_indices])]
                return [(self.categories[i], float(category_probs[i])) for i in top_k_indices]

    def classify_batch(self, images: List[Image.Image], return_all: bool = False) -> List[List[Tuple[str, float]]]:
        """Classify multiple images in batch"""
        if not images:
            return []
        
        inputs = self.processor(
            text=None,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ self.text_features.T
            
            prompt_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            batch_results = []
            for img_probs in prompt_probs:
                category_probs = np.zeros(len(self.categories))
                for prompt_idx, prob in enumerate(img_probs):
                    category_idx = self.prompt_to_category[prompt_idx]
                    category_probs[category_idx] += prob
                
                category_probs /= category_probs.sum()
                
                if return_all:
                    sorted_indices = np.argsort(-category_probs)
                    batch_results.append(
                        [(self.categories[i], float(category_probs[i])) for i in sorted_indices]
                    )
                else:
                    top_k_indices = np.argpartition(category_probs, -self.top_k)[-self.top_k:]
                    top_k_indices = top_k_indices[np.argsort(-category_probs[top_k_indices])]
                    batch_results.append(
                        [(self.categories[i], float(category_probs[i])) for i in top_k_indices]
                    )
            
            return batch_results

    def evaluate_directory(
        self,
        directory: str,
        ground_truth: Optional[Dict[str, str]] = None,
        return_all: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate all images in directory with ground truth comparison
        
        Args:
            directory: Path to image directory
            ground_truth: Dictionary of {filename: correct_label} (optional)
            return_all: Whether to return all predictions or just top-k
            
        Returns:
            DataFrame with columns:
            - Image: Filename
            - True Label: Ground truth label
            - Top Prediction: Model's top prediction with confidence
            - Correct: Whether prediction matches ground truth
            - All Predictions: All category probabilities
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} not found")
        
        image_files = [
            f for f in os.listdir(directory)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        rows = []
        
        for i in range(0, len(image_files), self.batch_size):
            batch_files = image_files[i:i+self.batch_size]
            batch_images = []
            
            for f in batch_files:
                try:
                    img = Image.open(os.path.join(directory, f))
                    img = self.preprocess_image(img)
                    batch_images.append(img)
                except Exception as e:
                    print(f"Error loading {f}: {str(e)}")
                    continue
            
            batch_preds = self.classify_batch(batch_images, return_all=return_all)
            
            for f, preds in zip(batch_files, batch_preds):
                # Extract ground truth from filename if not provided
                if ground_truth is None:
                    # Remove file extension and split by underscores
                    base_name = os.path.splitext(f)[0]
                    parts = base_name.split('_')
                    # Join all parts except the last (which is usually a number)
                    true_label = ' '.join(parts[:-1]).lower()
                else:
                    true_label = ground_truth.get(f, "Unknown")
                
                top_pred, top_conf = preds[0] if preds else ("N/A", 0)
                
                # Only take top k predictions (k is set in __init__)
                top_k_preds = preds[:self.top_k] if not return_all else preds
                
                rows.append({
                    "Image": f,
                    "True Label": true_label,
                    "Top Prediction": f"{top_pred} ({top_conf:.2%})",
                    "Correct": "✔" if top_pred.lower() == true_label.lower() else "✗",
                    "Top Predictions": "\n".join([f"{cat}: {prob:.2%}" for cat, prob in top_k_preds])
                })
        
        return pd.DataFrame(rows)
    
    def compute_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute classification metrics from evaluation results DataFrame.
        
        Args:
            results_df: DataFrame returned by evaluate_directory method
            
        Returns:
            Dictionary containing:
            - accuracy: Overall accuracy
            - macro_precision: Macro-averaged precision (only for classes with support)
            - macro_recall: Macro-averaged recall (only for classes with support)
            - macro_f1: Macro-averaged F1-score (only for classes with support)
            - weighted_precision: Support-weighted precision
            - weighted_recall: Support-weighted recall
            - weighted_f1: Support-weighted F1-score
            - class_metrics: Dictionary with metrics per class
        """
        if 'True Label' not in results_df or 'Correct' not in results_df:
            raise ValueError("DataFrame must contain 'True Label' and 'Correct' columns")
        
        # Extract predictions (handling the "pred (XX.XX%)" format)
        pred_labels = results_df['Top Prediction'].str.extract(r'^([^(]+)')[0].str.lower().str.strip()
        true_labels = results_df['True Label'].str.lower()
        
        # Get all unique classes that actually appear in ground truth
        present_classes = [cls.lower() for cls in self.categories if cls.lower() in true_labels.unique()]
        if not present_classes:
            return {
                'accuracy': 0,
                'macro_precision': 0,
                'macro_recall': 0,
                'macro_f1': 0,
                'weighted_precision': 0,
                'weighted_recall': 0,
                'weighted_f1': 0,
                'class_metrics': {}
            }
        
        # Initialize metrics storage
        class_metrics = {}
        tp = {cls: 0 for cls in present_classes}  # True positives
        fp = {cls: 0 for cls in present_classes}  # False positives
        fn = {cls: 0 for cls in present_classes}  # False negatives
        support = {cls: 0 for cls in present_classes}  # Support count
        
        # Calculate TP, FP, FN for each class
        for true, pred in zip(true_labels, pred_labels):
            if true in present_classes:
                support[true] += 1
                if pred == true:
                    tp[true] += 1
                else:
                    fn[true] += 1
                    if pred in present_classes:
                        fp[pred] += 1
        
        # Calculate class-wise metrics
        valid_classes = []
        sum_precision = 0
        sum_recall = 0
        sum_f1 = 0
        total_support = sum(support.values())
        
        for cls in present_classes:
            if support[cls] > 0:
                precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
                recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[cls] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'true_positives' : tp[cls],
                    'false_negatives' : fn[cls],
                    'false_positives' : fp[cls],
                    'support': support[cls]
                }
                
                sum_precision += precision
                sum_recall += recall
                sum_f1 += f1
                valid_classes.append(cls)
        
        # Calculate macro-averaged metrics (only over classes with support)
        num_valid_classes = len(valid_classes)
        macro_precision = sum_precision / num_valid_classes if num_valid_classes > 0 else 0
        macro_recall = sum_recall / num_valid_classes if num_valid_classes > 0 else 0
        macro_f1 = sum_f1 / num_valid_classes if num_valid_classes > 0 else 0
        
        # Calculate weighted averages
        weighted_precision = sum(
            class_metrics[cls]['precision'] * class_metrics[cls]['support']
            for cls in valid_classes
        ) / total_support if total_support > 0 else 0
        
        weighted_recall = sum(
            class_metrics[cls]['recall'] * class_metrics[cls]['support']
            for cls in valid_classes
        ) / total_support if total_support > 0 else 0
        
        weighted_f1 = sum(
            class_metrics[cls]['f1'] * class_metrics[cls]['support']
            for cls in valid_classes
        ) / total_support if total_support > 0 else 0
        
        # Calculate accuracy
        accuracy = sum(results_df['Correct'] == '✔') / len(results_df) if len(results_df) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'class_metrics': class_metrics
        }
    
    def compute_class_wise_topk_metrics(self, results_df: pd.DataFrame, max_k: int = 10) -> Dict[str, Dict]:
        """
        Compute meaningful class-wise metrics for top-k predictions with styled output

        Args:
            results_df: Evaluation results DataFrame
            max_k: Maximum k value to compute metrics for

        Returns:
            Dictionary containing:
            - overall: Dict of overall metrics for each k
            - classes: Dict of class-wise metrics for each k
            - summary: Styled DataFrame with summary statistics
            - html: HTML representation of styled table
        """
        if 'True Label' not in results_df or 'Top Predictions' not in results_df:
            raise ValueError("DataFrame must contain 'True Label' and 'Top Predictions' columns")

        true_labels = results_df['True Label'].str.lower()
        present_classes = [cls.lower() for cls in self.categories if cls.lower() in true_labels.unique()]

        # Initialize data structures
        results = {
            'overall': {k: {'correct': 0, 'total': len(results_df)} for k in range(1, max_k + 1)},
            'classes': {cls: {k: {'correct': 0, 'total': 0} for k in range(1, max_k + 1)} 
                       for cls in present_classes}
        }

        # Process each sample
        for idx, row in results_df.iterrows():
            true_label = row['True Label'].lower()
            if true_label not in present_classes:
                continue

            pred_entries = row['Top Predictions'].split('\n')
            pred_labels = [entry.split(':')[0].strip().lower() for entry in pred_entries]

            # Update counts for each k value
            for k in range(1, max_k + 1):
                top_k_preds = pred_labels[:k]
                if true_label in top_k_preds:
                    results['overall'][k]['correct'] += 1
                    results['classes'][true_label][k]['correct'] += 1
                results['classes'][true_label][k]['total'] += 1

        # Calculate metrics
        metrics = {
            'overall': {},
            'classes': {},
            'summary': pd.DataFrame(columns=['Class', 'Support'] + [f'Top-{k}' for k in range(1, max_k + 1)])
        }

        # Overall metrics
        for k in range(1, max_k + 1):
            metrics['overall'][k] = {
                'accuracy': results['overall'][k]['correct'] / results['overall'][k]['total'],
                'correct': results['overall'][k]['correct'],
                'total': results['overall'][k]['total']
            }

        # Class-wise metrics
        class_rows = []
        for cls in present_classes:
            cls_data = {
                'Class': cls.title(),
                'Support': results['classes'][cls][1]['total']
            }
            for k in range(1, max_k + 1):
                accuracy = results['classes'][cls][k]['correct'] / results['classes'][cls][k]['total']
                cls_data[f'Top-{k}'] = accuracy

            metrics['classes'][cls] = cls_data
            class_rows.append(cls_data)

        # Create summary DataFrame
        df = pd.DataFrame(class_rows)

        # Calculate mean across classes (removed STD as requested)
        mean_row = {
            'Class': 'MEAN',
            'Support': df['Support'].mean()
        }
        for k in range(1, max_k + 1):
            mean_row[f'Top-{k}'] = df[f'Top-{k}'].mean()

        # Modern way to add a row - create a new DataFrame and concatenate
        mean_df = pd.DataFrame([mean_row])
        df = pd.concat([df, mean_df], ignore_index=True)

        # Create styled DataFrame with the requested style
        styled_df = df.style \
            .set_caption("Class-wise Top-K Performance Metrics") \
            .set_properties(**{
                'text-align': 'center',
                'border': '1px solid black',
                'padding': '5px'
            }) \
            .format({
                'Support': '{:.0f}',
                **{f'Top-{k}': '{:.1%}' for k in range(1, max_k + 1)}
            }) \
            .background_gradient(
                cmap=mcolors.LinearSegmentedColormap.from_list(
                    "custom", ["#FF9999", "#FFFF99", "#FFA500", "#90EE90"]),
                subset=[f'Top-{k}' for k in range(1, max_k + 1)],
                vmin=0, vmax=1
            ) \
            .set_table_styles([{
                'selector': 'th',
                'props': [('background-color', '#40466e'), 
                         ('color', 'white'),
                         ('font-weight', 'bold'),
                         ('text-align', 'center')]
            }])

        # Add to metrics dictionary
        metrics['summary'] = styled_df

        # Get HTML using to_html() instead of render()
        metrics['html'] = styled_df.to_html()

        return metrics
    
    def save_class_metrics_html(self, metrics: Dict[str, float], filename: str = "class_metrics.html"):
        """
        Save per-class metrics as a styled HTML table

        Args:
            metrics: Dictionary returned by compute_metrics()
            filename: Output HTML filename
        """
        if not metrics['class_metrics']:
            print("No class metrics to save")
            return

        # Convert class metrics to DataFrame
        metrics_list = []
        for cls, m in metrics['class_metrics'].items():
            metrics_list.append({
                'Class': cls.title(),
                'Precision': m['precision'],
                'Recall': m['recall'],
                'F1-Score': m['f1'],
                'TP': m['true_positives'],
                'FP': m['false_positives'],
                'FN': m['false_negatives'],
                'Support': m['support']
            })

        df = pd.DataFrame(metrics_list)

        # Style the DataFrame
        styled_df = df.style \
            .set_caption("Per-Class Performance Metrics") \
            .set_properties(**{
                'text-align': 'center',
                'border': '1px solid black',
                'padding': '5px'
            }) \
            .format({
                'Precision': '{:.1%}',
                'Recall': '{:.1%}',
                'F1-Score': '{:.1%}'
            }, precision=1) \
            .background_gradient(cmap='Blues', subset=['Precision', 'Recall', 'F1-Score']) \
            .set_table_styles([{
                'selector': 'th',
                'props': [('background-color', '#40466e'), 
                         ('color', 'white'),
                         ('font-weight', 'bold'),
                         ('text-align', 'center')]
            }])

        # Save to HTML - version compatible approach
        try:
            # For newer pandas versions
            html_content = styled_df.to_html()
        except AttributeError:
            # For older pandas versions
            html_content = styled_df._repr_html_()

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Saved class metrics to {filename}")

if __name__ == "__main__":
    classifier = RoadSignClassifier(use_finetuned=True)

    results_df = classifier.evaluate_directory(
        directory="Germany/test_germany",
        #directory="India/test_india",
        #directory="China/cn_test",
        ground_truth=None,
        return_all=False
    )

    max_k = 5
    metrics = classifier.compute_metrics(results_df)
    metrics_top_k = classifier.compute_class_wise_topk_metrics(results_df, max_k=max_k)

    print("----------------------------------------------------------")
    print("Top 1 Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("----------------------------------------------------------")
    print("Overall Performance:")
    for k in range(1, max_k+1):
        acc = metrics_top_k['overall'][k]['accuracy']
        print(f"Top-{k} Accuracy: {acc:.2%} ({metrics_top_k['overall'][k]['correct']}/{metrics_top_k['overall'][k]['total']})")
    print("----------------------------------------------------------")

    #results_df.to_html("results_table.html")
    classifier.save_class_metrics_html(metrics, "road_sign_metrics.html")
    metrics_top_k['summary'].to_html("road_sign_metrics_top_k_ft_art_ger_ger.html")
