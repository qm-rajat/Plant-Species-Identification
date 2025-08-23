"""
Evaluation Module for Plant Species Identification
Provides comprehensive evaluation metrics and visualization tools
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import json
from collections import defaultdict


class ModelEvaluator:
    """Comprehensive model evaluation and metrics calculation"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.results_history = []
        self.edge_detection_results = {}
        self.classification_results = {}
    
    def evaluate_edge_detection(self, original_images: List[np.ndarray],
                               edge_results: Dict[str, List[np.ndarray]],
                               ground_truth: Optional[List[np.ndarray]] = None,
                               save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate edge detection methods
        
        Args:
            original_images: List of original images
            edge_results: Dictionary with method names and their edge detection results
            ground_truth: Optional ground truth edge maps
            save_dir: Directory to save results
            
        Returns:
            Evaluation results dictionary
        """
        print("Evaluating edge detection methods...")
        
        results = {
            'methods': list(edge_results.keys()),
            'n_images': len(original_images),
            'metrics': {},
            'processing_times': {},
            'visual_quality': {},
            'comparison_summary': {}
        }
        
        # Evaluate each method
        for method_name, method_results in edge_results.items():
            print(f"Evaluating {method_name}...")
            
            method_metrics = {
                'psnr_scores': [],
                'ssim_scores': [],
                'edge_density': [],
                'edge_strength': [],
                'processing_time': 0
            }
            
            # Measure processing time
            start_time = time.time()
            
            for i, (original, edges) in enumerate(zip(original_images, method_results)):
                # Convert to grayscale if needed
                if len(original.shape) == 3:
                    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
                else:
                    original_gray = original.copy()
                
                # Calculate edge density
                edge_pixels = np.sum(edges > 0)
                total_pixels = edges.shape[0] * edges.shape[1]
                edge_density = edge_pixels / total_pixels
                method_metrics['edge_density'].append(edge_density)
                
                # Calculate edge strength
                edge_strength = np.mean(edges[edges > 0]) if edge_pixels > 0 else 0
                method_metrics['edge_strength'].append(edge_strength)
                
                # Compare with ground truth if available
                if ground_truth and i < len(ground_truth):
                    gt = ground_truth[i]
                    
                    # Ensure same size
                    if edges.shape != gt.shape:
                        edges_resized = cv2.resize(edges, (gt.shape[1], gt.shape[0]))
                    else:
                        edges_resized = edges
                    
                    # Calculate PSNR
                    try:
                        psnr = peak_signal_noise_ratio(gt, edges_resized, data_range=255)
                        method_metrics['psnr_scores'].append(psnr)
                    except:
                        method_metrics['psnr_scores'].append(0)
                    
                    # Calculate SSIM
                    try:
                        ssim = structural_similarity(gt, edges_resized, data_range=255)
                        method_metrics['ssim_scores'].append(ssim)
                    except:
                        method_metrics['ssim_scores'].append(0)
            
            method_metrics['processing_time'] = time.time() - start_time
            
            # Calculate statistics
            results['metrics'][method_name] = {
                'psnr_mean': np.mean(method_metrics['psnr_scores']) if method_metrics['psnr_scores'] else 0,
                'psnr_std': np.std(method_metrics['psnr_scores']) if method_metrics['psnr_scores'] else 0,
                'ssim_mean': np.mean(method_metrics['ssim_scores']) if method_metrics['ssim_scores'] else 0,
                'ssim_std': np.std(method_metrics['ssim_scores']) if method_metrics['ssim_scores'] else 0,
                'edge_density_mean': np.mean(method_metrics['edge_density']),
                'edge_density_std': np.std(method_metrics['edge_density']),
                'edge_strength_mean': np.mean(method_metrics['edge_strength']),
                'edge_strength_std': np.std(method_metrics['edge_strength']),
                'processing_time': method_metrics['processing_time'],
                'processing_time_per_image': method_metrics['processing_time'] / len(original_images)
            }
        
        # Create comparison summary
        if ground_truth:
            best_psnr = max(results['metrics'].keys(), 
                           key=lambda x: results['metrics'][x]['psnr_mean'])
            best_ssim = max(results['metrics'].keys(), 
                           key=lambda x: results['metrics'][x]['ssim_mean'])
            
            results['comparison_summary'] = {
                'best_psnr_method': best_psnr,
                'best_psnr_score': results['metrics'][best_psnr]['psnr_mean'],
                'best_ssim_method': best_ssim,
                'best_ssim_score': results['metrics'][best_ssim]['ssim_mean']
            }
        
        # Find fastest method
        fastest_method = min(results['metrics'].keys(), 
                           key=lambda x: results['metrics'][x]['processing_time'])
        results['comparison_summary']['fastest_method'] = fastest_method
        results['comparison_summary']['fastest_time'] = results['metrics'][fastest_method]['processing_time']
        
        # Save results
        if save_dir:
            self._save_edge_detection_results(results, save_dir)
        
        self.edge_detection_results = results
        return results
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None,
                              class_names: Optional[List[str]] = None,
                              model_name: str = "model",
                              save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate classification performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            class_names: Names of classes
            model_name: Name of the model
            save_dir: Directory to save results
            
        Returns:
            Evaluation results dictionary
        """
        print(f"Evaluating classification performance for {model_name}...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(len(np.unique(y_true)))]
        
        class_report = classification_report(y_true, y_pred, target_names=class_names, 
                                           output_dict=True, zero_division=0)
        
        results = {
            'model_name': model_name,
            'n_samples': len(y_true),
            'n_classes': len(class_names),
            'class_names': class_names,
            'overall_metrics': {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_micro': precision_micro,
                'recall_micro': recall_micro,
                'f1_micro': f1_micro
            },
            'confusion_matrix': cm,
            'classification_report': class_report,
            'per_class_metrics': {}
        }
        
        # Extract per-class metrics
        for class_name in class_names:
            if class_name in class_report:
                results['per_class_metrics'][class_name] = {
                    'precision': class_report[class_name]['precision'],
                    'recall': class_report[class_name]['recall'],
                    'f1_score': class_report[class_name]['f1-score'],
                    'support': class_report[class_name]['support']
                }
        
        # ROC and PR curves (if probabilities available)
        if y_pred_proba is not None:
            roc_auc_results = self._calculate_roc_auc(y_true, y_pred_proba, class_names)
            pr_auc_results = self._calculate_pr_auc(y_true, y_pred_proba, class_names)
            
            results['roc_auc'] = roc_auc_results
            results['pr_auc'] = pr_auc_results
        
        # Additional analysis
        results['error_analysis'] = self._analyze_classification_errors(y_true, y_pred, class_names)
        results['class_balance'] = self._analyze_class_balance(y_true, class_names)
        
        # Save results
        if save_dir:
            self._save_classification_results(results, save_dir)
        
        self.classification_results[model_name] = results
        return results
    
    def compare_models(self, model_results: Dict[str, Dict], 
                      save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple models
        
        Args:
            model_results: Dictionary of model evaluation results
            save_dir: Directory to save comparison results
            
        Returns:
            Model comparison results
        """
        print("Comparing models...")
        
        comparison = {
            'models': list(model_results.keys()),
            'metrics_comparison': {},
            'rankings': {},
            'best_model': {},
            'summary_table': None
        }
        
        # Extract metrics for comparison
        metrics_to_compare = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        for metric in metrics_to_compare:
            comparison['metrics_comparison'][metric] = {}
            for model_name, results in model_results.items():
                if 'overall_metrics' in results:
                    comparison['metrics_comparison'][metric][model_name] = \
                        results['overall_metrics'].get(metric, 0)
        
        # Create rankings
        for metric in metrics_to_compare:
            if metric in comparison['metrics_comparison']:
                sorted_models = sorted(comparison['metrics_comparison'][metric].items(),
                                     key=lambda x: x[1], reverse=True)
                comparison['rankings'][metric] = [model for model, score in sorted_models]
        
        # Find best overall model (based on F1 macro)
        if 'f1_macro' in comparison['rankings']:
            best_model_name = comparison['rankings']['f1_macro'][0]
            comparison['best_model'] = {
                'name': best_model_name,
                'metrics': model_results[best_model_name]['overall_metrics']
            }
        
        # Create summary table
        summary_data = []
        for model_name in comparison['models']:
            if model_name in model_results and 'overall_metrics' in model_results[model_name]:
                metrics = model_results[model_name]['overall_metrics']
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                    'Precision': f"{metrics.get('precision_macro', 0):.3f}",
                    'Recall': f"{metrics.get('recall_macro', 0):.3f}",
                    'F1-Score': f"{metrics.get('f1_macro', 0):.3f}"
                })
        
        comparison['summary_table'] = pd.DataFrame(summary_data)
        
        # Save comparison
        if save_dir:
            self._save_model_comparison(comparison, save_dir)
        
        return comparison
    
    def evaluate_noise_robustness(self, model, test_images: List[np.ndarray],
                                 test_labels: List[str], noise_levels: List[str] = ['low', 'medium', 'high'],
                                 save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate model robustness to noise
        
        Args:
            model: Trained model
            test_images: Test images
            test_labels: Test labels
            noise_levels: Noise levels to test
            save_dir: Directory to save results
            
        Returns:
            Noise robustness evaluation results
        """
        from app.utils.preprocessing import ImagePreprocessor
        
        print("Evaluating noise robustness...")
        
        preprocessor = ImagePreprocessor()
        results = {
            'noise_levels': noise_levels,
            'original_accuracy': 0,
            'noise_results': {},
            'degradation_analysis': {}
        }
        
        # Evaluate on original images
        if hasattr(model, 'predict'):
            original_predictions = []
            for img in test_images:
                pred = model.predict_single(img)
                original_predictions.append(pred['predicted_class'])
            
            results['original_accuracy'] = accuracy_score(test_labels, original_predictions)
        
        # Evaluate on noisy images
        for noise_level in noise_levels:
            print(f"Testing noise level: {noise_level}")
            
            noisy_predictions = []
            for img in test_images:
                # Add noise
                if len(img.shape) == 3:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = img.copy()
                
                noisy_img = preprocessor.add_gaussian_noise(gray_img, noise_level)
                
                # Predict
                if hasattr(model, 'predict_single'):
                    pred = model.predict_single(noisy_img)
                    noisy_predictions.append(pred['predicted_class'])
            
            # Calculate metrics
            noisy_accuracy = accuracy_score(test_labels, noisy_predictions)
            accuracy_drop = results['original_accuracy'] - noisy_accuracy
            relative_drop = (accuracy_drop / results['original_accuracy']) * 100 if results['original_accuracy'] > 0 else 0
            
            results['noise_results'][noise_level] = {
                'accuracy': noisy_accuracy,
                'accuracy_drop': accuracy_drop,
                'relative_drop_percent': relative_drop,
                'predictions': noisy_predictions
            }
        
        # Analyze degradation pattern
        accuracies = [results['original_accuracy']] + \
                    [results['noise_results'][level]['accuracy'] for level in noise_levels]
        
        results['degradation_analysis'] = {
            'accuracy_curve': accuracies,
            'total_degradation': results['original_accuracy'] - accuracies[-1],
            'degradation_rate': np.mean(np.diff(accuracies)) if len(accuracies) > 1 else 0
        }
        
        # Save results
        if save_dir:
            self._save_noise_robustness_results(results, save_dir)
        
        return results
    
    def generate_evaluation_report(self, save_path: str):
        """Generate comprehensive evaluation report"""
        print("Generating evaluation report...")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'edge_detection_results': self.edge_detection_results,
            'classification_results': self.classification_results,
            'summary': {}
        }
        
        # Generate summary
        if self.edge_detection_results:
            report['summary']['edge_detection'] = {
                'methods_evaluated': len(self.edge_detection_results.get('methods', [])),
                'images_processed': self.edge_detection_results.get('n_images', 0),
                'best_method': self.edge_detection_results.get('comparison_summary', {}).get('best_psnr_method', 'N/A')
            }
        
        if self.classification_results:
            best_model = max(self.classification_results.items(), 
                           key=lambda x: x[1]['overall_metrics'].get('accuracy', 0))
            report['summary']['classification'] = {
                'models_evaluated': len(self.classification_results),
                'best_model': best_model[0],
                'best_accuracy': best_model[1]['overall_metrics'].get('accuracy', 0)
            }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Evaluation report saved to {save_path}")
        return report
    
    def plot_edge_detection_comparison(self, save_path: Optional[str] = None):
        """Plot edge detection method comparison"""
        if not self.edge_detection_results:
            print("No edge detection results to plot")
            return
        
        metrics = self.edge_detection_results['metrics']
        methods = list(metrics.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # PSNR comparison
        psnr_scores = [metrics[method]['psnr_mean'] for method in methods]
        psnr_errors = [metrics[method]['psnr_std'] for method in methods]
        
        axes[0, 0].bar(methods, psnr_scores, yerr=psnr_errors, capsize=5)
        axes[0, 0].set_title('PSNR Comparison')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # SSIM comparison
        ssim_scores = [metrics[method]['ssim_mean'] for method in methods]
        ssim_errors = [metrics[method]['ssim_std'] for method in methods]
        
        axes[0, 1].bar(methods, ssim_scores, yerr=ssim_errors, capsize=5)
        axes[0, 1].set_title('SSIM Comparison')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Processing time comparison
        processing_times = [metrics[method]['processing_time'] for method in methods]
        
        axes[1, 0].bar(methods, processing_times)
        axes[1, 0].set_title('Processing Time Comparison')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Edge density comparison
        edge_densities = [metrics[method]['edge_density_mean'] for method in methods]
        edge_density_errors = [metrics[method]['edge_density_std'] for method in methods]
        
        axes[1, 1].bar(methods, edge_densities, yerr=edge_density_errors, capsize=5)
        axes[1, 1].set_title('Edge Density Comparison')
        axes[1, 1].set_ylabel('Edge Density')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_results(self, model_name: str, save_path: Optional[str] = None):
        """Plot classification results for a specific model"""
        if model_name not in self.classification_results:
            print(f"No results found for model: {model_name}")
            return
        
        results = self.classification_results[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Confusion matrix
        cm = results['confusion_matrix']
        class_names = results['class_names']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0])
        axes[0, 0].set_title(f'Confusion Matrix - {model_name}')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Per-class metrics
        per_class = results['per_class_metrics']
        classes = list(per_class.keys())
        precisions = [per_class[cls]['precision'] for cls in classes]
        recalls = [per_class[cls]['recall'] for cls in classes]
        f1_scores = [per_class[cls]['f1_score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        axes[0, 1].bar(x - width, precisions, width, label='Precision')
        axes[0, 1].bar(x, recalls, width, label='Recall')
        axes[0, 1].bar(x + width, f1_scores, width, label='F1-Score')
        
        axes[0, 1].set_title('Per-Class Metrics')
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(classes, rotation=45)
        axes[0, 1].legend()
        
        # Class distribution
        class_balance = results['class_balance']
        class_counts = [class_balance[cls] for cls in classes]
        
        axes[1, 0].pie(class_counts, labels=classes, autopct='%1.1f%%')
        axes[1, 0].set_title('Class Distribution')
        
        # ROC curves (if available)
        if 'roc_auc' in results:
            for cls in classes[:5]:  # Limit to first 5 classes for readability
                if cls in results['roc_auc']:
                    fpr = results['roc_auc'][cls]['fpr']
                    tpr = results['roc_auc'][cls]['tpr']
                    auc_score = results['roc_auc'][cls]['auc']
                    axes[1, 1].plot(fpr, tpr, label=f'{cls} (AUC = {auc_score:.2f})')
            
            axes[1, 1].plot([0, 1], [0, 1], 'k--')
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].set_title('ROC Curves')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'ROC curves not available\n(probabilities needed)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('ROC Curves')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # Helper methods
    def _calculate_roc_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                          class_names: List[str]) -> Dict[str, Any]:
        """Calculate ROC AUC for each class"""
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        roc_auc = {}
        for i, class_name in enumerate(class_names):
            if y_true_bin.shape[1] > 1:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            else:
                # Binary case
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            
            auc_score = auc(fpr, tpr)
            roc_auc[class_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': auc_score
            }
        
        return roc_auc
    
    def _calculate_pr_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                         class_names: List[str]) -> Dict[str, Any]:
        """Calculate Precision-Recall AUC for each class"""
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        pr_auc = {}
        for i, class_name in enumerate(class_names):
            if y_true_bin.shape[1] > 1:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                ap_score = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            else:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
                ap_score = average_precision_score(y_true, y_pred_proba[:, 1])
            
            pr_auc[class_name] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'ap_score': ap_score
            }
        
        return pr_auc
    
    def _analyze_classification_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     class_names: List[str]) -> Dict[str, Any]:
        """Analyze classification errors"""
        error_analysis = {
            'total_errors': np.sum(y_true != y_pred),
            'error_rate': np.mean(y_true != y_pred),
            'confusion_pairs': defaultdict(int)
        }
        
        # Find most common confusion pairs
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                pair = f"{class_names[true_label]} -> {class_names[pred_label]}"
                error_analysis['confusion_pairs'][pair] += 1
        
        # Sort by frequency
        error_analysis['confusion_pairs'] = dict(
            sorted(error_analysis['confusion_pairs'].items(), 
                  key=lambda x: x[1], reverse=True)
        )
        
        return error_analysis
    
    def _analyze_class_balance(self, y_true: np.ndarray, class_names: List[str]) -> Dict[str, int]:
        """Analyze class balance in dataset"""
        unique, counts = np.unique(y_true, return_counts=True)
        return {class_names[i]: count for i, count in zip(unique, counts)}
    
    def _save_edge_detection_results(self, results: Dict, save_dir: str):
        """Save edge detection results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(save_dir, 'edge_detection_metrics.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save plots
        self.plot_edge_detection_comparison(
            os.path.join(save_dir, 'edge_detection_comparison.png')
        )
    
    def _save_classification_results(self, results: Dict, save_dir: str):
        """Save classification results"""
        os.makedirs(save_dir, exist_ok=True)
        model_name = results['model_name']
        
        # Save metrics
        with open(os.path.join(save_dir, f'{model_name}_metrics.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save plots
        self.plot_classification_results(
            model_name, 
            os.path.join(save_dir, f'{model_name}_results.png')
        )
    
    def _save_model_comparison(self, comparison: Dict, save_dir: str):
        """Save model comparison results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save comparison data
        with open(os.path.join(save_dir, 'model_comparison.json'), 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Save summary table
        if comparison['summary_table'] is not None:
            comparison['summary_table'].to_csv(
                os.path.join(save_dir, 'model_comparison_summary.csv'), 
                index=False
            )
    
    def _save_noise_robustness_results(self, results: Dict, save_dir: str):
        """Save noise robustness results"""
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, 'noise_robustness.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    # Sample classification data
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_pred_proba = np.random.rand(n_samples, n_classes)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)  # Normalize
    
    class_names = [f'Species_{i}' for i in range(n_classes)]
    
    # Evaluate classification
    results = evaluator.evaluate_classification(
        y_true, y_pred, y_pred_proba, class_names, "RandomForest"
    )
    
    print(f"Classification accuracy: {results['overall_metrics']['accuracy']:.3f}")
    print(f"F1-score (macro): {results['overall_metrics']['f1_macro']:.3f}")