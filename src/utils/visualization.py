"""
Visualization utilities for model evaluation.

Reusable plotting functions for confusion matrices, ROC curves, 
precision-recall curves, and other evaluation visualizations.

Can be used across different models (KIOCMIL CADA, YOLO, DETR, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from pathlib import Path


def plot_confusion_matrix(cm, classes, save_path, title="Confusion Matrix"):
    """
    Plot confusion matrix with counts.
    
    Args:
        cm: Confusion matrix (numpy array)
        classes: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", 
        xticklabels=classes, yticklabels=classes
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_confusion_matrix_normalized(cm, classes, save_path, title="Normalized Confusion Matrix"):
    """
    Plot normalized confusion matrix (percentages).
    
    Args:
        cm: Confusion matrix (numpy array)
        classes: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        vmin=0, vmax=1
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved normalized confusion matrix to {save_path}")


def plot_roc_curve(targets, probs, n_classes, class_names, save_path, title="ROC Curve"):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        targets: Ground truth labels (numpy array)
        probs: Predicted probabilities (numpy array, shape: N x n_classes)
        n_classes: Number of classes
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    # Convert targets to one-hot
    targets_one_hot = np.eye(n_classes)[targets]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        label = class_names[i] if class_names else f"Class {i}"
        fpr[i], tpr[i], _ = metrics.roc_curve(targets_one_hot[:, i], probs[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"{label} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to {save_path}")


def plot_precision_recall_curve(targets, probs, n_classes, class_names, save_path, title="Precision-Recall Curve"):
    """
    Plot precision-recall curves for multi-class classification.
    
    Args:
        targets: Ground truth labels (numpy array)
        probs: Predicted probabilities (numpy array, shape: N x n_classes)
        n_classes: Number of classes
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    targets_one_hot = np.eye(n_classes)[targets]
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        label = class_names[i] if class_names else f"Class {i}"
        precision, recall, _ = metrics.precision_recall_curve(
            targets_one_hot[:, i], probs[:, i]
        )
        ap = metrics.average_precision_score(targets_one_hot[:, i], probs[:, i])
        plt.plot(recall, precision, label=f"{label} (AP = {ap:.2f})")
    
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved PR curve to {save_path}")


def plot_metric_curves(targets, probs, preds, n_classes, class_names, save_dir):
    """
    Plot F1, Precision, and Recall bar charts by class.
    
    Args:
        targets: Ground truth labels (numpy array)
        probs: Predicted probabilities (numpy array)
        preds: Predicted labels (numpy array)
        n_classes: Number of classes
        class_names: List of class names
        save_dir: Directory to save plots (Path object)
    """
    # Calculate per-class metrics
    report = metrics.classification_report(targets, preds, output_dict=True, zero_division=0)
    
    # Extract metrics for each class
    class_metrics = []
    for i in range(n_classes):
        class_key = str(i)
        if class_key in report:
            class_metrics.append({
                'class': class_names[i] if class_names else f"Class {i}",
                'precision': report[class_key]['precision'],
                'recall': report[class_key]['recall'],
                'f1': report[class_key]['f1-score'],
                'support': report[class_key]['support']
            })
    
    classes = [m['class'] for m in class_metrics]
    precisions = [m['precision'] for m in class_metrics]
    recalls = [m['recall'] for m in class_metrics]
    f1s = [m['f1'] for m in class_metrics]
    
    # F1 Score Curve
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(classes)), f1s, color='steelblue', edgecolor='black')
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("F1 Score by Class", fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "F1_curve.png", dpi=150)
    plt.close()
    print(f"Saved F1 curve to {save_dir}/F1_curve.png")
    
    # Precision Curve
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(classes)), precisions, color='coral', edgecolor='black')
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision by Class", fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "P_curve.png", dpi=150)
    plt.close()
    print(f"Saved Precision curve to {save_dir}/P_curve.png")
    
    # Recall Curve
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(classes)), recalls, color='mediumseagreen', edgecolor='black')
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Recall", fontsize=12)
    plt.title("Recall by Class", fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "R_curve.png", dpi=150)
    plt.close()
    print(f"Saved Recall curve to {save_dir}/R_curve.png")


def plot_label_distribution(targets, class_names, save_path):
    """
    Plot label distribution in dataset.
    
    Args:
        targets: Ground truth labels (numpy array)
        class_names: List of class names
        save_path: Path to save the plot
    """
    unique, counts = np.unique(targets, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
    bars = plt.bar(range(len(unique)), counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(targets)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    labels = [class_names[i] if class_names else f"Class {i}" for i in unique]
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title("Label Distribution in Dataset", fontsize=14, fontweight='bold')
    plt.xticks(range(len(unique)), labels, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved label distribution to {save_path}")


def plot_training_curves(history, save_dir, metrics_to_plot=['loss', 'accuracy']):
    """
    Plot training history curves (loss, accuracy, etc.).
    
    Args:
        history: Dictionary with 'train' and 'val' keys containing metric lists
        save_dir: Directory to save plots (Path object)
        metrics_to_plot: List of metric names to plot
    
    Example history format:
        {
            'train': {'loss': [...], 'accuracy': [...]},
            'val': {'loss': [...], 'accuracy': [...]}
        }
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for metric in metrics_to_plot:
        if metric not in history.get('train', {}):
            continue
            
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(history['train'][metric]) + 1)
        plt.plot(epochs, history['train'][metric], 'b-', label=f'Train {metric}')
        
        if 'val' in history and metric in history['val']:
            plt.plot(epochs, history['val'][metric], 'r-', label=f'Val {metric}')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'{metric.capitalize()} over Epochs', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f"{metric}_curve.png", dpi=150)
        plt.close()
        print(f"Saved {metric} curve to {save_dir}/{metric}_curve.png")
