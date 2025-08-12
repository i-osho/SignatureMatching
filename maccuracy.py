import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from dataset1 import SiameseNetwork, predict
import torchvision.transforms as transforms
import os

path = r"69.60.pth"

def calculate_optimal_threshold(model, test_pairs, labels, transform, device):
    """
    Calculate optimal threshold for signature matching
    
    Args:
        model: Trained Siamese network
        test_pairs: List of tuples (ref_path, test_path)
        labels: List of ground truth labels (1 for genuine, 0 for forged)
        transform: Image transformation pipeline
        device: Computing device
    
    Returns:
        optimal_threshold: Best threshold value
        metrics: Dictionary containing accuracy, precision, recall, etc.
    """
    
    # Calculate similarity scores for all test pairs
    scores = []
    print("Calculating similarity scores...")
    for i, (ref_path, test_path) in enumerate(test_pairs):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_pairs)} pairs")
        score = predict(model, ref_path, test_path, transform, device)
        scores.append(score)
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold using Youden's index
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    predictions = (scores >= optimal_threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate precision, recall, F1
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'auc': roc_auc,
        'true_positive_rate': tpr[optimal_idx],
        'false_positive_rate': fpr[optimal_idx]
    }
    
    return optimal_threshold, metrics, scores, labels, fpr, tpr

def calculate_score_distribution(model, test_pairs, labels, transform, device):
    """
    Calculate score distribution for genuine-only or forged-only pairs
    (No threshold calculation as we only have one class)
    """
    scores = []
    print("Calculating similarity scores...")
    for i, (ref_path, test_path) in enumerate(test_pairs):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_pairs)} pairs")
        score = predict(model, ref_path, test_path, transform, device)
        scores.append(score)
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Calculate basic statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    print(f"\nScore Statistics:")
    print(f"Mean: {mean_score:.4f}")
    print(f"Std: {std_score:.4f}")
    print(f"Min: {min_score:.4f}")
    print(f"Max: {max_score:.4f}")
    
    return scores, labels

def plot_threshold_analysis(scores, labels, fpr, tpr, optimal_threshold, metrics):
    """Plot ROC curve and threshold analysis"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {metrics["auc"]:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.scatter(metrics['false_positive_rate'], metrics['true_positive_rate'], 
               color='red', s=100, label=f'Optimal threshold = {optimal_threshold:.3f}')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # Score Distribution
    genuine_scores = scores[labels == 1]
    forged_scores = scores[labels == 0]
    
    ax2.hist(genuine_scores, bins=30, alpha=0.7, label='Genuine', color='green')
    ax2.hist(forged_scores, bins=30, alpha=0.7, label='Forged', color='red')
    ax2.axvline(optimal_threshold, color='black', linestyle='--', 
               label=f'Threshold = {optimal_threshold:.3f}')
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Score Distribution')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    # plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_single_class_distribution(scores, labels, class_name):
    """Plot score distribution for single class (genuine-only or forged-only)"""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    color = 'green' if class_name == 'Genuine' else 'red'
    ax.hist(scores, bins=30, alpha=0.7, label=f'{class_name} Pairs', color=color)
    
    # Add statistics lines
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    ax.axvline(mean_score, color='black', linestyle='-', linewidth=2,
               label=f'Mean = {mean_score:.3f}')
    ax.axvline(mean_score + std_score, color='gray', linestyle='--', 
               label=f'Mean + Std = {mean_score + std_score:.3f}')
    ax.axvline(mean_score - std_score, color='gray', linestyle='--', 
               label=f'Mean - Std = {mean_score - std_score:.3f}')
    
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{class_name} Pairs Score Distribution')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def load_test_pairs_detailed(input_file="test_pairs_all.txt"):
    """Load test pairs from detailed CSV file"""
    test_pairs = []
    labels = []
    
    print(f"Loading test pairs from {input_file}...")
    
    with open(input_file, 'r') as f:
        # Skip header
        next(f)
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                test_pairs.append((parts[0], parts[1]))
                labels.append(int(parts[2]))
    
    return test_pairs, labels

def filter_pairs_by_class(test_pairs, labels, class_type):
    """Filter pairs to keep only genuine or forged pairs"""
    filtered_pairs = []
    filtered_labels = []
    
    target_label = 1 if class_type == "genuine" else 0
    
    for pair, label in zip(test_pairs, labels):
        if label == target_label:
            filtered_pairs.append(pair)
            filtered_labels.append(label)
    
    return filtered_pairs, filtered_labels

if __name__ == "__main__":
    # Load your model
    model_path = path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Generate or load test pairs
    base_directory = r"c:\Users\osho\Documents\University\Sem 4\Internship\SignatureMatching"
    
    print("Choose test pairs to use:")
    print("1. Use 10,000 random sample pairs (recommended for quick testing)")
    print("2. Use 10,000 balanced pairs (equal genuine/forged)")
    print("3. Use ALL pairs (very slow)")
    print("4. Generate new sample")
    print("5. Genuine pairs ONLY (score distribution analysis)")
    print("6. Forged pairs ONLY (score distribution analysis)")
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == "4":
        # Generate new sample
        print("\nGenerating new sample...")
        from generate_test_pairs import generate_random_sample_pairs, save_test_pairs_detailed
        
        sample_size = int(input("Enter sample size (default 10000): ") or "10000")
        test_pairs, labels = generate_random_sample_pairs(base_directory, sample_size)
        # save_test_pairs_detailed(test_pairs, labels, f"test_pairs_random_{sample_size}.txt")
        
    elif choice == "3":
        # Use ALL pairs (warning about time)
        if os.path.exists("test_pairs_all.txt"):
            print("Loading ALL test pairs (this may take a while)...")
            test_pairs, labels = load_test_pairs_detailed("test_pairs_all.txt")
        else:
            print("Generating ALL test pairs...")
            from generate_test_pairs import generate_all_test_pairs, save_test_pairs_detailed
            test_pairs, labels = generate_all_test_pairs(base_directory)
            # save_test_pairs_detailed(test_pairs, labels, "test_pairs_all.txt")
            
    elif choice == "2":
        # Use balanced sample
        if os.path.exists("test_pairs_balanced_10k.txt"):
            print("Loading balanced test pairs...")
            test_pairs, labels = load_test_pairs_detailed("test_pairs_balanced_10k.txt")
        else:
            print("Generating balanced test pairs...")
            from generate_test_pairs import generate_balanced_sample_pairs, save_test_pairs_detailed
            test_pairs, labels = generate_balanced_sample_pairs(base_directory, 10000)
            save_test_pairs_detailed(test_pairs, labels, "test_pairs_balanced_10k.txt")
    
    elif choice == "5" or choice == "6":
        # Genuine or Forged only
        class_type = "genuine" if choice == "5" else "forged"
        
        # First load a dataset with both classes
        if os.path.exists("test_pairs_random_10k.txt"):
            print("Loading random test pairs...")
            all_pairs, all_labels = load_test_pairs_detailed("test_pairs_random_10k.txt")
        else:
            print("Generating random test pairs...")
            from generate_test_pairs import generate_random_sample_pairs, save_test_pairs_detailed
            all_pairs, all_labels = generate_random_sample_pairs(base_directory, 10000)
        
        # Filter to get only the desired class
        test_pairs, labels = filter_pairs_by_class(all_pairs, all_labels, class_type)
        print(f"Filtered to {len(test_pairs)} {class_type} pairs")
        
    else:  # Default choice == "1"
        # Use random sample (default)
        if os.path.exists("test_pairs_random_10k.txt"):
            print("Loading random test pairs...")
            test_pairs, labels = load_test_pairs_detailed("test_pairs_random_10k.txt")
        else:
            print("Generating random test pairs...")
            from generate_test_pairs import generate_random_sample_pairs, save_test_pairs_detailed
            test_pairs, labels = generate_random_sample_pairs(base_directory, 10000)
            # save_test_pairs_detailed(test_pairs, labels, "test_pairs_random_10k.txt")
    
    print(f"\nDataset Summary:")
    print(f"Total test pairs: {len(test_pairs)}")
    print(f"Genuine pairs: {sum(labels)}")
    print(f"Forged pairs: {len(labels) - sum(labels)}")
    if len(labels) > 0:
        print(f"Genuine ratio: {sum(labels)/len(labels)*100:.1f}%")
    
    # Ask for confirmation if using large dataset
    if len(test_pairs) > 50000:
        confirm = input(f"\nWarning: You have {len(test_pairs)} pairs. This will take a long time.\nContinue? (y/n): ")
        if confirm.lower() != 'y':
            print("Exiting...")
            exit()
    
    # Check if we have both classes for threshold calculation
    unique_labels = set(labels)
    
    if len(unique_labels) == 1:
        # Single class - just show score distribution
        class_name = "Genuine" if labels[0] == 1 else "Forged"
        print(f"\nAnalyzing {class_name.lower()} pairs only...")
        scores, labels_array = calculate_score_distribution(model, test_pairs, labels, transform, device)
        plot_single_class_distribution(scores, labels_array, class_name)
        
    else:
        # Both classes - calculate optimal threshold
        print(f"\nCalculating optimal threshold for {len(test_pairs)} pairs...")
        threshold, metrics, scores, labels_array, fpr, tpr = calculate_optimal_threshold(
            model, test_pairs, labels, transform, device
        )
        
        print("\nOptimal Threshold Analysis:")
        print(f"Threshold: {threshold:.6f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"AUC: {metrics['auc']:.3f}")
        
        # Plot analysis
        plot_threshold_analysis(scores, labels_array, fpr, tpr, threshold, metrics)
        
        # Save threshold to file for use in main application
        # with open("optimal_threshold.txt", "w") as f:
        #     f.write(f"{threshold:.6f}\n")
        #     f.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
        #     f.write(f"Precision: {metrics['precision']:.3f}\n")
        #     f.write(f"Recall: {metrics['recall']:.3f}\n")
        #     f.write(f"F1-Score: {metrics['f1_score']:.3f}\n")
        #     f.write(f"AUC: {metrics['auc']:.3f}\n")
        #     f.write(f"Dataset_size: {len(test_pairs)}\n")
        
        # print(f"\nOptimal threshold saved to optimal_threshold.txt")