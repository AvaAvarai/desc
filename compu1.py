import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

def find_pure_intervals_single_feature(X, y, feature_idx, feature_name, class_names, min_purity=0.90, min_samples=2):
    """
    Find pure intervals in a single feature that classify cases.
    Uses a sliding window approach to find optimal intervals.
    Returns intervals with their boundaries, predicted class, purity, and coverage.
    """
    intervals = []
    seen_intervals = set()  # To avoid duplicates
    
    # Get the feature values and corresponding classes
    feature_values = X[:, feature_idx]
    sorted_indices = np.argsort(feature_values)
    sorted_values = feature_values[sorted_indices]
    sorted_classes = y[sorted_indices]
    
    # Try different thresholds to find pure intervals
    unique_values = np.unique(sorted_values)
    
    # For each potential split point, evaluate intervals
    for i in range(len(unique_values)):
        threshold = unique_values[i]
        
        # Interval 1: <= threshold
        mask1 = feature_values <= threshold
        if np.sum(mask1) >= min_samples:
            classes_in_interval = y[mask1]
            class_counts = Counter(classes_in_interval)
            dominant_class = class_counts.most_common(1)[0][0]
            purity = class_counts[dominant_class] / len(classes_in_interval)
            
            interval_key = (np.min(feature_values[mask1]), threshold, dominant_class)
            if purity >= min_purity and interval_key not in seen_intervals:
                seen_intervals.add(interval_key)
                intervals.append({
                    'feature': feature_name,
                    'feature_idx': feature_idx,
                    'condition': f"{feature_name} <= {threshold:.2f}",
                    'interval': (np.min(feature_values[mask1]), threshold),
                    'class': class_names[dominant_class],
                    'class_idx': dominant_class,
                    'purity': purity,
                    'coverage': len(classes_in_interval),
                    'num_features': 1
                })
        
        # Interval 2: > threshold (only if not last value)
        if i < len(unique_values) - 1:
            mask2 = feature_values > threshold
            if np.sum(mask2) >= min_samples:
                classes_in_interval = y[mask2]
                class_counts = Counter(classes_in_interval)
                dominant_class = class_counts.most_common(1)[0][0]
                purity = class_counts[dominant_class] / len(classes_in_interval)
                
                interval_key = (threshold, np.max(feature_values[mask2]), dominant_class)
                if purity >= min_purity and interval_key not in seen_intervals:
                    seen_intervals.add(interval_key)
                    intervals.append({
                        'feature': feature_name,
                        'feature_idx': feature_idx,
                        'condition': f"{feature_name} > {threshold:.2f}",
                        'interval': (threshold, np.max(feature_values[mask2])),
                        'class': class_names[dominant_class],
                        'class_idx': dominant_class,
                        'purity': purity,
                        'coverage': len(classes_in_interval),
                        'num_features': 1
                    })
    
    # Try range intervals (between two thresholds) for better granularity
    for i in range(len(unique_values) - 1):
        for j in range(i + 2, min(i + 5, len(unique_values))):  # Limit to avoid too many combinations
            lower = unique_values[i]
            upper = unique_values[j]
            
            mask = (feature_values > lower) & (feature_values <= upper)
            if np.sum(mask) >= min_samples:
                classes_in_interval = y[mask]
                class_counts = Counter(classes_in_interval)
                dominant_class = class_counts.most_common(1)[0][0]
                purity = class_counts[dominant_class] / len(classes_in_interval)
                
                interval_key = (lower, upper, dominant_class)
                if purity >= min_purity and interval_key not in seen_intervals:
                    seen_intervals.add(interval_key)
                    intervals.append({
                        'feature': feature_name,
                        'feature_idx': feature_idx,
                        'condition': f"{lower:.2f} < {feature_name} <= {upper:.2f}",
                        'interval': (lower, upper),
                        'class': class_names[dominant_class],
                        'class_idx': dominant_class,
                        'purity': purity,
                        'coverage': len(classes_in_interval),
                        'num_features': 1
                    })
    
    return intervals

def combine_intervals(intervals, X, y, class_names, max_combinations=3):
    """
    Combine intervals from different features to form more specific rules.
    """
    combined_rules = []
    
    # Start with single-feature intervals
    for interval in intervals:
        combined_rules.append(interval)
    
    # Try combining 2-feature rules
    for i, interval1 in enumerate(intervals):
        for j, interval2 in enumerate(intervals[i+1:], i+1):
            if interval1['feature_idx'] != interval2['feature_idx']:
                # Check if combined rule improves classification
                mask1 = evaluate_condition(X, interval1['condition'], interval1['feature_idx'])
                mask2 = evaluate_condition(X, interval2['condition'], interval2['feature_idx'])
                combined_mask = mask1 & mask2
                
                if np.sum(combined_mask) >= 2:
                    classes_in_interval = y[combined_mask]
                    class_counts = Counter(classes_in_interval)
                    dominant_class = class_counts.most_common(1)[0][0]
                    purity = class_counts[dominant_class] / len(classes_in_interval)
                    
                    if purity >= 0.9:  # Slightly lower threshold for combined rules
                        combined_rules.append({
                            'feature': f"{interval1['feature']} AND {interval2['feature']}",
                            'feature_idx': (interval1['feature_idx'], interval2['feature_idx']),
                            'condition': f"{interval1['condition']} AND {interval2['condition']}",
                            'interval': (interval1['interval'], interval2['interval']),
                            'class': class_names[dominant_class],
                            'class_idx': dominant_class,
                            'purity': purity,
                            'coverage': len(classes_in_interval),
                            'num_features': 2
                        })
    
    return combined_rules

def evaluate_condition(X, condition, feature_idx):
    """
    Evaluate a condition on the dataset.
    """
    if isinstance(feature_idx, tuple):
        # Multi-feature condition - evaluate both parts
        parts = condition.split(' AND ')
        if len(parts) == 2:
            # Parse first condition
            cond1 = parts[0].strip()
            idx1 = feature_idx[0]
            mask1 = evaluate_single_condition(X, cond1, idx1)
            
            # Parse second condition
            cond2 = parts[1].strip()
            idx2 = feature_idx[1]
            mask2 = evaluate_single_condition(X, cond2, idx2)
            
            return mask1 & mask2
        return np.ones(X.shape[0], dtype=bool)
    else:
        return evaluate_single_condition(X, condition, feature_idx)

def evaluate_single_condition(X, condition, feature_idx):
    """
    Evaluate a single condition on one feature.
    """
    feature_values = X[:, feature_idx]
    feature_name = condition.split()[0] if ' ' in condition else None
    
    if '<=' in condition:
        threshold = float(condition.split('<=')[1].strip())
        return feature_values <= threshold
    elif '>' in condition and '<=' in condition and '<' in condition:
        # Range condition: "lower < feature <= upper"
        parts = condition.split()
        lower = float(parts[0])
        upper = float(parts[2])
        return (feature_values > lower) & (feature_values <= upper)
    elif '>' in condition and not '<=' in condition:
        threshold = float(condition.split('>')[1].strip())
        return feature_values > threshold
    else:
        return np.ones(X.shape[0], dtype=bool)

def find_most_informative_rules(rules, X_test, y_test, class_names):
    """
    Evaluate and rank rules by informativeness:
    - Coverage (how many cases they capture)
    - Purity/Accuracy (how pure the classification is)
    - Generalization (simpler rules preferred)
    """
    rule_scores = []
    
    for rule in rules:
        # Evaluate rule on test set if possible
        mask = evaluate_condition(X_test, rule['condition'], rule['feature_idx'])
        
        if np.sum(mask) > 0:
            test_classes = y_test[mask]
            test_class_counts = Counter(test_classes)
            test_dominant = test_class_counts.most_common(1)[0][0]
            test_accuracy = test_class_counts[test_dominant] / len(test_classes) if len(test_classes) > 0 else 0
        else:
            test_accuracy = 0
        
        # Use training accuracy (purity) as primary metric
        accuracy = rule['purity']
        coverage = rule['coverage']
        
        # Simplicity score (fewer features = simpler)
        simplicity = 1.0 / (1.0 + rule['num_features'])
        
        # Combined informativeness score
        # Weight: coverage (30%), accuracy (50%), simplicity (20%)
        score = (coverage * 0.3) + (accuracy * 100 * 0.5) + (simplicity * 100 * 0.2)
        
        rule_scores.append({
            'rule': rule['condition'],
            'class': rule['class'],
            'coverage': coverage,
            'purity': accuracy,
            'test_accuracy': test_accuracy if np.sum(mask) > 0 else None,
            'simplicity': simplicity,
            'score': score,
            'num_features': rule['num_features'],
            'interval': rule['interval']
        })
    
    # Sort by score (highest first)
    rule_scores.sort(key=lambda x: x['score'], reverse=True)
    
    return rule_scores

def print_rules(rules, top_n=20):
    """
    Print the classification rules in a readable format.
    """
    print("\n" + "="*80)
    print("CLASSIFICATION RULES FROM PURE INTERVALS")
    print("="*80)
    print(f"\nShowing top {min(top_n, len(rules))} most informative rules:\n")
    
    for i, rule_info in enumerate(rules[:top_n], 1):
        print(f"Rule {i}:")
        print(f"  IF {rule_info['rule']}")
        print(f"  THEN Class = {rule_info['class']}")
        print(f"  Interval: {rule_info['interval']}")
        print(f"  Coverage: {rule_info['coverage']} samples")
        print(f"  Purity: {rule_info['purity']:.2%}")
        if rule_info['test_accuracy'] is not None:
            print(f"  Test Accuracy: {rule_info['test_accuracy']:.2%}")
        print(f"  Features used: {rule_info['num_features']}")
        print(f"  Informativeness Score: {rule_info['score']:.2f}")
        print()

def main():
    # Load the Iris dataset
    print("Loading Fisher Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Classes: {', '.join(class_names)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Find pure intervals for each feature
    print("\nFinding pure intervals in attributes...")
    all_intervals = []
    
    for feature_idx, feature_name in enumerate(feature_names):
        print(f"  Analyzing {feature_name}...")
        intervals = find_pure_intervals_single_feature(
            X_train, y_train, feature_idx, feature_name, class_names,
            min_purity=0.90, min_samples=2
        )
        all_intervals.extend(intervals)
        print(f"    Found {len(intervals)} pure intervals")
    
    print(f"\nTotal single-feature intervals found: {len(all_intervals)}")
    
    # Combine intervals to form multi-feature rules
    print("\nCombining intervals to form classification rules...")
    combined_rules = combine_intervals(all_intervals, X_train, y_train, class_names)
    print(f"Total rules (including combinations): {len(combined_rules)}")
    
    # Find most informative rules
    print("\nEvaluating and ranking rules by informativeness...")
    informative_rules = find_most_informative_rules(combined_rules, X_test, y_test, class_names)
    
    # Print the most informative rules
    print_rules(informative_rules, top_n=20)
    
    # Evaluate overall model accuracy using top rules
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Use top rules to make predictions
    y_pred = np.zeros(len(X_test), dtype=int)
    covered = np.zeros(len(X_test), dtype=bool)
    
    # Apply rules in order of informativeness
    for rule_info in informative_rules:
        rule = [r for r in combined_rules if r['condition'] == rule_info['rule']][0]
        mask = evaluate_condition(X_test, rule['condition'], rule['feature_idx'])
        
        # Apply rule to uncovered samples
        apply_mask = mask & ~covered
        if np.sum(apply_mask) > 0:
            class_idx = np.where(class_names == rule['class'])[0][0]
            y_pred[apply_mask] = class_idx
            covered[apply_mask] = True
    
    # For uncovered samples, use majority class from training
    if np.sum(~covered) > 0:
        majority_class = np.bincount(y_train).argmax()
        y_pred[~covered] = majority_class
    
    accuracy = accuracy_score(y_test, y_pred)
    coverage_rate = np.sum(covered) / len(y_test)
    
    print(f"\nRule-based Model Accuracy: {accuracy:.2%}")
    print(f"Coverage: {coverage_rate:.2%} of test samples covered by rules")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Print summary statistics
    print("\n" + "="*80)
    print("RULE SUMMARY STATISTICS")
    print("="*80)
    print(f"Total number of rules found: {len(combined_rules)}")
    print(f"Single-feature rules: {sum(1 for r in combined_rules if r['num_features'] == 1)}")
    print(f"Multi-feature rules: {sum(1 for r in combined_rules if r['num_features'] > 1)}")
    print(f"Average coverage: {np.mean([r['coverage'] for r in informative_rules]):.1f} samples")
    print(f"Average purity: {np.mean([r['purity'] for r in informative_rules]):.2%}")
    
    # Print rules grouped by class
    print("\n" + "="*80)
    print("RULES GROUPED BY PREDICTED CLASS")
    print("="*80)
    for class_name in class_names:
        class_rules = [r for r in informative_rules if r['class'] == class_name]
        print(f"\n{class_name.upper()} ({len(class_rules)} rules):")
        for i, rule_info in enumerate(class_rules[:8], 1):  # Top 8 per class
            print(f"  {i}. IF {rule_info['rule']}")
            print(f"     Coverage: {rule_info['coverage']} samples, Purity: {rule_info['purity']:.2%}")

if __name__ == "__main__":
    main()
