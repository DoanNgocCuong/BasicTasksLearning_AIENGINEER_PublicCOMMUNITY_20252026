#!/usr/bin/env python3
# Test script for the optimized classifier using data.json

import json
import os
import argparse
from sklearn.model_selection import train_test_split
from optimized_hierarchical_classifier import HierarchicalTextClassifier

def load_data_json(file_path):
    """Load and convert data.json to the expected format"""
    print(f"Loading data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} records")
    
    # Convert to expected format
    converted_data = []
    for item in raw_data:
        # Combine title and abstract as text
        text = f"{item.get('title', '')} {item.get('abstract', '')}"
        
        # Parse json_categories
        categories_str = item.get('json_categories', '{}')
        try:
            categories = json.loads(categories_str)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse categories for item: {item.get('title', 'Unknown')[:50]}...")
            continue
        
        converted_data.append({
            "text": text,
            "categories": categories
        })
    
    print(f"Successfully converted {len(converted_data)} records")
    return converted_data

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Hierarchical Text Classifier with data.json')
    parser.add_argument('--samples', '-s', type=int, default=None, 
                       help='Number of samples to use for training (default: use all)')
    parser.add_argument('--test-samples', '-t', type=int, default=5,
                       help='Number of samples to use for testing (default: 5, use -1 for all test data)')
    parser.add_argument('--max-features', '-f', type=int, default=10000,
                       help='Maximum number of features for TF-IDF (default: 10000)')
    parser.add_argument('--data-file', '-d', type=str, default='data.json',
                       help='Path to data.json file (default: data.json)')
    parser.add_argument('--threshold', type=float, default=0.2,
                       help='Probability threshold for predictions (default: 0.2)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print("Testing Optimized Hierarchical Text Classifier with data.json")
    print("=" * 60)
    print(f"Arguments: samples={args.samples}, test_samples={args.test_samples}, max_features={args.max_features}, threshold={args.threshold}")
    print(f"Data file: {args.data_file}")

    # Load data from JSON file
    if not os.path.exists(args.data_file):
        print(f"Error: {args.data_file} not found in current directory")
        return
    
    try:
        data = load_data_json(args.data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if len(data) == 0:
        print("No valid data found")
        return
    
    # Limit samples if specified
    if args.samples is not None:
        if args.samples > len(data):
            print(f"Warning: Requested {args.samples} samples but only {len(data)} available. Using all {len(data)} samples.")
            data = data
        else:
            data = data[:args.samples]
            print(f"Using {len(data)} samples as requested")
    
    # Split data into train and test sets
    print(f"\nSplitting data into train/test sets...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Initialize classifier
    classifier = HierarchicalTextClassifier(max_features=args.max_features, random_state=42)
    
    print(f"\nTraining classifier on {len(train_data)} samples...")
    
    # Train on train data only
    classifier.fit(train_data)
    
    print("Training completed!")
    
    # Test on test data (use all test data or limit to specified number)
    if args.test_samples == -1:  # Use all test data
        test_samples = test_data
        print(f"\nTesting on ALL {len(test_samples)} samples from test set...")
    else:
        test_count = min(args.test_samples, len(test_data))
        test_samples = test_data[:test_count]
        print(f"\nTesting on {test_count} sample(s) from test set...")
    
    # Always evaluate on FULL test set for accurate metrics
    print(f"\nEvaluating on FULL test set ({len(test_data)} samples)...")
    results = classifier.compare_approaches(test_data, threshold=args.threshold)
    
    print("\nResults:")
    print("Your Approach:")
    for key, value in results['your_approach'].items():
        print(f"  {key}: {value}")
    
    print("\nHiClass Approach:")
    for key, value in results['hiclass_approach'].items():
        print(f"  {key}: {value}")
    
    # Show some predictions (limit to first 5 for readability)
    show_count = min(5, len(test_samples))
    print(f"\nSample predictions (showing first {show_count}):")
    test_texts = [item['text'][:100] + "..." for item in test_samples[:show_count]]
    predictions = classifier.predict([item['text'] for item in test_samples[:show_count]], threshold=args.threshold)
    
    for i, (text, pred, true) in enumerate(zip(test_texts, predictions, test_samples[:show_count])):
        print(f"\nSample {i+1}:")
        print(f"Text: {text}")
        print(f"True categories: {true['categories']}")
        print(f"Predicted categories: {pred}")

if __name__ == "__main__":
    main()
    
"""    
# Use all samples (default behavior)
python main.py

# Use only 1000 samples for training
python main.py --samples 1000

# Use 500 samples for training and 10 for testing
python main.py -s 500 -t 10

# Use 2000 samples with 5000 max features
python main.py -s 2000 -f 5000

# Use different data file
python main.py -d my_data.json -s 1000

"""