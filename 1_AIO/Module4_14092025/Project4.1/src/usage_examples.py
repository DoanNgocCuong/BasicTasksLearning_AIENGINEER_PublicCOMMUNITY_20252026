#!/usr/bin/env python3
# Simple test script for the optimized classifier

from optimized_hierarchical_classifier import HierarchicalTextClassifier

def main():
    print("Testing Optimized Hierarchical Text Classifier")
    print("=" * 50)

    # Initialize
    classifier = HierarchicalTextClassifier(max_features=500, random_state=42)

    # Sample data
    data = [
        {"text": "quantum physics theory", "categories": {"Science": ["Physics"]}},
        {"text": "calculus mathematics", "categories": {"Science": ["Math"]}},
        {"text": "artificial intelligence", "categories": {"Technology": ["AI"]}},
    ]

    # Train
    classifier.fit(data)

    # Test
    test_case = [{"text": "physics concepts", "categories": {"Science": ["Math"]}}]
    results = classifier.compare_approaches(test_case)

    print("Results:")
    print("Your Approach:", results['your_approach'])
    print("HiClass Approach:", results['hiclass_approach'])

if __name__ == "__main__":
    main()