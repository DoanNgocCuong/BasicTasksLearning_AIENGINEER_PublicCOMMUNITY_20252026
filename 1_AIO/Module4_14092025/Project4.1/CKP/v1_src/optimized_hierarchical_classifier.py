# Optimized Multi-Label Hierarchical Text Classifier - v3 (Streamlined)
"""
Optimized version focusing only on key evaluation metrics:
- Your approach: {'f1_macro_parent': float, 'f1_macro_per_parent': {...}, 'f1_macro_children_overall': float}  
- HiClass approach: {'hierarchical_precision': float, 'hierarchical_recall': float, 'hierarchical_f1': float}

Architecture: Text â†’ Parent Classifier â†’ [Text + Parent] â†’ Child Classifier
"""

import pandas as pd
import numpy as np
import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any, Optional
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, jaccard_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

class HierarchicalTextClassifier:
    """
    Optimized Multi-Label Hierarchical Text Classifier

    Architecture:
    Text â†’ Parent Classifier â†’ Parent Labels
    [Text + Parent Labels] â†’ Child Classifier â†’ Child Labels

    Key Features:
    - Streamlined evaluation with focused metrics
    - Minimal verbose output
    - Core hierarchical architecture maintained
    """

    def __init__(self, 
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 base_classifier = None,
                 random_state: int = 42):
        """Initialize hierarchical classifier"""
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state

        # Initialize vectorizer with better preprocessing
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True  # Apply sublinear tf scaling
        )

        # Initialize base classifier with class_weight for imbalanced data
        if base_classifier is None:
            base_classifier = RandomForestClassifier(
                n_estimators=200,  # Increase for better performance
                max_depth=20,  # Limit depth to prevent overfitting
                min_samples_split=5,  # Prevent overfitting
                min_samples_leaf=2,  # Prevent overfitting
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced',  # Handle class imbalance
                max_features='sqrt'  # Better for high-dimensional data
            )

        # Hierarchical classifiers
        self.parent_classifier = MultiOutputClassifier(base_classifier)
        self.child_classifier = MultiOutputClassifier(base_classifier)

        # Label encoders
        self.mlb_parent = MultiLabelBinarizer()
        self.mlb_child = MultiLabelBinarizer()

        # Hierarchy mapping
        self.parent_to_children = {}
        self.child_to_parents = {}
        self.all_parent_categories = set()
        self.all_child_categories = set()

        self.is_fitted = False
        self.feature_names = None

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better feature extraction"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        
        return text

    def _build_hierarchy_mapping(self, data: List[Dict]):
        """Build parent-child mapping from data"""
        self.parent_to_children = defaultdict(set)
        self.child_to_parents = defaultdict(set)
        self.all_parent_categories = set()
        self.all_child_categories = set()

        for item in data:
            categories = item['categories']
            for parent, children in categories.items():
                self.all_parent_categories.add(parent)

                for child in children:
                    self.all_child_categories.add(child)
                    self.parent_to_children[parent].add(child)
                    self.child_to_parents[child].add(parent)

        # Convert sets to lists
        self.parent_to_children = {k: list(v) for k, v in self.parent_to_children.items()}
        self.child_to_parents = {k: list(v) for k, v in self.child_to_parents.items()}

    def _extract_labels_from_data(self, data: List[Dict]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """Extract texts and labels from nested data"""
        texts = []
        parent_labels = []
        child_labels = []

        for item in data:
            texts.append(self._preprocess_text(item['text']))

            # Extract parent labels
            parents = list(item['categories'].keys())
            parent_labels.append(parents)

            # Extract all child labels
            children = []
            for parent, child_list in item['categories'].items():
                children.extend(child_list)
            child_labels.append(children)

        return texts, parent_labels, child_labels

    def fit(self, data: List[Dict], validation_split: float = 0.1):
        """
        Fit the hierarchical classifier

        Architecture:
        1. Train Parent Classifier: Text â†' Parent Labels
        2. Train Child Classifier: [Text + Parent Labels] â†' Child Labels
        """
        # Validate input data
        if not data or len(data) == 0:
            raise ValueError("Training data cannot be empty")
        
        # Check data format
        for i, item in enumerate(data[:5]):  # Check first 5 items
            if not isinstance(item, dict) or 'text' not in item or 'categories' not in item:
                raise ValueError(f"Invalid data format at index {i}. Expected dict with 'text' and 'categories' keys")
        
        print(f"Training on {len(data)} samples...")
        
        # Build hierarchy mapping
        self._build_hierarchy_mapping(data)

        # Extract labels
        texts, parent_labels, child_labels = self._extract_labels_from_data(data)

        # Vectorize texts
        X = self.vectorizer.fit_transform(texts).toarray()
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Encode labels
        y_parent = self.mlb_parent.fit_transform(parent_labels)
        y_child = self.mlb_child.fit_transform(child_labels)

        # Split data for validation
        if validation_split > 0:
            X_train, X_val, y_p_train, y_p_val, y_c_train, y_c_val = train_test_split(
                X, y_parent, y_child, 
                test_size=validation_split, 
                random_state=self.random_state
            )
        else:
            X_train = X
            y_p_train = y_parent
            y_c_train = y_child

        # Step 1: Train Parent classifier
        print("Training parent classifier...")
        self.parent_classifier.fit(X_train, y_p_train)
        print(f"Parent classifier trained on {X_train.shape[0]} samples with {y_p_train.shape[1]} parent classes")

        # Step 2: Train Child classifier with parent information
        # Use true parent labels for training (not predictions to avoid data leakage)
        print("Training child classifier...")
        X_hierarchical = np.hstack([X_train, y_p_train])
        self.child_classifier.fit(X_hierarchical, y_c_train)
        print(f"Child classifier trained on {X_hierarchical.shape[0]} samples with {y_c_train.shape[1]} child classes")

        self.is_fitted = True

    def _predict_binary(self, X, threshold=0.2):
        """Internal method for binary predictions using hierarchical approach"""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")

        # Step 1: Predict parents with probability threshold
        parent_proba = self.parent_classifier.predict_proba(X)
        # Fix: Correctly extract positive class probabilities for multi-label
        try:
            parent_proba_array = np.array([proba[:, 1] for proba in parent_proba]).T
        except IndexError:
            # Handle case where some classes might not have positive examples
            parent_proba_array = np.array([proba[:, 0] if proba.shape[1] == 1 else proba[:, 1] for proba in parent_proba]).T
        pred_parent = (parent_proba_array > threshold).astype(int)

        # Step 2: Predict children with parent context
        X_hierarchical = np.hstack([X, pred_parent])
        child_proba = self.child_classifier.predict_proba(X_hierarchical)
        # Fix: Correctly extract positive class probabilities for multi-label
        try:
            child_proba_array = np.array([proba[:, 1] for proba in child_proba]).T
        except IndexError:
            # Handle case where some classes might not have positive examples
            child_proba_array = np.array([proba[:, 0] if proba.shape[1] == 1 else proba[:, 1] for proba in child_proba]).T
        pred_child = (child_proba_array > threshold).astype(int)

        return pred_parent, pred_child

    def predict(self, texts: List[str], threshold: float = 0.2) -> List[Dict[str, List[str]]]:
        """
        Predict categories for new texts using hierarchical approach
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first. Call fit() first.")

        # Preprocess and vectorize texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(processed_texts).toarray()

        # Get binary predictions
        pred_parent, pred_child = self._predict_binary(X, threshold)

        # Decode to labels
        parent_labels = self.mlb_parent.inverse_transform(pred_parent)
        child_labels = self.mlb_child.inverse_transform(pred_child)

        # Format results in nested structure with hierarchy filtering
        results = []
        for i in range(len(texts)):
            predicted_parents = list(parent_labels[i])
            predicted_children = list(child_labels[i])

            # Create nested structure with hierarchy constraints
            categories = {}

            # Filter children based on predicted parents and hierarchy
            for parent in predicted_parents:
                valid_children = []
                for child in predicted_children:
                    if parent in self.child_to_parents.get(child, []):
                        valid_children.append(child)

                # Fix: Only add parent if it has valid children
                if valid_children:
                    categories[parent] = valid_children

            results.append(categories)

        return results

    # =============================================================================
    # CORE EVALUATION METHODS - STREAMLINED
    # =============================================================================

    def evaluate_your_approach(self, test_data: List[Dict], threshold: float = 0.2) -> Dict:
        """
        Your Approach Evaluation - Returns only key metrics

        Returns:
            {
                'f1_macro_parent': float,
                'f1_macro_per_parent': {parent: f1_score},
                'f1_macro_children_overall': float
            }
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")

        # Extract test data
        texts, parent_labels, child_labels = self._extract_labels_from_data(test_data)

        # Vectorize and get true labels
        X_test = self.vectorizer.transform(texts).toarray()
        y_true_parent = self.mlb_parent.transform(parent_labels)
        y_true_child = self.mlb_child.transform(child_labels)

        # Get predictions
        pred_parent, pred_child = self._predict_binary(X_test, threshold)

        # 1. F1 macro parent
        f1_macro_parent = f1_score(y_true_parent, pred_parent, average='macro', zero_division=0)

        # 2. F1 macro per parent (for children of each parent)
        f1_macro_per_parent = {}

        for parent_name in self.parent_to_children.keys():
            parent_children = self.parent_to_children[parent_name]

            if len(parent_children) == 0:
                f1_macro_per_parent[parent_name] = 0.0
                continue

            # Get indices of children for this parent
            child_indices = []
            for child in parent_children:
                if child in self.mlb_child.classes_:
                    idx = list(self.mlb_child.classes_).index(child)
                    child_indices.append(idx)

            if len(child_indices) == 0:
                f1_macro_per_parent[parent_name] = 0.0
                continue

            # Extract predictions for children of this parent
            y_true_parent_children = y_true_child[:, child_indices]
            y_pred_parent_children = pred_child[:, child_indices]

            # Calculate F1 macro for these children
            if y_true_parent_children.shape[1] > 0:
                f1_macro_per_parent[parent_name] = f1_score(
                    y_true_parent_children, 
                    y_pred_parent_children, 
                    average='macro', 
                    zero_division=0
                )
            else:
                f1_macro_per_parent[parent_name] = 0.0

        # 3. Overall children F1 (macro average of per-parent F1s)
        f1_macro_children_overall = np.mean(list(f1_macro_per_parent.values())) if f1_macro_per_parent else 0.0

        return {
            'f1_macro_parent': f1_macro_parent,
            'f1_macro_per_parent': f1_macro_per_parent,
            'f1_macro_children_overall': f1_macro_children_overall
        }

    def _expand_with_ancestors(self, nested_sample: Dict) -> set:
        """
        Expand nested sample with ancestors following HiClass approach

        Input: {"Science": ["Physics"]}
        Output: {"ROOT", "ROOTâ†’Science", "ROOTâ†’Scienceâ†’Physics"}
        """
        expanded = set()

        for parent, children in nested_sample.items():
            # Add ROOT
            expanded.add("ROOT")

            # Add ROOT â†’ Parent path
            expanded.add(f"ROOTâ†’{parent}")

            # Add ROOT â†’ Parent â†’ Child paths
            for child in children:
                expanded.add(f"ROOTâ†’{parent}â†’{child}")

        return expanded

    def evaluate_hiclass_approach(self, test_data: List[Dict], threshold: float = 0.2) -> Dict:
        """
        HiClass Approach Evaluation - Returns only key metrics

        Returns:
            {
                'hierarchical_precision': float,
                'hierarchical_recall': float, 
                'hierarchical_f1': float
            }
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")

        # Get predictions
        texts = [item['text'] for item in test_data]
        predictions = self.predict(texts, threshold)

        # Convert to expanded ancestor sets
        total_intersection = 0
        total_predicted = 0
        total_true = 0

        for true_sample, pred_sample in zip(test_data, predictions):
            true_categories = true_sample['categories']

            # Expand with ancestors
            true_expanded = self._expand_with_ancestors(true_categories)
            pred_expanded = self._expand_with_ancestors(pred_sample)

            # Calculate intersection
            intersection = true_expanded & pred_expanded

            # Accumulate for overall metrics
            total_intersection += len(intersection)
            total_predicted += len(pred_expanded)
            total_true += len(true_expanded)

        # Calculate hierarchical metrics
        h_precision = total_intersection / total_predicted if total_predicted > 0 else 0
        h_recall = total_intersection / total_true if total_true > 0 else 0
        h_f1 = 2 * h_precision * h_recall / (h_precision + h_recall) if (h_precision + h_recall) > 0 else 0

        return {
            'hierarchical_precision': h_precision,
            'hierarchical_recall': h_recall,
            'hierarchical_f1': h_f1
        }

    def compare_approaches(self, test_data: List[Dict], threshold: float = 0.2) -> Dict:
        """
        Compare both evaluation approaches

        Returns:
            {
                'your_approach': {...},
                'hiclass_approach': {...}
            }
        """
        your_metrics = self.evaluate_your_approach(test_data, threshold)
        hiclass_metrics = self.evaluate_hiclass_approach(test_data, threshold)

        return {
            'your_approach': your_metrics,
            'hiclass_approach': hiclass_metrics
        }

    # Utility methods
    def save_model(self, file_path: str):
        """Save the trained model to file"""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, file_path: str) -> 'HierarchicalTextClassifier':
        """Load a trained model from file"""
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def demo_example():
    """
    Single focused example showing both evaluation approaches
    """
    # Initialize classifier
    classifier = HierarchicalTextClassifier(max_features=1000, random_state=42)

    # Training data
    training_data = [
        {"text": "quantum mechanics theoretical physics", "categories": {"Science": ["Physics"]}},
        {"text": "calculus differential equations mathematics", "categories": {"Science": ["Math"]}},
        {"text": "machine learning artificial intelligence", "categories": {"Technology": ["AI"]}},
        {"text": "poetry creative writing literature", "categories": {"Arts": ["Poetry"]}},
        {"text": "business strategy management", "categories": {"Business": ["Strategy"]}},
    ]

    # Train
    classifier.fit(training_data)

    # Test case: Science â†’ Physics vs Science â†’ Math
    test_case = [
        {
            "text": "advanced physics concepts and quantum theory", 
            "categories": {"Science": ["Math"]}  # True: Math, Will predict: Physics
        }
    ]

    # Compare approaches
    results = classifier.compare_approaches(test_case)

    print("Your Approach:")
    print(results['your_approach'])
    print("\nHiClass Approach:")
    print(results['hiclass_approach'])

    return classifier, results

if __name__ == "__main__":
    demo_example()