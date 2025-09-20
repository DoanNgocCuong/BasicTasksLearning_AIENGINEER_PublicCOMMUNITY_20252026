# multilabel_hierarchical_classifier_v2.py
"""
Multi-Label Hierarchical Text Classifier - Format 2 (Nested Structure)
Clear parent-child relationships: {parent: [child1, child2, ...]}
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any, Optional
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, jaccard_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class MultiLabelHierarchicalClassifier:
    """
    Multi-Label Hierarchical Text Classifier với Nested Format
    
    Data Format:
    {
        "text": "student analyzes business data",
        "categories": {
            "Business": ["Economics", "Finance"],
            "Science": ["DataAnalysis"]
        }
    }
    """
    
    def __init__(self, 
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 base_classifier = None,
                 random_state: int = 42):
        """
        Initialize classifier
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
            base_classifier: Base classifier (default: RandomForest)
            random_state: Random state for reproducibility
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        
        # Initialize base classifier
        if base_classifier is None:
            base_classifier = RandomForestClassifier(
                n_estimators=100, 
                random_state=random_state,
                n_jobs=-1
            )
        
        # Multi-output classifiers
        self.parent_classifier = MultiOutputClassifier(base_classifier)
        self.child_classifier = MultiOutputClassifier(base_classifier)
        self.hierarchical_classifier = MultiOutputClassifier(base_classifier)
        
        # Label encoders
        self.mlb_parent = MultiLabelBinarizer()
        self.mlb_child = MultiLabelBinarizer()
        
        # Hierarchy mapping
        self.parent_to_children = {}  # parent -> list of possible children
        self.child_to_parents = {}   # child -> list of possible parents
        self.all_parent_categories = set()
        self.all_child_categories = set()
        
        # Training state
        self.is_fitted = False
        self.feature_names = None
        
    def create_sample_data(self, save_to: str = None) -> List[Dict]:
        """
        Create sample data in nested format
        
        Returns:
            List of {text: str, categories: {parent: [children]}}
        """
        sample_data = [
            {
                "text": "student analyzes business data and economic market trends",
                "categories": {
                    "Business": ["Economics", "Finance"],
                    "Science": ["DataAnalysis"]
                }
            },
            {
                "text": "student studies mathematics programming and computer algorithms",
                "categories": {
                    "Science": ["Math"],
                    "Technology": ["Programming", "ComputerScience"]
                }
            },
            {
                "text": "student writes creative poetry and literature analysis",
                "categories": {
                    "Arts": ["Poetry", "Literature", "Writing"]
                }
            },
            {
                "text": "student researches biology chemistry and environmental science",
                "categories": {
                    "Science": ["Biology", "Chemistry", "Environment"]
                }
            },
            {
                "text": "student creates digital art and user interface design",
                "categories": {
                    "Arts": ["DigitalArt", "Design"],
                    "Technology": ["UserInterface"]
                }
            },
            {
                "text": "student develops artificial intelligence and machine learning models",
                "categories": {
                    "Technology": ["AI", "MachineLearning", "Programming"],
                    "Science": ["Research"]
                }
            },
            {
                "text": "student studies quantum physics and theoretical research",
                "categories": {
                    "Science": ["Physics", "QuantumMechanics", "Research"]
                }
            },
            {
                "text": "student writes historical analysis and cultural studies papers",
                "categories": {
                    "Humanities": ["History", "Culture", "Writing"]
                }
            },
            {
                "text": "student builds web applications and mobile development projects",
                "categories": {
                    "Technology": ["WebDevelopment", "MobileDev", "Programming"]
                }
            },
            {
                "text": "student composes music and studies audio engineering technology",
                "categories": {
                    "Arts": ["Music", "Composition"],
                    "Technology": ["AudioEngineering"]
                }
            },
            {
                "text": "student investigates psychology neuroscience and cognitive research",
                "categories": {
                    "Science": ["Psychology", "Neuroscience", "Research"]
                }
            },
            {
                "text": "student practices business management and entrepreneurship strategies",
                "categories": {
                    "Business": ["Management", "Entrepreneurship", "Strategy"]
                }
            },
            {
                "text": "student explores philosophy ethics and critical thinking methods",
                "categories": {
                    "Humanities": ["Philosophy", "Ethics", "CriticalThinking"]
                }
            },
            {
                "text": "student designs mechanical engineering and robotics systems",
                "categories": {
                    "Technology": ["Engineering", "Robotics", "Design"],
                    "Science": ["Engineering"]
                }
            },
            {
                "text": "student teaches education pedagogy and curriculum development",
                "categories": {
                    "Education": ["Pedagogy", "Curriculum", "Teaching"]
                }
            }
        ]
        
        if save_to:
            with open(save_to, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Sample data saved to {save_to}")
        
        print(f"✅ Created {len(sample_data)} sample documents")
        print(f"📊 Sample format:")
        print(json.dumps(sample_data[0], indent=2))
        
        return sample_data
    
    def load_data_from_json(self, file_path: str) -> List[Dict]:
        """
        Load data from JSON file in nested format
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of {text: str, categories: {parent: [children]}}
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} samples from {file_path}")
        print(f"📊 Sample: '{data[0]['text'][:50]}...'")
        print(f"📊 Categories: {data[0]['categories']}")
        
        return data
    
    def save_data_to_json(self, data: List[Dict], file_path: str):
        """Save data to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Data saved to {file_path}")
    
    def load_data_from_csv(self, file_path: str,
                          text_col: str = 'text',
                          categories_col: str = 'categories') -> List[Dict]:
        """
        Load data from CSV file (categories column contains JSON string)
        
        CSV Format:
        text,categories
        "student studies math","{""Science"": [""Math"", ""Statistics""]}"
        """
        df = pd.read_csv(file_path)
        
        data = []
        for _, row in df.iterrows():
            text = row[text_col]
            categories_str = row[categories_col]
            
            # Parse JSON string
            if isinstance(categories_str, str):
                categories = json.loads(categories_str)
            else:
                categories = {}
            
            data.append({
                'text': text,
                'categories': categories
            })
        
        print(f"✅ Loaded {len(data)} samples from CSV")
        return data
    
    def convert_to_csv(self, data: List[Dict], file_path: str):
        """
        Convert nested data to CSV format
        """
        csv_data = []
        for item in data:
            csv_data.append({
                'text': item['text'],
                'categories': json.dumps(item['categories'])
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(file_path, index=False)
        print(f"💾 Data converted and saved to {file_path}")
    
    def _build_hierarchy_mapping(self, data: List[Dict]):
        """
        Build parent-child mapping from data
        """
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
        
        print(f"📊 Hierarchy built:")
        print(f"   Parents: {len(self.all_parent_categories)} - {sorted(self.all_parent_categories)}")
        print(f"   Children: {len(self.all_child_categories)} - {sorted(self.all_child_categories)}")
    
    def _extract_labels_from_data(self, data: List[Dict]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """
        Extract texts and labels from nested data
        """
        texts = []
        parent_labels = []
        child_labels = []
        
        for item in data:
            texts.append(item['text'])
            
            # Extract parent labels
            parents = list(item['categories'].keys())
            parent_labels.append(parents)
            
            # Extract all child labels
            children = []
            for parent, child_list in item['categories'].items():
                children.extend(child_list)
            child_labels.append(children)
        
        return texts, parent_labels, child_labels
    
    def fit(self, data: List[Dict], validation_split: float = 0.2):
        """
        Fit the classifier with nested data
        
        Args:
            data: List of {text: str, categories: {parent: [children]}}
            validation_split: Fraction of data for validation
        """
        print("🚀 Training Multi-Label Hierarchical Classifier (Nested Format)...")
        
        # Build hierarchy mapping
        self._build_hierarchy_mapping(data)
        
        # Extract labels
        texts, parent_labels, child_labels = self._extract_labels_from_data(data)
        
        # Vectorize texts
        print("   📝 Vectorizing texts...")
        X = self.vectorizer.fit_transform(texts).toarray()
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Encode labels
        print("   🏷️ Encoding labels...")
        y_parent = self.mlb_parent.fit_transform(parent_labels)
        y_child = self.mlb_child.fit_transform(child_labels)
        
        print(f"   📊 Parent classes: {len(self.mlb_parent.classes_)} - {list(self.mlb_parent.classes_)}")
        print(f"   📊 Child classes: {len(self.mlb_child.classes_)} - {list(self.mlb_child.classes_)}")
        
        # Split data for validation
        if validation_split > 0:
            X_train, X_val, y_p_train, y_p_val, y_c_train, y_c_val = train_test_split(
                X, y_parent, y_child, 
                test_size=validation_split, 
                random_state=self.random_state
            )
        else:
            X_train, X_val = X, None
            y_p_train, y_p_val = y_parent, None
            y_c_train, y_c_val = y_child, None
        
        # Train Parent classifier
        print("   🎯 Training Parent classifier...")
        self.parent_classifier.fit(X_train, y_p_train)
        
        # Train Child classifier (independent)
        print("   🎯 Training Child classifier...")
        self.child_classifier.fit(X_train, y_c_train)
        
        # Train hierarchical classifier (children depend on parents)
        print("   🔗 Training Hierarchical classifier...")
        parent_pred_train = self.parent_classifier.predict(X_train)
        X_hierarchical = np.hstack([X_train, parent_pred_train])
        self.hierarchical_classifier.fit(X_hierarchical, y_c_train)
        
        self.is_fitted = True
        
        # Validation if requested
        if validation_split > 0:
            print(f"\n   📈 Validation Results:")
            self._evaluate_on_validation(X_val, y_p_val, y_c_val)
        
        print("   ✅ Training completed!")
    
    def _evaluate_on_validation(self, X_val, y_p_val, y_c_val):
        """Internal validation evaluation"""
        # Independent predictions
        pred_p_ind, pred_c_ind = self._predict_binary(X_val, method='independent')
        
        # Hierarchical predictions  
        pred_p_hier, pred_c_hier = self._predict_binary(X_val, method='hierarchical')
        
        # Calculate metrics
        metrics = {
            'Independent Parent Jaccard': jaccard_score(y_p_val, pred_p_ind, average='samples'),
            'Independent Child Jaccard': jaccard_score(y_c_val, pred_c_ind, average='samples'),
            'Hierarchical Parent Jaccard': jaccard_score(y_p_val, pred_p_hier, average='samples'),
            'Hierarchical Child Jaccard': jaccard_score(y_c_val, pred_c_hier, average='samples'),
        }
        
        for metric, score in metrics.items():
            print(f"      {metric}: {score:.3f}")
    
    def _predict_binary(self, X, method: str = 'hierarchical'):
        """Internal method for binary predictions"""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")
        
        if method == 'hierarchical':
            pred_parent = self.parent_classifier.predict(X)
            X_hierarchical = np.hstack([X, pred_parent])
            pred_child = self.hierarchical_classifier.predict(X_hierarchical)
        else:  # independent
            pred_parent = self.parent_classifier.predict(X)
            pred_child = self.child_classifier.predict(X)
        
        return pred_parent, pred_child
    
    def predict(self, texts: List[str], method: str = 'hierarchical') -> List[Dict[str, List[str]]]:
        """
        Predict categories for new texts
        
        Args:
            texts: List of text documents to classify
            method: 'hierarchical' or 'independent'
            
        Returns:
            List of {parent: [children]} dictionaries
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first. Call fit() first.")
        
        # Vectorize texts
        X = self.vectorizer.transform(texts).toarray()
        
        # Get binary predictions
        pred_parent, pred_child = self._predict_binary(X, method)
        
        # Decode to labels
        parent_labels = self.mlb_parent.inverse_transform(pred_parent)
        child_labels = self.mlb_child.inverse_transform(pred_child)
        
        # Format results in nested structure
        results = []
        for i in range(len(texts)):
            predicted_parents = list(parent_labels[i])
            predicted_children = list(child_labels[i])
            
            # Create nested structure
            categories = {}
            
            if method == 'hierarchical':
                # Filter children based on predicted parents and hierarchy
                for parent in predicted_parents:
                    valid_children = []
                    for child in predicted_children:
                        if parent in self.child_to_parents.get(child, []):
                            valid_children.append(child)
                    
                    if valid_children:
                        categories[parent] = valid_children
                    elif parent in predicted_parents:
                        # Parent predicted but no valid children
                        categories[parent] = []
            else:
                # Independent: group children under their possible parents
                for parent in predicted_parents:
                    categories[parent] = []
                
                for child in predicted_children:
                    possible_parents = self.child_to_parents.get(child, [])
                    assigned = False
                    
                    for parent in possible_parents:
                        if parent in predicted_parents:
                            if parent not in categories:
                                categories[parent] = []
                            categories[parent].append(child)
                            assigned = True
                            break
                    
                    # If child's parent not predicted, assign to first possible parent
                    if not assigned and possible_parents:
                        first_parent = possible_parents[0]
                        if first_parent not in categories:
                            categories[first_parent] = []
                        categories[first_parent].append(child)
            
            results.append(categories)
        
        return results
    
    def predict_with_probabilities(self, texts: List[str]) -> List[Dict]:
        """
        Predict with probability scores
        
        Returns:
            List of {categories: {parent: [children]}, probabilities: {...}}
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")
        
        X = self.vectorizer.transform(texts).toarray()
        
        # Get predictions
        predictions = self.predict(texts, method='hierarchical')
        
        # Get probabilities
        parent_proba = self.parent_classifier.predict_proba(X)
        child_proba = self.child_classifier.predict_proba(X)
        
        results = []
        for i in range(len(texts)):
            # Extract probabilities
            parent_prob_dict = {}
            for j, class_name in enumerate(self.mlb_parent.classes_):
                prob = parent_proba[j][i][1] if len(parent_proba[j][i]) > 1 else parent_proba[j][i][0]
                parent_prob_dict[class_name] = round(prob, 3)
            
            child_prob_dict = {}
            for j, class_name in enumerate(self.mlb_child.classes_):
                prob = child_proba[j][i][1] if len(child_proba[j][i]) > 1 else child_proba[j][i][0]
                child_prob_dict[class_name] = round(prob, 3)
            
            results.append({
                'text': texts[i],
                'categories': predictions[i],
                'probabilities': {
                    'parents': parent_prob_dict,
                    'children': child_prob_dict
                }
            })
        
        return results
    
    def get_hierarchy_info(self) -> Dict:
        """Get information about the learned hierarchy"""
        if not self.is_fitted:
            return {"status": "Not fitted"}
        
        return {
            "status": "Fitted",
            "hierarchy": {
                "parent_to_children": self.parent_to_children,
                "child_to_parents": self.child_to_parents
            },
            "classes": {
                "parents": list(self.mlb_parent.classes_),
                "children": list(self.mlb_child.classes_)
            },
            "stats": {
                "num_parents": len(self.all_parent_categories),
                "num_children": len(self.all_child_categories),
                "num_features": len(self.feature_names) if self.feature_names is not None else 0
            }
        }
    
    def save(self, file_path: str):
        """Save the trained model to file"""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Model saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'MultiLabelHierarchicalClassifier':
        """Load a trained model from file"""
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"📂 Model loaded from {file_path}")
        return model

# =============================================================================
# DEMO & USAGE FUNCTIONS
# =============================================================================

def quick_demo():
    """Quick demonstration of the nested format classifier"""
    print("🚀 Quick Demo - Nested Format Multi-Label Hierarchical Classifier")
    print("=" * 70)
    
    # Initialize classifier
    classifier = MultiLabelHierarchicalClassifier(max_features=1000)
    
    # Create sample data
    data = classifier.create_sample_data('sample_nested_data.json')
    
    # Train
    classifier.fit(data, validation_split=0.2)
    
    # Test predictions
    test_texts = [
        "student develops machine learning algorithms and data science projects",
        "student writes poetry and studies ancient literature",
        "student analyzes stock market trends and economic policies",
        "student creates digital art and designs user interfaces"
    ]
    
    print(f"\n🔍 Test Predictions (Hierarchical):")
    print("-" * 50)
    
    predictions = classifier.predict(test_texts, method='hierarchical')
    
    for text, pred in zip(test_texts, predictions):
        print(f"\nText: '{text[:50]}...'")
        for parent, children in pred.items():
            print(f"  {parent}: {children}")
    
    # Test with probabilities
    print(f"\n📊 Detailed Prediction with Probabilities:")
    print("-" * 50)
    
    detailed_results = classifier.predict_with_probabilities([test_texts[0]])
    result = detailed_results[0]
    
    print(f"Text: '{result['text'][:50]}...'")
    print(f"Categories:")
    for parent, children in result['categories'].items():
        print(f"  {parent}: {children}")
    
    print(f"Top Parent Probabilities:")
    sorted_parents = sorted(result['probabilities']['parents'].items(), 
                           key=lambda x: x[1], reverse=True)
    for parent, prob in sorted_parents[:5]:
        print(f"  {parent}: {prob}")
    
    # Show hierarchy info
    hierarchy_info = classifier.get_hierarchy_info()
    print(f"\n📊 Hierarchy Summary:")
    print(f"  Parents: {hierarchy_info['stats']['num_parents']}")
    print(f"  Children: {hierarchy_info['stats']['num_children']}")
    print(f"  Features: {hierarchy_info['stats']['num_features']}")
    
    # Save model
    classifier.save('nested_format_model.pkl')
    
    print(f"\n✅ Demo completed! Files created:")
    print("  • sample_nested_data.json")
    print("  • nested_format_model.pkl")

def create_your_data_template():
    """Create a template for users to fill with their data"""
    template = [
        {
            "text": "Replace this with your first text document",
            "categories": {
                "YourParentCategory1": ["ChildCategory1", "ChildCategory2"],
                "YourParentCategory2": ["ChildCategory3"]
            }
        },
        {
            "text": "Replace this with your second text document", 
            "categories": {
                "YourParentCategory1": ["ChildCategory1"],
                "YourParentCategory3": ["ChildCategory4", "ChildCategory5"]
            }
        }
    ]
    
    with open('your_data_template.json', 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print("📝 Template created: your_data_template.json")
    print("📋 Edit this file with your data, then load with:")
    print("   data = classifier.load_data_from_json('your_data_template.json')")

if __name__ == "__main__":
    # Run quick demo
    quick_demo()
    
    # Create template for users
    create_your_data_template()