# usage_examples_format2.py
"""
Usage Examples - Format 2 (Nested Structure)
Ví dụ sử dụng với nested format: {parent: [children]}
Clear hierarchical relationships - không bao giờ bị confused!
"""

from multilabel_hierarchical_classifier import MultiLabelHierarchicalClassifier
import json

# =============================================================================
# 1. QUICKSTART - CHỈ 4 DÒNG CODE 🚀
# =============================================================================

def quickstart_example():
    """Cách nhanh nhất để bắt đầu"""
    print("🚀 QUICKSTART - Chỉ 4 dòng code:")
    print("-" * 40)
    
    classifier = MultiLabelHierarchicalClassifier()
    data = classifier.create_sample_data()
    classifier.fit(data)
    results = classifier.predict(["student develops AI and studies machine learning"])
    
    print(f"✅ Result: {results[0]}")

# =============================================================================
# 2. TẠO DỮ LIỆU CỦA BẠN 📝
# =============================================================================

def create_your_own_data():
    """Ví dụ tạo data của riêng bạn"""
    print("\n📝 TẠO DỮ LIỆU CỦA BẠN:")
    print("-" * 30)
    
    # Dữ liệu của bạn theo format nested
    your_data = [
        {
            "text": "student analyzes business data and economic trends",
            "categories": {
                "Business": ["Economics", "Finance"],  # Rõ ràng: Business có Economics và Finance
                "Science": ["DataAnalysis"]            # Science có DataAnalysis
            }
        },
        {
            "text": "student studies mathematics programming and algorithms",
            "categories": {
                "Science": ["Math"],                   # Science có Math
                "Technology": ["Programming", "ComputerScience"]  # Technology có Programming và CS
            }
        },
        {
            "text": "student writes poetry and creates digital artwork",
            "categories": {
                "Arts": ["Poetry", "Literature"],      # Arts có Poetry và Literature
                "Technology": ["DigitalArt"]           # Technology có DigitalArt
            }
        },
        {
            "text": "student researches machine learning and artificial intelligence",
            "categories": {
                "Technology": ["AI", "MachineLearning", "Programming"],  # Technology có AI, ML, Programming
                "Science": ["Research"]                # Science có Research
            }
        },
        {
            "text": "student manages startup business and develops strategy",
            "categories": {
                "Business": ["Management", "Strategy", "Entrepreneurship"]  # Business có 3 children
            }
        }
    ]
    
    # Save data
    with open('my_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(your_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Đã tạo 'my_training_data.json'")
    print("📊 Format mẫu:")
    print(json.dumps(your_data[0], indent=2))
    
    print(f"\n🎯 Advantages of Format 2:")
    print("• 100% rõ ràng parent nào có children nào")
    print("• Không bao giờ confused như format cũ")  
    print("• Dễ validate và debug")
    print("• Intuitive cho humans")
    
    return your_data

# =============================================================================
# 3. TRAINING TỪ FILE JSON 🏋️
# =============================================================================

def train_from_your_data():
    """Train từ data của bạn"""
    print("\n🏋️ TRAINING TỪ DATA CỦA BẠN:")
    print("-" * 35)
    
    # Tạo data mẫu
    your_data = create_your_own_data()
    
    # Initialize classifier
    classifier = MultiLabelHierarchicalClassifier(
        max_features=2000,
        ngram_range=(1, 2)
    )
    
    # Train từ data
    classifier.fit(your_data, validation_split=0.0)  # Không validation vì data ít
    
    # Test prediction
    test_texts = [
        "student develops artificial intelligence applications",
        "student writes creative fiction and poetry",
        "student analyzes financial markets and investment data"
    ]
    
    predictions = classifier.predict(test_texts)
    
    print(f"\n🔍 Test Results:")
    for text, pred in zip(test_texts, predictions):
        print(f"\nText: '{text}'")
        for parent, children in pred.items():
            print(f"  {parent}: {children}")
    
    # Save model
    classifier.save('my_trained_model.pkl')
    print(f"\n💾 Model saved as 'my_trained_model.pkl'")

# =============================================================================
# 4. LOAD MODEL ĐÃ TRAIN 📂
# =============================================================================

def use_trained_model():
    """Sử dụng model đã train"""
    print("\n📂 SỬ DỤNG MODEL ĐÃ TRAIN:")
    print("-" * 30)
    
    try:
        # Load model
        classifier = MultiLabelHierarchicalClassifier.load('my_trained_model.pkl')
        
        # Test với texts mới
        new_texts = [
            "student builds machine learning algorithms for data science",
            "student composes music and studies audio engineering",
            "student creates business plans and marketing strategies"
        ]
        
        predictions = classifier.predict(new_texts, method='hierarchical')
        
        print("🔍 Predictions from loaded model:")
        for i, (text, pred) in enumerate(zip(new_texts, predictions)):
            print(f"\n{i+1}. '{text[:45]}...'")
            for parent, children in pred.items():
                print(f"   {parent}: {children}")
        
        # Show hierarchy info
        hierarchy = classifier.get_hierarchy_info()
        print(f"\n📊 Model Info:")
        print(f"   Parents: {hierarchy['stats']['num_parents']}")
        print(f"   Children: {hierarchy['stats']['num_children']}")
        print(f"   Learned hierarchy:")
        for parent, children in hierarchy['hierarchy']['parent_to_children'].items():
            print(f"     {parent} -> {children}")
        
    except FileNotFoundError:
        print("❌ Model file not found. Run train_from_your_data() first.")

# =============================================================================
# 5. SO SÁNH INDEPENDENT VS HIERARCHICAL 🔀
# =============================================================================

def compare_prediction_methods():
    """So sánh 2 phương pháp predict"""
    print("\n🔀 SO SÁNH INDEPENDENT VS HIERARCHICAL:")
    print("-" * 45)
    
    # Create and train classifier
    classifier = MultiLabelHierarchicalClassifier()
    data = classifier.create_sample_data()
    classifier.fit(data, validation_split=0.2)
    
    test_text = ["student researches artificial intelligence and develops neural networks"]
    
    # Independent prediction
    pred_independent = classifier.predict(test_text, method='independent')
    
    # Hierarchical prediction
    pred_hierarchical = classifier.predict(test_text, method='hierarchical')
    
    print(f"Text: '{test_text[0]}'")
    print(f"\n📊 Independent Method:")
    for parent, children in pred_independent[0].items():
        print(f"   {parent}: {children}")
    
    print(f"\n🔗 Hierarchical Method:")
    for parent, children in pred_hierarchical[0].items():
        print(f"   {parent}: {children}")
    
    print(f"\n💡 Explanation:")
    print("• Independent: Predict parents và children riêng biệt")
    print("• Hierarchical: Children depend on predicted parents")
    print("• Hierarchical thường accurate hơn vì respect hierarchy")

# =============================================================================
# 6. DETAILED ANALYSIS 🔬
# =============================================================================

def detailed_analysis():
    """Phân tích chi tiết với probabilities"""
    print("\n🔬 DETAILED ANALYSIS:")
    print("-" * 25)
    
    # Train classifier
    classifier = MultiLabelHierarchicalClassifier()
    data = classifier.create_sample_data()
    classifier.fit(data)
    
    test_text = "student develops machine learning models and analyzes big data"
    
    # Get detailed predictions with probabilities
    detailed_results = classifier.predict_with_probabilities([test_text])
    result = detailed_results[0]
    
    print(f"Text: '{test_text}'")
    
    print(f"\n🎯 Predicted Categories:")
    for parent, children in result['categories'].items():
        print(f"   {parent}: {children}")
    
    print(f"\n📈 Parent Probabilities (Top 5):")
    parent_probs = result['probabilities']['parents']
    sorted_parents = sorted(parent_probs.items(), key=lambda x: x[1], reverse=True)
    for parent, prob in sorted_parents[:5]:
        print(f"   {parent}: {prob}")
    
    print(f"\n📈 Child Probabilities (Top 8):")
    child_probs = result['probabilities']['children']
    sorted_children = sorted(child_probs.items(), key=lambda x: x[1], reverse=True)
    for child, prob in sorted_children[:8]:
        print(f"   {child}: {prob}")

# =============================================================================
# 7. PRODUCTION PIPELINE 🏭
# =============================================================================

def production_pipeline():
    """Pipeline cho production use"""
    print("\n🏭 PRODUCTION PIPELINE:")
    print("-" * 25)
    
    class ProductionClassifier:
        def __init__(self):
            self.classifier = None
        
        def train_from_file(self, json_file: str):
            """Train từ JSON file"""
            print(f"🏋️ Training from {json_file}...")
            
            self.classifier = MultiLabelHierarchicalClassifier(
                max_features=5000,
                ngram_range=(1, 3)
            )
            
            data = self.classifier.load_data_from_json(json_file)
            self.classifier.fit(data, validation_split=0.2)
            
            # Auto save
            model_name = f"production_model.pkl"
            self.classifier.save(model_name)
            print(f"💾 Model saved as {model_name}")
        
        def classify_text(self, text: str, with_details: bool = False):
            """Classify single text"""
            if not self.classifier:
                raise ValueError("Model not trained yet")
            
            if with_details:
                detailed = self.classifier.predict_with_probabilities([text])
                return detailed[0]
            else:
                prediction = self.classifier.predict([text])
                return prediction[0]
        
        def classify_batch(self, texts: list):
            """Classify multiple texts"""
            if not self.classifier:
                raise ValueError("Model not trained yet")
            
            return self.classifier.predict(texts)
        
        def get_model_info(self):
            """Get model information"""
            if not self.classifier:
                return "Model not trained"
            
            return self.classifier.get_hierarchy_info()
    
    # Demo production pipeline
    prod = ProductionClassifier()
    
    # Create sample data file
    classifier_temp = MultiLabelHierarchicalClassifier()
    sample_data = classifier_temp.create_sample_data('production_sample.json')
    
    # Train
    prod.train_from_file('production_sample.json')
    
    # Test single classification
    result = prod.classify_text(
        "student builds AI systems and machine learning applications",
        with_details=True
    )
    
    print(f"\n🎯 Single Classification:")
    print(f"Text: '{result['text'][:50]}...'")
    print(f"Categories:")
    for parent, children in result['categories'].items():
        print(f"  {parent}: {children}")
    
    # Test batch classification
    batch_texts = [
        "student studies quantum physics research",
        "student creates digital art and animations", 
        "student develops business strategies"
    ]
    
    batch_results = prod.classify_batch(batch_texts)
    
    print(f"\n📦 Batch Classification:")
    for text, pred in zip(batch_texts, batch_results):
        print(f"\n'{text}':")
        for parent, children in pred.items():
            print(f"  {parent}: {children}")
    
    # Show model info
    info = prod.get_model_info()
    print(f"\n📊 Production Model Info:")
    print(f"   Status: {info['status']}")
    print(f"   Parents: {len(info['classes']['parents'])}")
    print(f"   Children: {len(info['classes']['children'])}")

# =============================================================================
# 8. FORMAT COMPARISON 📋
# =============================================================================

def format_comparison():
    """So sánh format cũ vs format mới"""
    print("\n📋 FORMAT COMPARISON:")
    print("=" * 30)
    
    print("""
🔴 FORMAT CŨ (Confusing):
{
  'text': 'student analyzes business data and economic trends',
  'level_0_labels': 'Business|Science', 
  'level_1_labels': 'Economics|DataAnalysis|Finance'
}

❌ Problems:
• Economics thuộc Business hay Science?
• DataAnalysis thuộc Business hay Science? 
• Finance thuộc Business hay Science?
• CONFUSED! 😵

✅ FORMAT MỚI (Clear):
{
  "text": "student analyzes business data and economic trends",
  "categories": {
    "Business": ["Economics", "Finance"],    # 100% clear!
    "Science": ["DataAnalysis"]              # 100% clear!
  }
}

✅ Benefits:
• 0% confusion về parent-child relationships
• Intuitive và human-readable
• Dễ validate data
• Dễ debug khi có lỗi
• Professional data format
""")

# =============================================================================
# 9. QUICK REFERENCE 📚
# =============================================================================

def quick_reference():
    """Quick reference guide"""
    print("\n📚 QUICK REFERENCE:")
    print("=" * 20)
    
    reference = '''
🎯 NESTED FORMAT STRUCTURE:
{
  "text": "your text here",
  "categories": {
    "ParentCategory1": ["Child1", "Child2"],
    "ParentCategory2": ["Child3", "Child4", "Child5"]
  }
}

🚀 BASIC USAGE:
classifier = MultiLabelHierarchicalClassifier()
data = classifier.load_data_from_json('your_data.json')
classifier.fit(data)
results = classifier.predict(["text to classify"])

🎯 PREDICTION RESULT:
{
  "ParentCategory1": ["Child1", "Child2"],
  "ParentCategory2": ["Child3"]
}

📂 FILE OPERATIONS:
• classifier.create_sample_data('sample.json')    # Create sample
• classifier.load_data_from_json('data.json')     # Load from JSON
• classifier.save('model.pkl')                    # Save model
• classifier.load('model.pkl')                    # Load model

🔍 PREDICTION METHODS:
• method='hierarchical'  # Children depend on parents (recommended)
• method='independent'   # Parents and children predicted separately

📊 ANALYSIS:
• classifier.predict_with_probabilities(texts)    # Get probabilities
• classifier.get_hierarchy_info()                 # Get model info
'''
    
    print(reference)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎯 MULTI-LABEL HIERARCHICAL CLASSIFIER - FORMAT 2 (NESTED)")
    print("=" * 65)
    print("Clear parent-child relationships - No more confusion! 🎉")
    
    # Run all examples
    quickstart_example()
    train_from_your_data()
    use_trained_model()
    compare_prediction_methods()
    detailed_analysis()
    production_pipeline()
    format_comparison()
    quick_reference()
    
    print(f"\n🎉 All examples completed!")
    print("📁 Files created:")
    print("   • my_training_data.json")
    print("   • my_trained_model.pkl")
    print("   • production_sample.json")
    print("   • production_model.pkl")
    
    print(f"\n✨ Your turn!")
    print("1. Edit 'my_training_data.json' với data của bạn")
    print("2. Run train_from_your_data() để train model")
    print("3. Enjoy clear hierarchical classification! 🚀")