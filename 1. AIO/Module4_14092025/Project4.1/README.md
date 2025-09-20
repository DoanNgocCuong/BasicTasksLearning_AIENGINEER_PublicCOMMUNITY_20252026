# Hierarchical Classification Guide

A comprehensive guide for hierarchical text classification using modern Python libraries.

## Table of Contents

- [Overview](#overview)
- [Approaches Comparison](#approaches-comparison)
- [Quick Start](#quick-start)
- [HiClass vs Custom Implementation](#hiclass-vs-custom-implementation)
- [FAISS Role Explained](#faiss-role-explained)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Performance Comparison](#performance-comparison)
- [When to Use What](#when-to-use-what)
- [Contributing](#contributing)

## Overview

Hierarchical classification organizes labels in a tree or DAG structure, allowing for more intuitive and accurate multi-class prediction compared to flat classification.

### Example Hierarchy
```
Root
├── Science
│   ├── Math
│   ├── Physics
│   └── Chemistry
├── Arts
│   ├── Literature
│   └── Design
└── ComputerScience
    ├── AI
    └── Programming
```

## Approaches Comparison

| Approach | Development Time | Code Lines | Reliability | Performance | Maintenance |
|----------|------------------|------------|-------------|-------------|-------------|
| **Custom Implementation** | Weeks | 200+ | DIY Testing | Manual Optimization | Self-maintained |
| **HiClass Library** | Hours | 10-15 | Battle-tested | Optimized | Community |

**Recommendation: Use HiClass for production systems.**

## Quick Start

### Option 1: HiClass (Recommended)

```bash
pip install hiclass
```

```python
from hiclass import LocalClassifierPerParentNode
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Data preparation
X_texts = [
    "student likes math and programming",
    "student loves physics and chemistry",
    "student enjoys literature and history"
]

y_hierarchical = [
    ["Science", "Math"],
    ["Science", "Physics"], 
    ["Arts", "Literature"]
]

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_vectors = vectorizer.fit_transform(X_texts).toarray()

# Train hierarchical classifier
classifier = LocalClassifierPerParentNode(
    local_classifier=RandomForestClassifier(n_estimators=100)
)
classifier.fit(X_vectors, y_hierarchical)

# Predict
test_texts = ["student studies calculus and algorithms"]
test_vectors = vectorizer.transform(test_texts).toarray()
predictions = classifier.predict(test_vectors)

print(f"Prediction: {predictions[0]}")  # ['Science', 'Math']
```

### Option 2: Custom Implementation

```python
# 200+ lines of custom hierarchy logic + FAISS integration
# See examples/custom_implementation.py for full code
```

## HiClass vs Custom Implementation

### HiClass Architecture

```
┌─────────────────────────────────────────┐
│            HiClass (Wrapper)            │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐ │
│  │    3 HIERARCHICAL PATTERNS:        │ │
│  │  • LCPN (Per Node)                 │ │
│  │  • LCPPN (Per Parent Node)         │ │
│  │  • LCPL (Per Level)                │ │
│  └─────────────────────────────────────┘ │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐ │
│  │   USER-CHOSEN BASE CLASSIFIERS:    │ │
│  │  • RandomForestClassifier          │ │
│  │  • LogisticRegression             │ │
│  │  • SVM, XGBoost, etc.             │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Key Differences

| Feature | Custom Implementation | HiClass |
|---------|----------------------|---------|
| **Hierarchy Logic** | Manual implementation | Built-in, tested |
| **Base Models** | Limited to what you code | Any sklearn-compatible |
| **Evaluation Metrics** | Manual implementation | Built-in hierarchical metrics |
| **Parallel Training** | Manual implementation | Automatic |
| **Error Handling** | DIY | Robust, tested |
| **Documentation** | Self-written | Comprehensive |

## FAISS Role Explained

### Common Misconception

❌ **Wrong Question:** "FAISS vs RandomForest vs SVM - which is better?"

✅ **Correct Understanding:** These serve completely different purposes!

### What Each Does

```
┌────────────────────┬─────────────────────┬─────────────────────┐
│      ASPECT        │       FAISS         │     ML MODELS       │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Purpose            │ Vector similarity   │ Classification      │
│ Input              │ Query vector        │ Features            │
│ Output             │ Nearest neighbors   │ Predicted class     │
│ Training           │ No training needed  │ Requires training   │
│ Use Case           │ Search, retrieval   │ Prediction          │
└────────────────────┴─────────────────────┴─────────────────────┘
```