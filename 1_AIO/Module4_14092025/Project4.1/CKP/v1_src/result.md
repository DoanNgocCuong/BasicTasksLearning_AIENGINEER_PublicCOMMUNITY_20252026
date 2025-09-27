```
θ69° [ubuntu@mgc-dev2-3090:~/cuong_dn/xongxoa/codeCuong] <venv> $ python main.py
Testing Optimized Hierarchical Text Classifier with data.json
============================================================
Arguments: samples=None, test_samples=5, max_features=10000, threshold=0.5
Data file: data.json
Loading data from: data.json
Loaded 29994 records
Successfully converted 29994 records

Splitting data into train/test sets...
Train samples: 23995
Test samples: 5999

Training classifier on 23995 samples...
Training completed!

Testing on 5 sample(s) from test set...

Results:
Your Approach:
  f1_macro_parent: 0.0505050505050505
  f1_macro_per_parent: {'cond-mat': 0.0, 'q-bio': 0.0, 'hep': 0.0, 'nucl': 0.0, 'q-fin': 0.0, 'cs': 0.0, 'astro-ph': 0.0, 'nlin': 0.0, 'math': 0.0, 'astro': 0.0, 'gr': 1.0, 'eess': 0.0, 'quant': 1.0, 'econ': 0.0, 'stat': 0.0, 'physics': 0.0, 'cond': 0.0, 'alg-geom': 1.0, 'adap-org': 1.0, 'chao-dyn': 1.0, 'cmp-lg': 1.0, 'patt-sol': 1.0, 'q-alg': 1.0, 'chem-ph': 1.0, 'funct-an': 1.0, 'solv-int': 1.0, 'supr-con': 1.0, 'mtrl-th': 1.0, 'atom-ph': 1.0, 'dg-ga': 1.0, 'comp-gas': 1.0, 'plasm-ph': 1.0, 'bayes-an': 1.0}
  f1_macro_children_overall: 0.5454545454545454

HiClass Approach:
  hierarchical_precision: 1.0
  hierarchical_recall: 0.16
  hierarchical_f1: 0.2758620689655173

Sample predictions (showing first 5):

Sample 1:
Text: Optimization of Lattice Boltzmann Simulations on Heterogeneous Computers   High-performance computin...
True categories: {'cs': ['cs.DC']}
Predicted categories: {}

Sample 2:
Text: Demographic inference using genetic data from a single individual:
  separating population size vari...
True categories: {'q-bio': ['q-bio.PE'], 'stat': ['stat.AP']}
Predicted categories: {'q-bio': []}

Sample 3:
Text: Preventing dataset shift from breaking machine-learning biomarkers   Machine learning brings the hop...
True categories: {'cs': ['cs.LG'], 'math': ['math.ST'], 'q-bio': ['q-bio.QM'], 'stat': ['stat.TH']}
Predicted categories: {}

Sample 4:
Text: Composite Consensus-Building Process: Permissible Meeting Analysis and
  Compromise Choice Explorati...
True categories: {'cs': ['cs.GT'], 'econ': ['econ.TH']}
Predicted categories: {}

Sample 5:
Text: An algebraic method of classification of S-integrable discrete models   A method of classification o...
True categories: {'nlin': ['nlin.SI']}
Predicted categories: {'nlin': []}
θ68° [ubuntu@mgc-dev2-3090:~/cuong_dn/xongxoa/codeCuong] <venv> 22m27s $ 
```
