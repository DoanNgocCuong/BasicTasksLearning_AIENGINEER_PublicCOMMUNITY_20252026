# Ref:
- https://aistudio.google.com/prompts/1Wx081g1dLhv2yotR0LhVG0D2Cji1t2pM

# Version 1
## **1. Mục Tiêu**

Mục tiêu của Version 1.0 là xây dựng và đánh giá một mô hình baseline đầu tiên cho bài toán phân loại chủ đề bài báo khoa học trên bộ dữ liệu ArXiv đã qua tiền xử lý. Các mục tiêu cụ thể bao gồm:
-   Xác thực tính hiệu quả của bộ dữ liệu `arxiv_perfectly_balanced.csv`.
-   Triển khai kiến trúc phân loại phân cấp hai tầng (Hierarchical Classification).
-   Sử dụng các mô hình học máy cổ điển (TF-IDF + LightGBM) để thiết lập một ngưỡng hiệu suất (baseline) có thể đo lường được.
-   Đánh giá chi tiết hiệu suất của mô hình ở cả hai tầng và phân tích các điểm cần cải thiện.

## **2. Kiến Trúc & Phương Pháp Thực Hiện**

Kiến trúc tổng thể được xây dựng theo một pipeline gồm 3 giai đoạn chính: Trích xuất Đặc trưng, Huấn luyện Mô hình Phân cấp, và Quy trình Dự đoán.

### **2.1. Trích Xuất Đặc Trưng (Feature Extraction)**

-   **Phương pháp:** Term Frequency-Inverse Document Frequency (TF-IDF).
-   **Chi tiết:**
    -   Sử dụng `TfidfVectorizer` của Scikit-learn.
    -   Giới hạn số lượng đặc trưng ở `max_features=5000` từ phổ biến nhất để cân bằng giữa hiệu suất và tốc độ tính toán.
    -   Loại bỏ các từ dừng (stop words) tiếng Anh.
-   **Kết quả:** Mỗi abstract được biểu diễn bằng một vector thưa (sparse vector) 5000 chiều.

### **2.2. Kiến Trúc Mô Hình Phân Cấp Hai Tầng**

Để xử lý cấu trúc cha-con của nhãn, chúng tôi đã triển khai một hệ thống gồm hai tầng mô hình:

#### **Tầng 1: Parent Classifier (Bộ phân loại Nhãn Cha)**
-   **Nhiệm vụ:** Dự đoán một hoặc nhiều trong số 17 nhãn cha chính (vd: `cs`, `math`, `hep`) từ abstract của bài báo.
-   **Mô hình:** `OneVsRestClassifier` kết hợp với `LGBMClassifier`.
-   **Xử lý Mất cân bằng:** Tham số `class_weight='balanced'` được kích hoạt trong `LGBMClassifier`. Đây là một bước cực kỳ quan trọng, giúp thuật toán tự động tăng trọng số cho các lớp cha thiểu số (`econ`, `cond`), buộc mô hình phải học chúng một cách công bằng.

#### **Tầng 2: Child Classifiers (Các bộ phân loại Nhãn Con)**
-   **Nhiệm vụ:** Với mỗi nhãn cha được dự đoán ở Tầng 1, một mô hình chuyên biệt ở Tầng 2 sẽ được kích hoạt để dự đoán các nhãn con cụ thể thuộc nhãn cha đó.
-   **Kiến trúc:** Một tập hợp gồm **15 mô hình con**, mỗi mô hình tương ứng với một nhãn cha có đủ dữ liệu để huấn luyện.
    -   *Ví dụ:* Nếu Tầng 1 dự đoán là `cs`, mô hình `cs_classifier` của Tầng 2 sẽ được dùng để dự đoán các nhãn con như `cs.AI`, `cs.CV`, `cs.LG`,...
-   **Mô hình:** Mỗi bộ phân loại con cũng là một `OneVsRestClassifier` với `LGBMClassifier`, cũng sử dụng `class_weight='balanced'`.

### **2.3. Quy Trình Huấn Luyện & Dự Đoán**

1.  **Huấn luyện:**
    -   Huấn luyện mô hình Tầng 1 trên toàn bộ tập train (23,995 mẫu) với 17 nhãn cha.
    -   Với mỗi nhãn cha, lọc ra các mẫu trong tập train thuộc về nhãn cha đó và huấn luyện một mô hình Tầng 2 tương ứng.
2.  **Dự đoán (trên tập test):**
    -   **Bước 1:** Đưa abstract vào mô hình Tầng 1 để nhận về các nhãn cha dự đoán (ví dụ: `['cs', 'math']`).
    -   **Bước 2:** Với mỗi nhãn cha dự đoán được, kích hoạt mô hình Tầng 2 tương ứng.
        -   `cs_classifier` sẽ dự đoán các nhãn con của `cs`.
        -   `math_classifier` sẽ dự đoán các nhãn con của `math`.
    -   **Bước 3:** Gộp tất cả các nhãn con dự đoán được từ các mô hình Tầng 2 để ra kết quả cuối cùng.

## **3. Kết Quả Thử Nghiệm (Version 1.0)**

| Tầng Đánh Giá | Metric | Giá Trị | Ghi Chú |
| :--- | :--- | :--- | :--- |
| **Tầng 1 (Nhãn Cha)** | **F1-Score (Weighted Avg)** | **0.6483** | **Metric chính**, phản ánh hiệu suất tổng thể có trọng số. |
| | F1-Score (Macro Avg) | 0.6474 | Cho thấy mô hình hoạt động tốt trên cả lớp đa số và thiểu số. |
| | F1-Score (Samples Avg) | 0.6396 | Hiệu suất trung bình trên từng mẫu, hữu ích cho đa nhãn. |
| **Toàn Hệ Thống (Nhãn Con)** | **F1-Score (Weighted Avg)** | **0.4047** | Phản ánh hiệu suất dự đoán nhãn con cuối cùng. |
| | F1-Score (Samples Avg) | 0.4142 | |
| | F1-Score (Macro Avg) | 0.2543 | **Rất thấp**, cho thấy mô hình cực kỳ khó khăn với các lớp con hiếm. |

## **4. Phân Tích & Đánh Giá**

### **4.1. Điểm Tích Cực**

-   **Chiến lược dữ liệu được xác thực:** Việc F1-macro và F1-weighted ở Tầng 1 gần như bằng nhau (chênh lệch chỉ 0.0009) khẳng định rằng chiến lược **cân bằng đơn/đa nhãn** kết hợp với `class_weight='balanced'` là hoàn toàn đúng đắn. Mô hình không bị thiên vị nặng về các lớp cha đa số.
-   **Thiết lập Baseline thành công:** Mô hình đã cung cấp một ngưỡng hiệu suất rõ ràng (F1 ~0.65 cho nhãn cha, ~0.40 cho nhãn con) để các phiên bản tương lai có thể so sánh và cải thiện.

### **4.2. Hạn Chế & Nguyên Nhân Hiệu Suất**

Kết quả hiện tại là một baseline tốt, nhưng chưa cao. Nguyên nhân không nằm ở khâu chuẩn bị dữ liệu mà đến từ các yếu tố sau:

1.  **Sụt giảm hiệu suất từ Tầng 1 -> Tầng 2:** F1-weighted giảm từ **0.65 xuống 0.40**. Đặc biệt, F1-macro giảm mạnh từ **0.65 xuống 0.25**, cho thấy nút thắt cổ chai nằm ở việc dự đoán các nhãn con. Đây là bài toán phân loại chi tiết (fine-grained) với hàng trăm lớp con, trong đó rất nhiều lớp có số lượng mẫu cực kỳ ít (vấn đề đuôi dài - long-tail problem), khiến mô hình không đủ dữ liệu để học.

2.  **Điểm mù ngữ nghĩa của TF-IDF:** `TF-IDF` chỉ đếm từ, không hiểu nghĩa. Nó không nhận ra rằng "machine learning" và "deep learning" có liên quan đến nhau. Đây là hạn chế lớn nhất về mặt trích xuất đặc trưng, ngăn mô hình "hiểu" sâu hơn về nội dung abstract.

3.  **Mô hình Baseline chưa được tinh chỉnh:** Các tham số của LightGBM (`n_estimators=100`) và TF-IDF (`max_features=5000`) đang ở mức cơ bản để chạy nhanh. Chúng chưa được tối ưu để đạt hiệu suất cao nhất.

## **5. Hướng Cải Thiện cho Version 2.0**

Nền tảng dữ liệu đã vững chắc. Lộ trình cải thiện cho phiên bản tiếp theo sẽ tập trung vào việc nâng cấp mô hình.

-   **Ưu tiên #1 (Tác động lớn nhất): Nâng cấp Feature Extraction.**
    -   **Thử nghiệm:** Thay thế TF-IDF bằng các mô hình nhúng từ có khả năng hiểu ngữ nghĩa như **SciBERT**. Đây là một mô hình Transformer đã được huấn luyện trước trên một kho văn bản khoa học khổng lồ, hứa hẹn sẽ mang lại sự cải thiện đột phá.

-   **Ưu tiên #2: Tinh chỉnh siêu tham số (Hyperparameter Tuning).**
    -   **Thử nghiệm:** Sử dụng các thư viện như Optuna hoặc Hyperopt để tự động tìm ra bộ tham số tốt nhất cho `LGBMClassifier` (ví dụ: `n_estimators`, `learning_rate`, `num_leaves`, ...).

-   **Ưu tiên #3 (Tùy chọn): Tối ưu hóa ngưỡng quyết định.**
    -   **Thử nghiệm:** Sau khi có dự đoán xác suất, tìm một ngưỡng quyết định (threshold) tối ưu cho mỗi nhãn thay vì dùng mặc định 0.5 để tối đa hóa F1-score.

## **6. Kết Luận Chung**

Version 1.0 đã thành công trong việc xây dựng một pipeline hoàn chỉnh và thiết lập một baseline hiệu suất đáng tin cậy. Phân tích đã chỉ ra rằng chiến lược chuẩn bị dữ liệu là đúng đắn và các điểm nghẽn về hiệu suất nằm ở khả năng trích xuất đặc trưng và sự tinh chỉnh của mô hình. Các bước tiếp theo sẽ tập trung vào việc giải quyết các điểm nghẽn này.

# Version 2
## Script:
```python
# ===================================================================
#                      VERSION 2.2: SPACY + FASTTEXT + OPTUNA
# ===================================================================
# Script này thay thế hoàn toàn NLTK bằng spaCy để giải quyết triệt để
# lỗi LookupError, trong khi vẫn giữ nguyên các cải tiến về ngữ nghĩa
# (FastText) và tối ưu hóa (Optuna).
# ===================================================================

# ### BƯỚC 0: CÀI ĐẶT CẦN THIẾT ###
# Chạy ô này TRƯỚC TIÊN trong Colab để cài đặt spaCy và tải mô hình.
# !python -m spacy download en_core_web_sm

# ===================================================================
# PHẦN 0: CÁC THƯ VIỆN CẦN THIẾT
# ===================================================================
# !pip install optuna
# !pip install gensim
# !python -m spacy download en_core_web_sm

print("🚀 Đang import các thư viện...")
import pandas as pd
import numpy as np
import ast
import pickle
import os
import json
from collections import Counter
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report
from tqdm.auto import tqdm
import optuna
import gensim.downloader
import spacy # ### V2.2 CẢI TIẾN: Thay thế NLTK bằng spaCy
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
print("✅ Import thư viện hoàn tất.")

# ===================================================================
# PHẦN 0B: CẤU HÌNH
# ===================================================================
N_TARGET_LABELS = 17
OPTUNA_N_TRIALS = 25
OPTUNA_TIMEOUT = 3600

LGBM_FIXED_PARAMS = {
    'device': 'gpu',
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'
}
print("⚡️ Đã kích hoạt chế độ huấn luyện GPU và cấu hình cho Version 2.2 (spaCy + FastText)!")

# ===================================================================
# PHẦN 1: TẢI DỮ LIỆU VÀ MÔ HÌNH NLP
# ===================================================================
print("\n🚀 [Bước 1/7] Tải dữ liệu và mô hình spaCy...")
# Tải mô hình spaCy nhỏ gọn, chỉ cần tokenizer và stopwords
print("   - Đang tải mô hình spaCy 'en_core_web_sm'...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
stop_words = nlp.Defaults.stop_words
print("   - Tải spaCy hoàn tất.")

FILE_PATH = "/content/drive/MyDrive/data/arxiv_perfectly_balanced.csv"
try:
    df = pd.read_csv(FILE_PATH)
    print(f"✅ Tải thành công file: '{FILE_PATH}' ({len(df):,} mẫu)")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file '{FILE_PATH}'. Hãy đảm bảo bạn đã kết nối Google Drive và đường dẫn là chính xác.")
    exit()

df['parent_labels'] = df['parent_labels'].apply(ast.literal_eval)
df['child_labels'] = df['child_labels'].apply(ast.literal_eval)

# ===================================================================
# PHẦN 2: MÃ HÓA VĂN BẢN VỚI FASTTEXT
# ===================================================================
print("\n🚀 [Bước 2/7] Tải mô hình FastText và mã hóa văn bản...")
print("   - Đang tải mô hình fasttext-wiki-news-subwords-300... (Lần đầu có thể mất vài phút)")
ft_model = gensim.downloader.load('fasttext-wiki-news-subwords-300')
embedding_dim = ft_model.vector_size
print(f"✅ Tải mô hình FastText thành công (số chiều vector: {embedding_dim}).")

# ### V2.2 CẢI TIẾN: Hàm tiền xử lý dùng spaCy ###
def preprocess_text_spacy(text):
    doc = nlp(str(text).lower())
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return tokens

def abstract_to_vector(abstract, model, dim):
    tokens = preprocess_text_spacy(abstract)
    word_vectors = [model[word] for word in tokens if word in model.key_to_index]
    
    if not word_vectors:
        return np.zeros(dim)
    
    return np.mean(word_vectors, axis=0)

print("   - Đang tạo vector đặc trưng cho các abstract (sử dụng spaCy)...")
tqdm.pandas(desc="Mã hóa Abstract")
df['abstract_vector'] = df['abstract'].progress_apply(lambda x: abstract_to_vector(x, ft_model, embedding_dim))

all_embeddings = np.vstack(df['abstract_vector'].values)
print(f"✅ Mã hóa FastText hoàn tất. Kích thước ma trận đặc trưng: {all_embeddings.shape}")

# ===================================================================
# PHẦN 3: TÌM SIÊU THAM SỐ TỐI ƯU CHO TẦNG 1 VỚI OPTUNA
# ===================================================================
print(f"\n🚀 [Bước 3/7] Tối ưu siêu tham số cho Tầng 1 với Optuna...")

parent_label_counts = Counter([item for sublist in df['parent_labels'] for item in sublist])
target_parents = [label for label, count in parent_label_counts.most_common(N_TARGET_LABELS)]
mlb_parent = MultiLabelBinarizer(classes=target_parents)
y_parent_binarized = mlb_parent.fit_transform(df['parent_labels'])
indices = df.index.values

X_train_emb, X_test_emb, y_train_p, y_test_p, indices_train, indices_test = train_test_split(
    all_embeddings, y_parent_binarized, indices, test_size=0.2, random_state=42
)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1200, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }
    params.update(LGBM_FIXED_PARAMS)
    
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train_emb, y_train_p, test_size=0.25, random_state=42)
    
    model = OneVsRestClassifier(LGBMClassifier(**params), n_jobs=1)
    model.fit(X_train_opt, y_train_opt)
    
    preds = model.predict(X_val_opt)
    score = f1_score(y_val_opt, preds, average='weighted', zero_division=0)
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)
best_params_tier1 = study.best_params
print(f"✅ Tối ưu hóa hoàn tất sau {len(study.trials)} trials.")
print(f"   - F1-Score tốt nhất trên tập validation: {study.best_value:.4f}")
print(f"   - Siêu tham số tối ưu: {best_params_tier1}")

# ===================================================================
# PHẦN 4: HUẤN LUYỆN TẦNG 1 VỚI THAM SỐ TỐT NHẤT
# ===================================================================
print("\n🚀 [Bước 4/7] Huấn luyện Tầng 1 trên toàn bộ tập train với tham số tốt nhất...")
final_params = best_params_tier1.copy()
final_params.update(LGBM_FIXED_PARAMS)
parent_model = OneVsRestClassifier(LGBMClassifier(**final_params), n_jobs=1)
parent_model.fit(X_train_emb, y_train_p)
print("✅ Huấn luyện mô hình Tầng 1 cuối cùng hoàn tất.")

# ===================================================================
# PHẦN 5: HUẤN LUYỆN TẦNG 2
# ===================================================================
print("\n🚀 [Bước 5/7] Huấn luyện Tầng 2...")
tier2_classifiers, tier2_mlbs = {}, {}
df_train = df.loc[indices_train]
emb_train = all_embeddings[indices_train]
for parent_label in tqdm(mlb_parent.classes_, desc="Huấn luyện các mô hình Tầng 2"):
    indices_with_parent_local = [i for i, labels in enumerate(df_train['parent_labels']) if parent_label in labels]
    if len(indices_with_parent_local) < 20: continue
    df_child, X_child_emb = df_train.iloc[indices_with_parent_local], emb_train[indices_with_parent_local]
    y_child_raw = df_child['child_labels'].apply(lambda l: [c for c in l if c.startswith(parent_label)])
    if y_child_raw.apply(len).sum() == 0: continue
    mlb_child = MultiLabelBinarizer()
    y_child_binarized = mlb_child.fit_transform(y_child_raw)
    if y_child_binarized.shape[1] < 2: continue
    child_model = OneVsRestClassifier(LGBMClassifier(**final_params), n_jobs=1)
    child_model.fit(X_child_emb, y_child_binarized)
    tier2_classifiers[parent_label], tier2_mlbs[parent_label] = child_model, mlb_child
print(f"\n✅ Đã huấn luyện {len(tier2_classifiers)} mô hình Tầng 2.")

# ===================================================================
# PHẦN 6: ĐÁNH GIÁ VÀ TẠO BÁO CÁO METRICS CHI TIẾT
# ===================================================================
print("\n🚀 [Bước 6/7] Đánh giá và tạo báo cáo metrics chi tiết...")
df_test = df.loc[indices_test]
emb_test = all_embeddings[indices_test]

true_child_labels_raw = df_test['child_labels'].tolist()
mlb_all_children = MultiLabelBinarizer().fit(df['child_labels'])
y_test_child_true_binarized = mlb_all_children.transform(true_child_labels_raw)
y_pred_parent_binarized = parent_model.predict(emb_test)
final_parents_raw = mlb_parent.inverse_transform(y_pred_parent_binarized)
final_predictions_raw = []
for i in tqdm(range(len(df_test)), desc="Dự đoán Tầng 2 trên tập test"):
    predicted_parents = final_parents_raw[i]
    child_preds = set()
    if predicted_parents:
        emb_vector = emb_test[i:i+1]
        for parent in predicted_parents:
            if parent in tier2_classifiers:
                child_model, child_mlb = tier2_classifiers[parent], tier2_mlbs[parent]
                pred_child_binarized = child_model.predict(emb_vector)
                child_preds.update(child_mlb.inverse_transform(pred_child_binarized)[0])
    final_predictions_raw.append(sorted(list(child_preds)))
y_pred_child_final_binarized = mlb_all_children.transform(final_predictions_raw)

metrics_report = {}
report_parent_dict = classification_report(y_test_p, y_pred_parent_binarized, target_names=mlb_parent.classes_, output_dict=True, zero_division=0)
metrics_report['f1_macro_parent'] = report_parent_dict['macro avg']['f1-score']
metrics_report['f1_weighted_parent'] = report_parent_dict['weighted avg']['f1-score']
metrics_report['f1_samples_parent'] = f1_score(y_test_p, y_pred_parent_binarized, average='samples', zero_division=0)
metrics_report['f1_macro_children_overall'] = f1_score(y_test_child_true_binarized, y_pred_child_final_binarized, average='macro', zero_division=0)
metrics_report['f1_weighted_children_overall'] = f1_score(y_test_child_true_binarized, y_pred_child_final_binarized, average='weighted', zero_division=0)
metrics_report['f1_samples_children_overall'] = f1_score(y_test_child_true_binarized, y_pred_child_final_binarized, average='samples', zero_division=0)
metrics_report['best_hyperparameters_tier1'] = study.best_params

print("\n" + "="*80)
print(" " * 16 + "BÁO CÁO HIỆU SUẤT HỆ THỐNG - VERSION 2.2 (spaCy + FastText)")
print("="*80)
print(f"\n   - SIÊU THAM SỐ TỐI ƯU (từ Optuna):")
for key, value in metrics_report['best_hyperparameters_tier1'].items():
    if isinstance(value, float):
        print(f"     - {key}: {value:.4f}")
    else:
        print(f"     - {key}: {value}")

print("\n--- Tầng 1 (Dự đoán 17 Nhãn Cha chính) ---")
print(f"   - ⭐️ F1-Score (Weighted Avg): {metrics_report['f1_weighted_parent']:.4f}")
print(f"   - F1-Score (Macro Avg):        {metrics_report['f1_macro_parent']:.4f}")

print("\n--- Toàn Hệ Thống (Dự đoán Nhãn Con Cuối Cùng) ---")
print(f"   - ⭐️ F1-Score (Weighted Avg): {metrics_report['f1_weighted_children_overall']:.4f}")
print(f"   - F1-Score (Macro Avg):        {metrics_report['f1_macro_children_overall']:.4f}")
print("\n" + "="*80)

# ===================================================================
# PHẦN 7: LƯU KẾT QUẢ VÀ CÁC THÀNH PHẦN
# ===================================================================
print("\n🚀 [Bước 7/7] Lưu kết quả và các thành phần...")
MODEL_DIR = "/content/drive/MyDrive/data/saved_models_v2.2_spacy_fasttext_optuna/"
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, 'tier1_classifier.pkl'), 'wb') as f: pickle.dump(parent_model, f)
with open(os.path.join(MODEL_DIR, 'tier2_classifiers.pkl'), 'wb') as f: pickle.dump(tier2_classifiers, f)
with open(os.path.join(MODEL_DIR, 'tier1_mlb.pkl'), 'wb') as f: pickle.dump(mlb_parent, f)
with open(os.path.join(MODEL_DIR, 'tier2_mlbs.pkl'), 'wb') as f: pickle.dump(tier2_mlbs, f)
with open(os.path.join(MODEL_DIR, 'metrics_report.json'), 'w') as f: json.dump(metrics_report, f, indent=4)
print(f"✅ Đã lưu thành công các thành phần mô hình vào: {MODEL_DIR}")
```
**Version:** 2.2 - spaCy + FastText + Optuna  
**So với Version 1.0:** Thay thế TF-IDF bằng FastText embeddings và tối ưu hóa siêu tham số bằng Optuna.

## **1. Mục Tiêu**

Version 2.2 được phát triển với hai mục tiêu chính:
1.  **Giải quyết vấn đề hiệu năng:** Thay thế pipeline TF-IDF bằng một giải pháp nhẹ hơn (FastText) để chạy mượt mà trên Colab.
2.  **Cải thiện hiệu suất:** Kỳ vọng rằng việc sử dụng word embeddings có ngữ nghĩa và tối ưu hóa siêu tham số sẽ cho kết quả tốt hơn V1.0.

## **2. Kết Quả Thử Nghiệm (Version 2.2)**

### **Bảng So Sánh Hiệu Suất: V1.0 vs V2.2**

| Metric | V1.0 (TF-IDF 5k) | **V2.2 (FastText 300d)** | Thay Đổi | Phân Tích Nhanh |
| :--- | :--- | :--- | :--- | :--- |
| **Thời gian Dự đoán Tầng 2** | ~33 phút | **~5 phút** | **↓ 85%** | ✅ **Thành công lớn** |
| **F1-Weighted (Tầng 1)** | 0.6483 | **0.6386** | **↓ 1.5%** | ⚠️ Giảm nhẹ |
| **F1-Macro (Tầng 1)** | 0.6474 | **0.6359** | **↓ 1.8%** | ⚠️ Giảm nhẹ |
| **F1-Weighted (Tầng 2)** | 0.4047 | **0.3732** | **↓ 7.8%** | ⚠️ Giảm đáng kể |
| **F1-Macro (Tầng 2)** | 0.2543 | **0.2319** | **↓ 8.8%** | ⚠️ Giảm đáng kể |

### **Hyperparameters Tối Ưu (từ Optuna):**
-   `n_estimators`: 550
-   `learning_rate`: 0.0522
-   `num_leaves`: 146
-   `max_depth`: **5**
-   `reg_alpha`: 0.0081
-   `reg_lambda`: 0.0563

## **3. Phân Tích & Đánh Giá**

Version 2.2 là một thử nghiệm cực kỳ thành công trong việc cung cấp thông tin, dù các chỉ số F1-score đã giảm.

### **3.1. Điểm Tích Cực**
-   **Vấn đề Hiệu năng đã được giải quyết triệt để:** Thời gian dự đoán giảm từ 33 phút xuống chỉ còn 5 phút là một thắng lợi lớn, chứng tỏ FastText là một lựa chọn tuyệt vời về mặt tốc độ và tài nguyên. Pipeline hiện tại đã sẵn sàng cho việc thử nghiệm nhanh chóng hơn.

### **3.2. Phân Tích Sụt Giảm Hiệu Suất: Tại Sao Kết Quả Lại Thấp Hơn?**

Đây là điểm mấu chốt. Dù sử dụng kỹ thuật có vẻ "hiện đại" hơn, hiệu suất lại giảm. Nguyên nhân đến từ hai yếu tố chính:

#### **1. Sự "Pha Loãng" Tín Hiệu của Vector Trung Bình (Quan trọng nhất)**
-   **TF-IDF (V1.0):** Rất giỏi trong việc nhận diện các **từ khóa quan trọng nhưng hiếm**. Ví dụ, một thuật ngữ như "abelian variety" hoặc "hadronization" có thể có điểm TF-IDF rất cao và trở thành một tín hiệu cực mạnh cho mô hình.
-   **FastText (V2.2):** Phương pháp của chúng ta là **lấy trung bình vector của TẤT CẢ các từ** trong abstract. Điều này có một nhược điểm chí mạng: vector của một từ khóa cực kỳ quan trọng như "abelian variety" sẽ bị "pha loãng" bởi hàng trăm vector của các từ phổ biến khác như "study", "result", "paper", "method",... Tín hiệu đặc trưng mạnh mẽ của từ khóa đó bị mất đi trong giá trị trung bình.
-   **Kết luận:** Đối với văn bản khoa học, nơi các thuật ngữ cụ thể mang tính quyết định, phương pháp "túi từ" của TF-IDF đôi khi lại hiệu quả hơn phương pháp lấy trung bình vector một cách ngây thơ.

#### **2. Dấu Hiệu Overfitting trong Tối Ưu Hóa của Optuna**
-   Hãy nhìn vào các tham số Optuna tìm được: `max_depth: 5` và `num_leaves: 146`.
-   Đây là một **mâu thuẫn lớn**. Một cây quyết định có độ sâu tối đa là 5 (`max_depth=5`) chỉ có thể có tối đa **2^5 = 32** lá (`leaves`).
-   Việc Optuna chọn `num_leaves=146` (nhiều hơn 32 rất nhiều) cho thấy LightGBM đang cố gắng tạo ra những cây rất "rộng" và "nông". Nó đang tạo ra rất nhiều quy tắc phân chia rất cụ thể ở các cấp độ thấp mà không xây dựng được các quy tắc tổng quát ở các cấp độ cao hơn.
-   **Nguyên nhân:** Đây là dấu hiệu kinh điển của việc mô hình đang **overfit trên tập validation** trong quá trình tìm kiếm của Optuna. Nó đã tìm ra một bộ tham số "kỳ lạ" hoạt động tốt trên một phần nhỏ dữ liệu đó, nhưng lại không có khả năng tổng quát hóa tốt trên tập test cuối cùng.

## **4. Hướng Cải Thiện cho Version 3.0 (Dựa trên kết quả V2.2)**

Chúng ta đã học được rằng: 1) không thể bỏ qua tầm quan trọng của từ khóa, và 2) cần kiểm soát Optuna tốt hơn. Dưới đây là các bước đi tiếp theo rất rõ ràng.

### **Ưu tiên #1: Kết hợp Sức mạnh của TF-IDF và Word Embeddings (TF-IDF Weighted Embeddings)**
-   **Ý tưởng:** Thay vì lấy trung bình cộng các vector từ, chúng ta sẽ lấy **trung bình có trọng số**. Trọng số của mỗi từ chính là điểm TF-IDF của từ đó.
-   **Quy trình:**
    1.  Chạy `TfidfVectorizer` như V1.0 để có điểm số cho từng từ.
    2.  Với mỗi abstract, khi tạo vector cuối cùng, nhân vector FastText của mỗi từ với điểm TF-IDF của từ đó, sau đó lấy tổng và chia cho tổng các điểm TF-IDF.
-   **Lợi ích:** Cách tiếp cận "lai" này giữ lại được **ngữ nghĩa** của FastText và **tầm quan trọng** của từ khóa từ TF-IDF. Các từ quan trọng sẽ có đóng góp lớn hơn vào vector cuối cùng.

### **Ưu tiên #2: Tinh Chỉnh Lại Không Gian Tìm Kiếm của Optuna**
-   **Vấn đề:** Tham số `num_leaves` và `max_depth` đang mâu thuẫn.
-   **Giải pháp:** Ràng buộc không gian tìm kiếm để nó hợp lý hơn.
    -   Bỏ `max_depth` ra khỏi danh sách tìm kiếm và đặt một giá trị cố định (ví dụ: -1 để không giới hạn).
    -   Hoặc, ràng buộc `num_leaves` trong hàm `objective`: `num_leaves = trial.suggest_int('num_leaves', 10, 2**params['max_depth'] - 1)`. Điều này buộc số lá phải nhỏ hơn mức tối đa cho phép của độ sâu.
-   **Gợi ý:** Bắt đầu bằng cách chỉ tối ưu `n_estimators`, `learning_rate`, `num_leaves`, `reg_alpha`, `reg_lambda`. Đây là những tham số có tác động lớn nhất.

### **Ưu tiên #3 (Con đường dài hạn): Tiến tới Contextual Embeddings**
-   Kết quả này càng củng cố thêm giả thuyết rằng các mô hình có khả năng hiểu **ngữ cảnh** (như SciBERT) sẽ là chìa khóa để đạt được hiệu suất cao nhất, vì chúng không cần phải lấy trung bình vector và có thể hiểu được từ nào là quan trọng trong một câu cụ thể.

## **5. Kết Luận Chung**

Version 2.2 là một bước tiến quan trọng. Mặc dù F1-score giảm, chúng ta đã:
1.  **Thành công** giải quyết vấn đề hiệu năng.
2.  **Học được rằng** phương pháp lấy trung bình vector đơn giản không đủ tốt cho dữ liệu chuyên ngành.
3.  **Phát hiện ra** điểm yếu trong cách cấu hình Optuna.

Đây là những kinh nghiệm quý báu. Thất bại trong việc cải thiện metrics nhưng thành công trong việc thu thập thông tin để các phiên bản sau tốt hơn. Lộ trình cho V3.0 đã rất rõ ràng: kết hợp TF-IDF và FastText, đồng thời tinh chỉnh lại quy trình tối ưu hóa.

# Version 3
## Script
```python
# !pip install optuna
# !pip install gensim
# !python -m spacy download en_core_web_sm
# %pip install cuml-cu12 cudf-cu12

# ===================================================================
#      VERSION 3.2: GPU-ACCELERATED TF-IDF + WEIGHTED FASTTEXT
# ===================================================================
# Script này sử dụng RAPIDS cuML để tăng tốc TF-IDF trên GPU.
# Phiên bản này đã sửa lỗi TypeError khi xử lý vocabulary của cuML.
# ===================================================================

# ===================================================================
# PHẦN 0: CÁC THƯ VIỆN CẦN THIẾT
# ===================================================================
print("🚀 Đang import các thư viện...")
import pandas as pd
import numpy as np
import ast
import pickle
import os
import json
from collections import Counter
# ### V3.2 CẢI TIẾN: Thêm thư viện của RAPIDS ###
import cudf
from cuml.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# Các thư viện còn lại
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report
from tqdm.auto import tqdm
import optuna
import gensim.downloader
import spacy
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
print("✅ Import thư viện hoàn tất.")

# ===================================================================
# PHẦN 0B: CẤU HÌNH
# ===================================================================
N_TARGET_LABELS = 17
TFIDF_MAX_FEATURES = 15000

OPTUNA_N_TRIALS = 30
OPTUNA_TIMEOUT = 5400

LGBM_FIXED_PARAMS = {
    'device': 'gpu',
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'
}
print("⚡️ Đã kích hoạt chế độ huấn luyện GPU và cấu hình cho Version 3.2 (GPU TF-IDF)!")

# ===================================================================
# PHẦN 1: TẢI DỮ LIỆU VÀ MÔ HÌNH NLP
# ===================================================================
print("\n🚀 [Bước 1/8] Tải dữ liệu và mô hình NLP...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("   - Mô hình spaCy 'en_core_web_sm' đã có sẵn.")
except OSError:
    print("   - Lần đầu chạy, đang cài đặt và tải mô hình spaCy...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
nlp.disable_pipes("parser", "ner")
print("   - Tải spaCy hoàn tất.")

FILE_PATH = "/content/drive/MyDrive/AIO25/m04/data/arxiv_perfectly_balanced.csv"
try:
    df = pd.read_csv(FILE_PATH)
    print(f"✅ Tải thành công file: '{FILE_PATH}' ({len(df):,} mẫu)")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file '{FILE_PATH}'.")
    exit()

df['parent_labels'] = df['parent_labels'].apply(ast.literal_eval)
df['child_labels'] = df['child_labels'].apply(ast.literal_eval)

# ===================================================================
# PHẦN 2: TIỀN XỬ LÝ VĂN BẢN VÀ HUẤN LUYỆN TF-IDF TRÊN GPU
# ===================================================================
print("\n🚀 [Bước 2/8] Tiền xử lý văn bản và huấn luyện TF-IDF trên GPU...")

def preprocess_text_spacy(text):
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(tokens)

tqdm.pandas(desc="Tiền xử lý Abstract")
df['processed_abstract'] = df['abstract'].progress_apply(preprocess_text_spacy)

print("   - Bắt đầu huấn luyện TF-IDF trên GPU (sẽ nhanh hơn rất nhiều)...")
cudf_series = cudf.Series(df['processed_abstract'])
tfidf_vectorizer_gpu = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
tfidf_vectorizer_gpu.fit(cudf_series)

idf_values = tfidf_vectorizer_gpu.idf_
# ### SỬA LỖI LẦN 2: Dùng phương pháp đảo ngược Series, an toàn và hiệu quả ###
vocab_gpu = tfidf_vectorizer_gpu.vocabulary_
vocab_cpu = vocab_gpu.to_pandas()

# Tạo một Series mới với index là chỉ số cột và value là từ, sau đó sắp xếp
index_to_word_series = pd.Series(vocab_cpu.index, index=vocab_cpu.values).sort_index()
# Lấy danh sách từ đã được sắp xếp chính xác
feature_names = index_to_word_series.to_list()

idf_weights = dict(zip(feature_names, idf_values))
print("✅ Huấn luyện TF-IDF trên GPU và tạo trọng số IDF thành công.")

# ===================================================================
# PHẦN 3: TẢI FASTTEXT VÀ TẠO VECTOR KẾT HỢP
# ===================================================================
print("\n🚀 [Bước 3/8] Tải FastText và tạo vector đặc trưng kết hợp...")
print("   - Đang tải mô hình fasttext-wiki-news-subwords-300...")
ft_model = gensim.downloader.load('fasttext-wiki-news-subwords-300')
embedding_dim = ft_model.vector_size
print(f"✅ Tải mô hình FastText thành công (số chiều: {embedding_dim}).")

def weighted_average_vector(processed_text, ft_model, idf_dict, dim):
    tokens = processed_text.split()
    weighted_vectors = []
    total_weight = 0.0
    for token in tokens:
        if token in ft_model.key_to_index and token in idf_dict:
            vector = ft_model[token]
            weight = idf_dict[token]
            weighted_vectors.append(vector * weight)
            total_weight += weight
    if not weighted_vectors: return np.zeros(dim)
    final_vector = np.sum(weighted_vectors, axis=0) / total_weight
    return final_vector

print("   - Đang tạo vector đặc trưng kết hợp cho các abstract...")
tqdm.pandas(desc="Tạo Vector Kết Hợp")
df['abstract_vector'] = df['processed_abstract'].progress_apply(
    lambda x: weighted_average_vector(x, ft_model, idf_weights, embedding_dim)
)

all_embeddings = np.vstack(df['abstract_vector'].values)
print(f"✅ Tạo vector đặc trưng kết hợp hoàn tất. Kích thước: {all_embeddings.shape}")

# ===================================================================
# PHẦN 4: TỐI ƯU HÓA SIÊU THAM SỐ VỚI OPTUNA
# ===================================================================
print(f"\n🚀 [Bước 4/8] Tối ưu siêu tham số cho Tầng 1 với Optuna...")

parent_label_counts = Counter([item for sublist in df['parent_labels'] for item in sublist])
target_parents = [label for label, count in parent_label_counts.most_common(N_TARGET_LABELS)]
mlb_parent = MultiLabelBinarizer(classes=target_parents)
y_parent_binarized = mlb_parent.fit_transform(df['parent_labels'])
indices = df.index.values

X_train_emb, X_test_emb, y_train_p, y_test_p, indices_train, indices_test = train_test_split(
    all_embeddings, y_parent_binarized, indices, test_size=0.2, random_state=42
)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
    }
    params.update(LGBM_FIXED_PARAMS)
    
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train_emb, y_train_p, test_size=0.25, random_state=42)
    
    model = OneVsRestClassifier(LGBMClassifier(**params), n_jobs=1)
    model.fit(X_train_opt, y_train_opt)
    preds = model.predict(X_val_opt)
    score = f1_score(y_val_opt, preds, average='weighted', zero_division=0)
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)
best_params_tier1 = study.best_params
print(f"✅ Tối ưu hóa hoàn tất sau {len(study.trials)} trials.")
print(f"   - F1-Score tốt nhất trên tập validation: {study.best_value:.4f}")
print(f"   - Siêu tham số tối ưu: {best_params_tier1}")

# ===================================================================
# PHẦN 5: HUẤN LUYỆN TẦNG 1
# ===================================================================
print("\n🚀 [Bước 5/8] Huấn luyện Tầng 1 với tham số tốt nhất...")
final_params = best_params_tier1.copy()
final_params.update(LGBM_FIXED_PARAMS)
parent_model = OneVsRestClassifier(LGBMClassifier(**final_params), n_jobs=1)
parent_model.fit(X_train_emb, y_train_p)
print("✅ Huấn luyện Tầng 1 hoàn tất.")

# ===================================================================
# PHẦN 6: HUẤN LUYỆN TẦNG 2
# ===================================================================
print("\n🚀 [Bước 6/8] Huấn luyện Tầng 2...")
tier2_classifiers, tier2_mlbs = {}, {}
df_train = df.loc[indices_train]
for parent_label in tqdm(mlb_parent.classes_, desc="Huấn luyện các mô hình Tầng 2"):
    indices_with_parent_local = [i for i, labels in enumerate(df_train['parent_labels']) if parent_label in labels]
    if len(indices_with_parent_local) < 20: continue
    df_child, X_child_emb = df_train.iloc[indices_with_parent_local], X_train_emb[indices_with_parent_local]
    y_child_raw = df_child['child_labels'].apply(lambda l: [c for c in l if c.startswith(parent_label)])
    if y_child_raw.apply(len).sum() == 0: continue
    mlb_child = MultiLabelBinarizer()
    y_child_binarized = mlb_child.fit_transform(y_child_raw)
    if y_child_binarized.shape[1] < 2: continue
    child_model = OneVsRestClassifier(LGBMClassifier(**final_params), n_jobs=1)
    child_model.fit(X_child_emb, y_child_binarized)
    tier2_classifiers[parent_label], tier2_mlbs[parent_label] = child_model, mlb_child
print(f"\n✅ Đã huấn luyện {len(tier2_classifiers)} mô hình Tầng 2.")

# ===================================================================
# PHẦN 7: ĐÁNH GIÁ
# ===================================================================
print("\n🚀 [Bước 7/8] Đánh giá và tạo báo cáo metrics chi tiết...")
df_test = df.loc[indices_test]
emb_test = X_test_emb

true_child_labels_raw = df_test['child_labels'].tolist()
mlb_all_children = MultiLabelBinarizer().fit(df['child_labels'])
y_test_child_true_binarized = mlb_all_children.transform(true_child_labels_raw)
y_pred_parent_binarized = parent_model.predict(emb_test)
final_parents_raw = mlb_parent.inverse_transform(y_pred_parent_binarized)
final_predictions_raw = []
for i in tqdm(range(len(df_test)), desc="Dự đoán Tầng 2 trên tập test"):
    predicted_parents = final_parents_raw[i]
    child_preds = set()
    if predicted_parents:
        emb_vector = emb_test[i:i+1]
        for parent in predicted_parents:
            if parent in tier2_classifiers:
                child_model, child_mlb = tier2_classifiers[parent], tier2_mlbs[parent]
                pred_child_binarized = child_model.predict(emb_vector)
                child_preds.update(child_mlb.inverse_transform(pred_child_binarized)[0])
    final_predictions_raw.append(sorted(list(child_preds)))
y_pred_child_final_binarized = mlb_all_children.transform(final_predictions_raw)
metrics_report = {}
report_parent_dict = classification_report(y_test_p, y_pred_parent_binarized, target_names=mlb_parent.classes_, output_dict=True, zero_division=0)
metrics_report['f1_macro_parent'] = report_parent_dict['macro avg']['f1-score']
metrics_report['f1_weighted_parent'] = report_parent_dict['weighted avg']['f1-score']
metrics_report['f1_samples_parent'] = f1_score(y_test_p, y_pred_parent_binarized, average='samples', zero_division=0)
metrics_report['f1_macro_children_overall'] = f1_score(y_test_child_true_binarized, y_pred_child_final_binarized, average='macro', zero_division=0)
metrics_report['f1_weighted_children_overall'] = f1_score(y_test_child_true_binarized, y_pred_child_final_binarized, average='weighted', zero_division=0)
metrics_report['f1_samples_children_overall'] = f1_score(y_test_child_true_binarized, y_pred_child_final_binarized, average='samples', zero_division=0)
metrics_report['best_hyperparameters_tier1'] = study.best_params

print("\n" + "="*80)
print(" " * 12 + "BÁO CÁO HIỆU SUẤT HỆ THỐNG - VERSION 3.2 (GPU TF-IDF)")
print("="*80)
print(f"\n   - SIÊU THAM SỐ TỐI ƯU (từ Optuna):")
for key, value in metrics_report['best_hyperparameters_tier1'].items():
    if isinstance(value, float): print(f"     - {key}: {value:.4f}")
    else: print(f"     - {key}: {value}")
print("\n--- Tầng 1 (Dự đoán 17 Nhãn Cha chính) ---")
print(f"   - ⭐️ F1-Score (Weighted Avg): {metrics_report['f1_weighted_parent']:.4f}")
print(f"   - F1-Score (Macro Avg):        {metrics_report['f1_macro_parent']:.4f}")
print("\n--- Toàn Hệ Thống (Dự đoán Nhãn Con Cuối Cùng) ---")
print(f"   - ⭐️ F1-Score (Weighted Avg): {metrics_report['f1_weighted_children_overall']:.4f}")
print(f"   - F1-Score (Macro Avg):        {metrics_report['f1_macro_children_overall']:.4f}")
print("\n" + "="*80)

# ===================================================================
# PHẦN 8: LƯU KẾT QUẢ
# ===================================================================
print("\n🚀 [Bước 8/8] Lưu kết quả và các thành phần...")
MODEL_DIR = "/content/drive/MyDrive/AIO25/m04/data/saved_models_v3.2_gpu_tfidf_weighted_fasttext/"
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, 'tier1_classifier.pkl'), 'wb') as f: pickle.dump(parent_model, f)
with open(os.path.join(MODEL_DIR, 'tier2_classifiers.pkl'), 'wb') as f: pickle.dump(tier2_classifiers, f)
with open(os.path.join(MODEL_DIR, 'tier1_mlb.pkl'), 'wb') as f: pickle.dump(mlb_parent, f)
with open(os.path.join(MODEL_DIR, 'tier2_mlbs.pkl'), 'wb') as f: pickle.dump(tier2_mlbs, f)
with open(os.path.join(MODEL_DIR, 'cuml_tfidf_vectorizer_v3.pkl'), 'wb') as f: pickle.dump(tfidf_vectorizer_gpu, f)
with open(os.path.join(MODEL_DIR, 'metrics_report.json'), 'w') as f: json.dump(metrics_report, f, indent=4)
print(f"✅ Đã lưu thành công các thành phần mô hình vào: {MODEL_DIR}")
```
**Version:** 3.2 - GPU TF-IDF Weighted FastText + Smart Optuna  
**So với các phiên bản trước:** Một thử nghiệm kết hợp các phương pháp trích xuất đặc trưng nhằm tận dụng ưu điểm của cả hai.

## **1. Mục Tiêu**

Version 3.2 được xây dựng dựa trên những bài học từ V1.0 và V2.2, với một mục tiêu đầy tham vọng:
1.  **Kết hợp "Tốt nhất của cả hai thế giới":** Tạo ra một vector đặc trưng duy nhất vừa có khả năng **hiểu ngữ nghĩa** (từ FastText) vừa **nhấn mạnh tầm quan trọng của các từ khóa hiếm** (từ TF-IDF).
2.  **Duy trì hiệu năng cao:** Tiếp tục sử dụng RAPIDS cuML để tăng tốc quá trình tính toán TF-IDF trên GPU.
3.  **Tối ưu hóa thông minh:** Áp dụng quy trình tinh chỉnh siêu tham số bằng Optuna trên bộ đặc trưng "lai" mới này.

## **2. Kiến Trúc & Phương Pháp Thực Hiện (Chi tiết)**

Đây là pipeline hoàn chỉnh của Version 3.2, một kiến trúc phức tạp hơn đáng kể so với các phiên bản trước.

1.  **Tiền xử lý văn bản (với spaCy):**
    -   Mỗi abstract được đưa qua một pipeline tiền xử lý: chuyển thành chữ thường, loại bỏ ký tự đặc biệt, và quan trọng nhất là **lemmatization** (đưa từ về dạng gốc, ví dụ: `studies`, `studying` -> `study`).
    -   Kết quả là một phiên bản "sạch" của abstract, sẵn sàng cho các bước tiếp theo.

2.  **Huấn luyện TF-IDF trên GPU (Chỉ để lấy trọng số):**
    -   Toàn bộ 30,000 abstract đã được xử lý được đưa vào `TfidfVectorizer` của `cuML`.
    -   Mô hình này được `fit` trên dữ liệu để học và tính toán **trọng số IDF (Inverse Document Frequency)** cho 15,000 từ phổ biến nhất. IDF là một thước đo cho biết một từ hiếm hay phổ biến trong toàn bộ kho văn bản.
    -   **Lưu ý quan trọng:** Chúng ta **không** sử dụng ma trận TF-IDF mà nó tạo ra. Mục đích duy nhất của bước này là để có được một dictionary `idf_weights` chứa điểm số hiếm của từng từ.

3.  **Tải mô hình FastText:**
    -   Mô hình `fasttext-wiki-news-subwords-300` được tải về. Mô hình này chứa các vector 300 chiều đại diện cho ngữ nghĩa của hàng triệu từ.

4.  **Tạo Vector Đặc Trưng "Lai" (Cốt lõi của V3.2):**
    -   Đây là bước đột phá và cũng là trung tâm của thử nghiệm. Với mỗi abstract, chúng tôi thực hiện:
        a. Tách abstract thành các token (từ).
        b. Với mỗi token, lấy ra **vector FastText** (300 chiều) và **trọng số IDF** của nó.
        c. Nhân vector FastText với trọng số IDF. Thao tác này khuếch đại độ lớn (magnitude) của vector đối với các từ hiếm và giảm độ lớn đối với các từ phổ biến.
        d. Tính **trung bình có trọng số** của tất cả các vector đã được khuếch đại này để tạo ra một vector 300 chiều duy nhất đại diện cho toàn bộ abstract.
    -   **Kỳ vọng:** Vector cuối cùng sẽ vừa mang thông tin ngữ nghĩa, vừa được "lái" theo hướng của các từ khóa quan trọng nhất.

5.  **Tối ưu hóa và Huấn luyện:**
    -   Vector 300 chiều mới này được sử dụng làm đầu vào cho quy trình Optuna và huấn luyện mô hình phân cấp hai tầng LightGBM, tương tự như các phiên bản trước.

## **3. Kết Quả Thử Nghiệm (Version 3.2)**

### **Bảng So Sánh Hiệu Suất: V1.0 vs V2.2 vs V3.2**

| Metric | V1.0 (TF-IDF 5k) | V2.2 (FastText Avg) | **V3.2 (TF-IDF Weighted)** | Phân Tích |
| :--- | :--- | :--- | :--- | :--- |
| **F1-Weighted (Tầng 1)** | **0.6483** | 0.6386 | **0.0951** | **↓ 85%** (Sụp đổ) |
| **F1-Macro (Tầng 1)** | **0.6474** | 0.6359 | **0.0835** | **↓ 87%** (Sụp đổ) |
| **F1-Weighted (Tầng 2)** | **0.4047** | 0.3732 | **0.0264** | **↓ 93%** (Thất bại hoàn toàn) |
| **F1-Macro (Tầng 2)** | **0.2543** | 0.2319 | **0.0080** | **↓ 97%** (Thất bại hoàn toàn) |

## **4. Phân Tích Chuyên Sâu: Tại Sao Kết Quả Lại Tệ Hại Như Vậy?**

Kết quả không chỉ không cải thiện mà còn sụp đổ hoàn toàn. Đây không phải là một sự sụt giảm thông thường mà là dấu hiệu của một **sai lầm cơ bản trong phương pháp luận** khi kết hợp các đặc trưng.

**Nguyên nhân chính: Sự Thống Trị của các Từ Siêu Hiếm và Sự "Nhiễu Loạn" Ngữ Nghĩa**

1.  **Khuếch Đại Tín Hiệu Quá Mức:** Trọng số IDF có thang đo logarit. Một từ xuất hiện trong 10 tài liệu sẽ có điểm IDF cao hơn rất nhiều so với một từ xuất hiện trong 1,000 tài liệu. Khi chúng ta nhân vector FastText (có độ lớn tương đối đồng đều) với điểm IDF này, vector của các từ **siêu hiếm** (ví dụ: một thuật ngữ rất hẹp, một lỗi chính tả,...) sẽ bị khuếch đại lên gấp 10, 20 lần so với các từ khác.

2.  **"Pha Loãng" và "Bóp Méo" Ngữ Nghĩa:**
    -   Hãy tưởng tượng một abstract về "Computer Science" có các từ: `learning` (phổ biến, IDF thấp), `network` (phổ biến, IDF thấp), và một thuật ngữ toán học rất hiếm `Grothendieck-Riemann-Roch` (siêu hiếm, IDF cực cao).
    -   Trong phương pháp **Vector Trung Bình (V2.2)**, `Grothendieck...` chỉ đóng góp một phần nhỏ.
    -   Trong phương pháp **TF-IDF Weighted (V3.2)**, vector của `Grothendieck...` sẽ được nhân với một số rất lớn. Vector 300 chiều cuối cùng sẽ gần như chỉ là vector của `Grothendieck...` và bị bóp méo hoàn toàn. Nó đã **mất hết thông tin ngữ nghĩa** của `learning` và `network`.
    -   Mô hình không còn học về "Khoa học Máy tính" nữa, mà nó đang cố gắng phân loại dựa trên những thuật ngữ dị biệt, nhiễu và không mang tính đại diện cho chủ đề chính.

3.  **So sánh với TF-IDF Thuần Túy (V1.0):**
    -   Trong V1.0, `learning`, `network`, và `Grothendieck` là 3 cột (feature) riêng biệt trong ma trận 15,000 chiều. Mô hình LightGBM đủ thông minh để học rằng `learning` và `network` là những tín hiệu mạnh cho lớp `cs`, trong khi `Grothendieck` có thể là một tín hiệu nhiễu hoặc chỉ quan trọng trong một số trường hợp rất hẹp.
    -   Trong V3.2, chúng ta đã **ép** cả ba tín hiệu này vào một vector 300 chiều duy nhất một cách sai lầm, khiến tín hiệu nhiễu lấn át hoàn toàn tín hiệu chính **trước khi** mô hình có cơ hội học.

## **5. Bài Học Rút Ra và Hướng Đi Tiếp Theo**

Thất bại của V3.2 là bài học quý giá nhất từ trước đến nay.
-   **Bài học:** Việc kết hợp các đặc trưng một cách "ngây thơ" có thể phá hủy thông tin thay vì làm giàu nó. Phải luôn hiểu rõ bản chất và thang đo của từng loại đặc trưng trước khi kết hợp.
-   **Xác nhận:** TF-IDF vẫn là một baseline cực kỳ mạnh mẽ cho các tác vụ phân loại văn bản dựa trên từ khóa.

**Hướng đi cho Version 4.0: Giữ Lại Thông Tin Thay Vì Phá Hủy Nó**

Chúng ta sẽ không cố gắng "ép" các loại đặc trưng vào cùng một không gian nữa. Thay vào đó, chúng ta sẽ cho mô hình thấy tất cả chúng.

-   **Phương pháp:** **Nối Đặc Trưng (Feature Concatenation)**
    1.  Tạo ma trận TF-IDF 15,000 chiều từ V1.0 (sử dụng GPU để tăng tốc).
    2.  Tạo ma trận FastText 300 chiều từ V2.2 (dùng vector trung bình đơn giản).
    3.  **Nối (concatenate)** hai ma trận này lại với nhau theo chiều ngang để tạo ra một ma trận đặc trưng cuối cùng có `15,000 + 300 = 15,300` chiều cho mỗi abstract.
-   **Lợi ích:**
    -   **Bảo toàn thông tin:** Mô hình sẽ nhận được cả hai dạng thông tin một cách riêng biệt: 15,000 cột cho tín hiệu từ khóa và 300 cột cho tín hiệu ngữ nghĩa.
    -   **Tận dụng sức mạnh của LightGBM:** LightGBM và các mô hình cây quyết định khác cực kỳ giỏi trong việc xử lý các không gian đặc trưng có số chiều lớn và tự động chọn ra những đặc trưng quan trọng nhất để phân loại.

## **6. Kết Luận Chung**

Version 3.2 là một thất bại về mặt metrics nhưng là một thành công lớn về mặt khoa học. Nó đã chỉ ra một cách rõ ràng rằng phương pháp lai "TF-IDF Weighted Embedding" là không phù hợp cho bài toán này. Kết quả này giúp chúng ta loại bỏ một hướng đi sai lầm và củng cố cho một hướng đi mới, hứa hẹn hơn cho V4.0: nối đặc trưng.

