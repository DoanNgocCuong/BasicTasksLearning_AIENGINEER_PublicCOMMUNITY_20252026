### Tóm Tắt Chi Tiết Quy Trình và Kết Quả Dự Án

Tài liệu này ghi lại quá trình cải tiến bài toán phân loại chủ đề bài báo ArXiv, chuyển từ một phương pháp tiếp cận đơn giản sang một hệ thống phân cấp đa nhãn tinh vi hơn. Mục tiêu là xây dựng một mô hình không chỉ dự đoán đúng lĩnh vực mà còn có khả năng nhận diện tính liên ngành của khoa học.
#### 0. Code đang dùng
```python
# ===================================================================
# PHẦN 0: CÁC THƯ VIỆN CẦN THIẾT
# ===================================================================
print("🚀 Đang import các thư viện...")
# ... (Phần import giữ nguyên như trước) ...
import pandas as pd
import numpy as np
import ast
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("⚠️ Thư viện sentence-transformers chưa được cài đặt. Đang tiến hành cài đặt...")
    !pip install -U sentence-transformers
    from sentence_transformers import SentenceTransformer
    print("✅ Cài đặt sentence-transformers thành công!")
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, jaccard_score, classification_report
import warnings
warnings.filterwarnings('ignore')
print("✅ Import thư viện hoàn tất.")

# ===================================================================
# PHẦN 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ===================================================================
print("\n🚀 [Bước 1/5] Tải và chuẩn bị dữ liệu...")

# --- Cấu hình ---
# Đảm bảo đường dẫn này CHÍNH XÁC
FILE_PATH = "/content/drive/MyDrive/data/arxiv_perfectly_balanced.csv"
SAMPLE_SIZE = None # Đặt là một số (ví dụ: 10000) để chạy thử, hoặc None để chạy toàn bộ

df = None # Khởi tạo df là None
try:
    df = pd.read_csv(FILE_PATH)
    print(f"✅ Tải thành công file: '{FILE_PATH}' ({len(df):,} mẫu)")

    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        print(f"   - Lấy mẫu thử nghiệm với {SAMPLE_SIZE:,} dòng.")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

except FileNotFoundError:
    print(f"❌ LỖI NGHIÊM TRỌNG: Không tìm thấy file tại '{FILE_PATH}'.")
    print("   - Vui lòng kiểm tra lại đường dẫn và tên file.")
    # Chạy lệnh ls để giúp debug
    !ls "/content/drive/MyDrive/AIO25/m04/data/"

# --- Chỉ chạy phần còn lại nếu df được tải thành công ---
if df is not None:
    # Chuyển đổi các cột nhãn từ chuỗi về list
    df['parent_labels'] = df['parent_labels'].apply(ast.literal_eval)
    df['child_labels'] = df['child_labels'].apply(ast.literal_eval)

    # --- Chuẩn bị dữ liệu cho Tầng 1: Dự đoán Nhãn Cha ---
    X = df['abstract'].astype(str)
    y = df['parent_labels']

    # Mã hóa nhãn đa nhãn thành ma trận nhị phân
    mlb = MultiLabelBinarizer()
    y_binarized = mlb.fit_transform(y)
    print(f"✅ Đã mã hóa {len(mlb.classes_)} nhãn cha thành ma trận nhị phân.")
    print(f"   - Các lớp: {mlb.classes_}")

    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binarized, test_size=0.2, random_state=42
    )
    print(f"✅ Đã chia dữ liệu: {len(X_train):,} train, {len(X_test):,} test.")

    # GIẢI PHÓNG BỘ NHỚ
    del df
    import gc
    gc.collect()
    print("   - Đã giải phóng bộ nhớ của DataFrame gốc.")

# ===================================================================
# PHẦN 2: MÃ HÓA VĂN BẢN (FEATURE ENGINEERING) - ĐÃ SỬA LỖI
# ===================================================================
print("\n🚀 [Bước 2/5] Mã hóa văn bản (BoW, TF-IDF, Embeddings)...")

# --- 2.1 Bag-of-Words (BoW) ---
print("\n--- 2.1 Mã hóa bằng Bag-of-Words ---")
bow_vectorizer = CountVectorizer(max_features=10000, stop_words='english')
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)
print(f"   - Kích thước X_train_bow: {X_train_bow.shape}")

# --- 2.2 TF-IDF ---
print("\n--- 2.2 Mã hóa bằng TF-IDF ---")
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"   - Kích thước X_train_tfidf: {X_train_tfidf.shape}")


# --- 2.3 Sentence Embeddings (SỬ DỤNG CLASS MỚI ĐÃ TỐI ƯU) ---
print("\n--- 2.3 Mã hóa bằng Sentence Embeddings ---")

class EmbeddingVectorizer:
    """Mã hóa văn bản thành vector embeddings sử dụng SentenceTransformers."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.is_e5_model = 'e5' in model_name.lower()

    # Sửa đổi: Loại bỏ tham số precision khỏi định nghĩa hàm
    def transform(self, texts: pd.Series, batch_size: int = 64) -> np.ndarray:
        texts_list = texts.tolist()
        if self.is_e5_model:
            print(f"   - Mô hình E5 được phát hiện. Đang thêm tiền tố 'passage: '...")
            texts_to_encode = [f"passage: {text}" for text in texts_list]
        else:
            texts_to_encode = texts_list

        print(f"   - Bắt đầu mã hóa {len(texts_to_encode):,} văn bản với mô hình '{self.model.tokenizer.name_or_path}'...")
        embeddings = self.model.encode(
            texts_to_encode,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=batch_size
            # Không truyền tham số precision nữa
        )
        return embeddings

# **LỰA CHỌN MÔ HÌNH EMBEDDING**
model_name = 'all-MiniLM-L6-v2' # Nhanh, hiệu quả, 384 chiều

embedding_vectorizer = EmbeddingVectorizer(model_name=model_name)

# Sửa đổi: Loại bỏ tham số precision khi gọi hàm
X_train_embeddings = embedding_vectorizer.transform(X_train, batch_size=128)
X_test_embeddings = embedding_vectorizer.transform(X_test, batch_size=128)

print("✅ Mã hóa embeddings hoàn tất.")
print(f"   - Kích thước X_train_embeddings: {X_train_embeddings.shape}")

# ===================================================================
# PHẦN 3: ĐỊNH NGHĨA CÁC MÔ HÌNH (ĐÃ TỐI ƯU HÓA)
# ===================================================================
print("\n🚀 [Bước 3/5] Định nghĩa các mô hình hiệu năng cao...")

# Các mô hình này nhanh và mạnh mẽ
models_to_train = {
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'XGBoost': OneVsRestClassifier(
        XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
        n_jobs=-1
    ),
    'LightGBM': OneVsRestClassifier(
        LGBMClassifier(random_state=42, n_jobs=-1),
        n_jobs=-1
    ),
}

print(f"✅ Sẵn sàng huấn luyện {len(models_to_train)} mô hình hiệu năng cao.")

# ===================================================================
# PHẦN 4: HUẤN LUYỆN VÀ ĐÁNH GIÁ (ĐÃ SỬA LỖI VÀ THÊM SO SÁNH)
# ===================================================================
print("\n🚀 [Bước 4/5] Bắt đầu quá trình huấn luyện và đánh giá...")

from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

datasets_for_training = {
    'BoW': (X_train_bow.astype(np.float32), X_test_bow.astype(np.float32)), # ÉP KIỂU Ở ĐÂY
    'TF-IDF': (X_train_tfidf, X_test_tfidf),
    'Embeddings': (X_train_embeddings, X_test_embeddings)
}

results = []

# --- Chuẩn bị dữ liệu để tính Accuracy so sánh ---
# Lấy nhãn đầu tiên từ y_test đa nhãn
y_test_single_label = np.array([np.where(row == 1)[0][0] if np.sum(row) > 0 else -1 for row in y_test])


total_runs = len(models_to_train) * len(datasets_for_training)
with tqdm(total=total_runs, desc="Tổng tiến độ huấn luyện") as pbar:
    for model_name, model in models_to_train.items():
        for data_name, (X_train_data, X_test_data) in datasets_for_training.items():
            pbar.set_description(f"Huấn luyện {model_name} với {data_name}")
            
            model.fit(X_train_data, y_train)
            y_pred = model.predict(X_test_data)
            
            # --- TÍNH TOÁN CÁC METRICS ---
            subset_accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='samples', zero_division=0)
            jaccard = jaccard_score(y_test, y_pred, average='samples', zero_division=0)

            # --- TÍNH ACCURACY ĐỂ SO SÁNH ---
            # Chuyển y_pred đa nhãn thành đơn nhãn (lấy nhãn đầu tiên)
            y_pred_single_label = np.array([np.where(row == 1)[0][0] if np.sum(row) > 0 else -1 for row in y_pred])
            # Tính accuracy trên phiên bản đơn nhãn
            comparative_accuracy = accuracy_score(y_test_single_label, y_pred_single_label)
            
            results.append({
                'Model': model_name,
                'Encoding': data_name,
                'Comparative Accuracy': comparative_accuracy, # THÊM CỘT NÀY
                'Subset Accuracy': subset_accuracy,
                'F1 Score (Samples)': f1,
                'Jaccard Score (Samples)': jaccard
            })
            
            print(f"\n--- Kết quả cho: {model_name} với {data_name} ---")
            print(f"   -> Accuracy (So sánh): {comparative_accuracy:.4f}") # THÊM DÒNG NÀY
            print(f"   -> Subset Accuracy: {subset_accuracy:.4f}")
            print(f"   -> F1 Score: {f1:.4f}")
            print(f"   -> Jaccard Score: {jaccard:.4f}")
            
            pbar.update(1)


# ===================================================================
# PHẦN 5: TỔNG KẾT KẾT QUẢ
# ===================================================================
print("\n🚀 [Bước 5/5] Tổng kết kết quả...")
results_df = pd.DataFrame(results)
# Sắp xếp theo Comparative Accuracy để dễ so sánh nhất
results_df = results_df.sort_values(by='Comparative Accuracy', ascending=False).reset_index(drop=True)
print("\n" + "="*120)
print(" " * 40 + "BẢNG XẾP HẠNG KẾT QUẢ PHÂN LOẠI NHÃN CHA")
print("="*120)
print(results_df.to_string())
print("="*120)

# In ra classification report chi tiết cho mô hình tốt nhất
if not results_df.empty:
    best_model_info = results_df.iloc[0]
    best_model_name = best_model_info['Model']
    best_encoding_name = best_model_info['Encoding']

    print(f"\n🔍 Phân tích chi tiết cho mô hình tốt nhất: {best_model_name} với {best_encoding_name}")
    best_model = models_to_train[best_model_name]
    X_train_best, X_test_best = datasets_for_training[best_encoding_name]

    print("   - Đang huấn luyện lại mô hình tốt nhất để tạo report chi tiết...")
    best_model.fit(X_train_best, y_train)
    y_pred_best = best_model.predict(X_test_best)

    report = classification_report(y_test, y_pred_best, target_names=mlb.classes_, zero_division=0)
    print(report)
else:
    print("⚠️ Không có kết quả nào để phân tích.")
```
#### 1. So Sánh Cách Tiếp Cận: Cũ vs. Mới

Để hiểu rõ những cải tiến, chúng ta cần so sánh hai phương pháp:

| Tiêu chí | Phương Pháp Cũ (Project Gốc) | **Phương Pháp Mới (Cải Tiến)** |
| :--- | :--- | :--- |
| **Phạm vi dữ liệu** | Lấy 1,000 mẫu đơn giản, **chỉ thuộc 5 lĩnh vực** được chọn trước. | Làm việc trên toàn bộ **2.3 triệu bài báo** để phân tích và tạo ra một bộ dữ liệu con **30,000 mẫu** đại diện cho **17 lĩnh vực chính**. |
| **Xử lý nhãn** | Lấy nhãn đầu tiên, bỏ qua các nhãn phụ. Coi mỗi bài báo là **đơn nhãn**. | Phân tích cấu trúc `.` và `-` để tự động xác định **nhãn Cha (lĩnh vực lớn)** và **nhãn Con (chủ đề chi tiết)**. Chấp nhận và xử lý bài toán **đa nhãn**. |
| **Cân bằng dữ liệu** | Lấy 200 mẫu cho mỗi trong 5 lớp (cân bằng đơn giản). | Áp dụng một chiến lược lấy mẫu phức tạp để **cân bằng đồng thời cả về số lượng giữa 17 lớp và cả về tỷ lệ 50/50 giữa các bài báo đơn nhãn và đa nhãn**. |
| **Kiến trúc mô hình** | Một mô hình duy nhất, phân loại 1 trong 5 lớp. | Xây dựng nền tảng cho **kiến trúc 2 tầng**: Tầng 1 dự đoán các nhãn Cha, Tầng 2 (tương lai) sẽ dự đoán các nhãn Con tương ứng. |
| **Độ khó bài toán** | **Thấp:** Phân loại đơn nhãn, 5 lớp. | **Rất cao:** Phân loại đa nhãn, 17 lớp, yêu cầu dự đoán đúng một tập hợp các nhãn. |

Về cơ bản, chúng ta đã chuyển từ một bài toán "đồ chơi" sang một bài toán gần với thực tế hơn rất nhiều.

---

#### 2. Quy Trình Làm Việc Chi Tiết

##### Giai đoạn 1: Tạo Cấu Trúc Nhãn Cha-Con

Chúng tôi đã xây dựng một quy trình tự động để phân cấp hơn 2.3 triệu bài báo:
1.  **Tạo Ứng Cử Viên:** Quét qua 3.8 triệu lượt gán nhãn, trích xuất phần đầu của mỗi nhãn (ví dụ `math.CO` -> `math`) làm "ứng cử viên" nhãn cha.
2.  **Lựa Chọn Dựa Trên Dữ Liệu:** Đặt ra ngưỡng khách quan: chỉ những ứng cử viên chiếm hơn 0.1% "thị phần" trong tổng số lượt gán nhãn mới được công nhận là Nhãn Cha. Quá trình này đã xác định được **17 lĩnh vực lớn**.
3.  **Tạo Cột Mới:** Bổ sung hai cột `parent_labels` và `child_labels` vào dataset gốc.

##### Giai đoạn 2: Tạo Dataset Con Cân Bằng Tối Ưu

Từ 2.3 triệu dòng, chúng tôi đã tạo ra một bộ dữ liệu 30,000 mẫu (`arxiv_perfectly_balanced.csv`) với các đặc điểm sau:
*   **Cân bằng Cấu trúc:** Có chính xác **14,994 (50.0%)** bài báo đơn nhãn và **15,000 (50.0%)** bài báo đa nhãn.
*   **Cân bằng Lớp:** Sự chênh lệch số lượng mẫu giữa 17 lớp cha đã được giảm thiểu đáng kể, giúp mô hình không bị thiên vị.

##### Giai đoạn 3: Huấn Luyện và Đánh Giá Mô Hình Tầng 1

Chúng tôi đã xây dựng **Tầng 1** của hệ thống, có nhiệm vụ dự đoán các nhãn cha từ abstract.
1.  **Mã hóa Văn bản:** Dữ liệu abstract được mã hóa bằng 3 phương pháp để so sánh: `Bag-of-Words (BoW)`, `TF-IDF`, và `Sentence Embeddings (all-MiniLM-L6-v2)`.
2.  **Huấn luyện:** 5 mô hình Machine Learning hiệu năng cao (`KNN`, `DecisionTree`, `RandomForest`, `XGBoost`, `LightGBM`) đã được huấn luyện. Do tính chất đa nhãn, các mô hình boosting được bọc trong `OneVsRestClassifier`.
3.  **Đánh giá:** Chúng tôi sử dụng nhiều độ đo, bao gồm `Subset Accuracy` (độ chính xác khắt khe, yêu cầu đoán đúng toàn bộ tập hợp nhãn) và `Comparative Accuracy` (để so sánh tương đối với cách làm cũ).

---

#### 3. Kết Quả Chi Tiết và Diễn Giải

Toàn bộ quá trình huấn luyện 5 mô hình trên 3 loại dữ liệu (tổng cộng 15 lần chạy) mất khoảng **1 giờ 7 phút** trên Google Colab.

**Bảng xếp hạng kết quả:**
```
           Model    Encoding  Comparative Accuracy  Subset Accuracy  F1 Score (Samples)  Jaccard Score (Samples)
0            KNN  Embeddings                0.5814           0.4059              0.6782                   0.6083
1        XGBoost  Embeddings                0.4984           0.3296              0.5949                   0.5272
2       LightGBM         BoW                0.4674           0.3177              0.5727                   0.5077
...          ...         ...                   ...              ...                 ...                      ...
```

**Phân tích kết quả:**

1.  **Sự Kết Hợp Tốt Nhất:** **KNN** kết hợp với **Sentence Embeddings** cho kết quả vượt trội trên mọi chỉ số. Điều này cho thấy `Embeddings` đã tạo ra một không gian vector giàu ngữ nghĩa, và `KNN` (thuật toán dựa trên khoảng cách) đã tận dụng rất tốt không gian đó để tìm ra các bài báo tương tự.

2.  **So Sánh Với Cách Làm Cũ:**
    *   `Comparative Accuracy` cao nhất đạt **58.14%**. Con số này có vẻ thấp hơn so với `accuracy` (~88%) của project cũ, nhưng đây là một kết quả **rất tốt**.
    *   **Lý do:** Mô hình mới đang giải quyết một bài toán khó hơn rất nhiều (17 lớp đa nhãn vs. 5 lớp đơn nhãn). Tỷ lệ đoán mò chỉ là ~5.8%, mô hình của chúng ta làm tốt hơn gấp 10 lần. Việc so sánh trực tiếp là khập khiễng.

3.  **Hiệu Suất Thực Tế:**
    *   `Subset Accuracy` đạt **40.59%**, nghĩa là mô hình có khả năng dự đoán đúng hoàn toàn một tập hợp các nhãn (kể cả các nhãn phức tạp như `['cs', 'math']`) trong hơn 40% trường hợp. Đây là một con số rất ấn tượng.
    *   `F1 Score` và `Jaccard Score` đều cao (lần lượt là **67.8%** và **60.8%**), cho thấy mô hình dự đoán đúng phần lớn các nhãn cho mỗi bài báo, chứng tỏ khả năng nhận diện liên ngành rất tốt.

**Kết luận:**
Quy trình tiền xử lý và tạo dataset cân bằng đã thành công. Chúng ta đã xây dựng được một mô hình Tầng 1 mạnh mẽ, có khả năng phân loại đa nhãn hiệu quả, vượt xa khả năng của phương pháp tiếp cận đơn giản ban đầu. Nền tảng này đã sẵn sàng để tiếp tục xây dựng các mô hình Tầng 2 nhằm phân loại chi tiết các nhãn con.

### Tóm Tắt Chi Tiết Quy Trình và Kết Quả Dự Án (Xây dựng đầy đủ 2 tầng)

#### 1. Xây Dựng và Chuẩn Bị Dữ Liệu

Quy trình bắt đầu từ bộ dữ liệu gốc hơn 2.2 triệu bài báo, vốn rất lớn và mất cân bằng. Chúng tôi đã thực hiện các bước sau để tạo ra một tập dữ liệu chất lượng cao cho việc huấn luyện:

1.  **Phân Cấp Nhãn (Cha-Con):**
    *   **Phương pháp:** Chúng tôi đã phát triển một quy trình tự động để xác định các lĩnh vực khoa học lớn (Nhãn Cha). Bằng cách quét qua 3.8 triệu lượt gán nhãn, chúng tôi trích xuất các tiền tố (prefix) trước dấu `.` hoặc `-` (ví dụ: `math.CO` -> `math`).
    *   **Lựa chọn:** Chỉ những tiền tố chiếm hơn 0.1% "thị phần" trong tổng số các chủ đề mới được công nhận là Nhãn Cha. Quá trình này đã xác định được **17 Nhãn Cha** chính, tạo ra một cấu trúc phân cấp có ý nghĩa.

2.  **Tạo Dataset Con Cân Bằng (30,000 mẫu):**
    *   **Mục tiêu:** Tạo ra một bộ dữ liệu nhỏ hơn, dễ quản lý và **ít thiên vị** nhất có thể.
    *   **Chiến lược:** Chúng tôi đã áp dụng một phương pháp lấy mẫu hai chiều phức tạp để đảm bảo bộ dữ liệu 30,000 mẫu cuối cùng (`arxiv_perfectly_balanced.csv`) đạt được hai mục tiêu cân bằng quan trọng:
        *   **Cân bằng Cấu trúc:** Tỷ lệ bài báo **đơn nhãn (50.0%)** và **đa nhãn (50.0%)** được giữ ở mức cân bằng hoàn hảo.
        *   **Cân bằng Lớp:** Sự chênh lệch về số lượng mẫu giữa 17 lớp cha được giảm thiểu đáng kể, giúp mô hình học một cách công bằng hơn.

#### 2. Kiến Trúc Mô Hình Phân Cấp Hai Tầng

Chúng tôi đã xây dựng và huấn luyện một hệ thống gồm hai tầng:

*   **Tầng 1 (Dự đoán Nhãn Cha):**
    *   **Nhiệm vụ:** Nhận một `abstract` và dự đoán một hoặc nhiều trong số 17 Nhãn Cha.
    *   **Công nghệ:** Chúng tôi sử dụng mô hình `LightGBM` (bọc trong `OneVsRestClassifier` để xử lý đa nhãn) và mã hóa văn bản bằng `Sentence Embeddings` (mô hình `E5-base`) để tạo ra các vector ngữ nghĩa chất lượng cao.

*   **Tầng 2 (Dự đoán Nhãn Con):**
    *   **Nhiệm vụ:** Với mỗi Nhãn Cha được dự đoán từ Tầng 1, một mô hình con chuyên biệt sẽ được kích hoạt để dự đoán các Nhãn Con chi tiết.
    *   **Công nghệ:** Chúng tôi đã huấn luyện **15 mô hình `LightGBM` riêng biệt**, mỗi mô hình là một "chuyên gia" cho một lĩnh vực lớn (ví dụ: một mô hình cho `math`, một cho `cs`, v.v.).

#### 3. Kết Quả Đánh Giá Hiệu Suất

Sau khi huấn luyện, toàn bộ hệ thống đã được đánh giá trên một tập test gồm 5,999 bài báo.

**Kết quả định lượng:**

*   **Hiệu suất Tầng 1 (Nhãn Cha):**
    *   `Subset Accuracy`: **0.2275** (Đoán đúng hoàn toàn tập hợp nhãn cha trong 22.7% trường hợp).
    *   `F1 Score (Samples)`: **0.4880** (Trung bình, mô hình đoán đúng khoảng 49% các nhãn cha cho mỗi bài báo).

*   **Hiệu suất Toàn Hệ Thống (Nhãn Con Cuối Cùng):**
    *   `F1 Score (Samples)`: **0.2572**
    *   `Jaccard Score`: **0.2157**

**Diễn giải kết quả và Phân tích tại sao hiệu suất còn thấp:**

Kết quả F1-score cuối cùng là **25.7%** cho thấy đây là một baseline ban đầu và còn nhiều không gian để cải thiện. Nguyên nhân chính của hiệu suất còn khiêm tốn này đến từ sự cộng hưởng của nhiều yếu tố:

1.  **Lỗi Khuếch Đại từ Tầng 1:** Tầng 1 là "cửa ngõ" của hệ thống. Với F1-score chỉ 49%, nó thường xuyên dự đoán sai hoặc bỏ sót các nhãn cha. **Nếu Tầng 1 bỏ sót một nhãn cha, Tầng 2 sẽ không bao giờ có cơ hội để dự đoán các nhãn con tương ứng, gây ra lỗi dây chuyền.** Đây là điểm yếu lớn nhất của hệ thống hiện tại.

2.  **Độ Khó Cố Hữu của Bài Toán:** Việc phân loại chi tiết hàng trăm nhãn con khác nhau, đặc biệt là trong các lĩnh vực có sự chồng chéo lớn về ngôn ngữ (ví dụ: `hep-th` và `math-ph`), là một nhiệm vụ cực kỳ khó khăn.

3.  **Hiệu suất của các Mô hình Con (Tầng 2):** Mỗi mô hình con được huấn luyện trên một tập dữ liệu nhỏ hơn và có thể chưa được tối ưu hóa. Một số mô hình con có thể hoạt động rất kém, kéo hiệu suất chung đi xuống.

#### 4. Phân Tích Ví Dụ Dự Đoán Thực Tế

Để hiểu rõ hơn về hành vi của mô hình, hãy xem xét một vài ví dụ từ tập test:

*   **Ví dụ 1 (Thành công một phần, thất bại ở Tầng 2):**
    *   **Abstract:** Về "adaptive quantum circuits", "symmetry-breaking order", "gapless, local Hamiltonian".
    *   **Nhãn thật (Con):** `['cond-mat.stat-mech', 'quant-ph']`
    *   **Dự đoán Tầng 1:** `['cond-mat', 'quant']` -> **ĐÚNG HOÀN TOÀN!**
    *   **Dự đoán Tầng 2:** `[]` (trống) -> **SAI!**
    *   **Phân tích:** Tầng 1 đã hoạt động xuất sắc khi nhận diện đúng cả hai lĩnh vực. Tuy nhiên, các mô hình con của `cond-mat` và `quant` đã không đủ mạnh để xác định các chủ đề chi tiết.

*   **Ví dụ 2 (Thất bại ở Tầng 1 - Bỏ sót):**
    *   **Abstract:** Về "cellular networks", "full-duplex", "beamforming", "multi-cell network capacity".
    *   **Nhãn thật (Cha):** `['cs', 'eess', 'math']`
    *   **Dự đoán Tầng 1:** `['eess']` -> **SAI (thiếu)**. Mô hình chỉ nhận ra được khía cạnh Kỹ thuật Điện (`eess`) mà bỏ qua hoàn toàn khía cạnh Khoa học Máy tính (`cs`) và Toán học (`math`).
    *   **Phân tích:** Đây là lỗi phổ biến nhất. Do Tầng 1 bỏ sót `cs` và `math`, các mô hình con tương ứng không được kích hoạt, dẫn đến việc các nhãn con `cs.IT`, `cs.NI`, `math.IT` cũng bị bỏ lỡ.

*   **Ví dụ 3 (Thất bại ở Tầng 1 - Nhầm lẫn):**
    *   **Abstract:** Về "copositivity", "scalar potentials", "Higgs boson", "two Higgs doublet model".
    *   **Nhãn thật (Cha):** `['hep']`
    *   **Dự đoán Tầng 1:** `['gr', 'hep', 'patt-sol']` -> **SAI (thừa)**. Mô hình đã đoán đúng `hep` nhưng lại "ảo giác" ra cả Hấp dẫn Lượng tử (`gr`) và một nhãn không liên quan.
    *   **Phân tích:** Mô hình vẫn còn nhầm lẫn giữa các lĩnh vực có từ vựng tương tự nhau.

**Kết luận chung:**
Hệ thống phân cấp hai tầng hiện tại đã được xây dựng thành công và cho thấy tiềm năng trong việc xử lý bài toán phức tạp. Tuy nhiên, hiệu suất hiện tại còn hạn chế, chủ yếu do độ chính xác chưa cao của mô hình Tầng 1. Các bước cải thiện trong tương lai nên tập trung vào việc **tối ưu hóa mạnh mẽ mô hình dự đoán nhãn cha** để tạo ra một nền tảng vững chắc hơn cho Tầng 2 hoạt động.