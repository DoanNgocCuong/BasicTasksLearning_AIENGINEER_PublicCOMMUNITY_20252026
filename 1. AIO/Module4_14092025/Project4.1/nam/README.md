# **Phân Tích và Chuẩn Bị Dữ Liệu ArXiv cho Mô Hình Phân Loại**

Tài liệu này mô tả chi tiết quy trình tiền xử lý và lấy mẫu được áp dụng trên bộ dữ liệu ArXiv Abstracts Large (~2.3 triệu bài báo). Mục tiêu là biến đổi dữ liệu thô, phức tạp thành một tập dữ liệu nhỏ hơn, cân bằng và có cấu trúc, sẵn sàng cho việc huấn luyện các mô hình Machine Learning.

## **Mục Lục**
1.  [Giới Thiệu Vấn Đề](#1-giới-thiệu-vấn-đề)
2.  [Giai Đoạn 1: Phân Tích và Tạo Nhãn Cha-Con](#2-giai-đoạn-1-phân-tích-và-tạo-nhãn-cha-con)
3.  [Giai Đoạn 2: Tạo Dataset Con Cân Bằng Tối Ưu](#3-giai-đoạn-2-tạo-dataset-con-cân-bằng-tối-ưu-30000-mẫu)
4.  [Phân Tích Chi Tiết Bộ Dữ Liệu Cuối Cùng](#4-phân-tích-chi-tiết-bộ-dữ-liệu-cuối-cùng)
5.  [Luận Cứ Bảo Vệ Chiến Lược Lấy Mẫu: Tại Sao Đây Là Cách Tiếp Cận Tối Ưu?](#5-luận-cứ-bảo-vệ-chiến-lược-lấy-mẫu-tại-sao-đây-là-cách-tiếp-cận-tối-ưu)
6.  [Phân Tích Hiệu Suất và Nguyên Nhân Kết Quả Chưa Cao Hơn](#6-phân-tích-hiệu-suất-và-nguyên-nhân-kết-quả-chưa-cao-hơn)
7.  [Cách Sử Dụng File Dữ Liệu Cuối Cùng](#7-cách-sử-dụng-file-dữ-liệu-cuối-cùng)

---

## **1. Giới Thiệu Vấn Đề**

Bộ dữ liệu ArXiv gốc có những đặc điểm sau:
- **Kích thước lớn:** Hơn 2.2 triệu dòng.
- **Nhãn phức tạp:** Cột `categories` là một chuỗi văn bản, thường chứa nhiều nhãn con.
- **Tính liên ngành (Đa nhãn):** Phân tích cho thấy gần **25%** số bài báo thuộc về nhiều hơn một lĩnh vực lớn.
- **Mất cân bằng nghiêm trọng:** Một số lĩnh vực như `math` có số lượng bài báo gấp hơn 75 lần so với các lĩnh vực như `econ`.

Việc huấn luyện mô hình trực tiếp trên dữ liệu này sẽ gặp khó khăn về tài nguyên và mô hình sẽ bị thiên vị nặng về các lớp đa số. Do đó, quy trình tiền xử lý này là cực kỳ cần thiết.

---

## **2. Giai Đoạn 1: Phân Tích và Tạo Nhãn Cha-Con**

Mục tiêu của giai đoạn này là tạo ra một cấu trúc nhãn có hệ thống hơn từ các chuỗi `categories` thô, giúp mô hình có thể học ở các cấp độ trừu tượng khác nhau.

### **2.1. Quy trình tạo Nhãn Cha**

Chúng tôi nhận thấy các nhãn trong ArXiv tuân theo một quy tắc đặt tên chung: các chủ đề con thường được phân tách khỏi chủ đề chính bằng dấu chấm `.` (ví dụ: `math.CO`) hoặc dấu gạch nối `-` (ví dụ: `hep-ph`). Dựa trên quan sát này, chúng tôi đã xây dựng một quy trình tự động để xác định các "Nhãn Cha" (lĩnh vực lớn) một cách khách quan:

*   **Bước 1: Tạo "Ứng cử viên":**
    Chúng tôi quét qua toàn bộ **3.8 triệu lượt gán nhãn** trong dataset. Với mỗi nhãn con, chúng tôi trích xuất phần đứng trước dấu `.` hoặc `-` đầu tiên để tạo ra một "ứng cử viên" nhãn cha.
    *   `math.CO` → ứng cử viên là `math`.
    *   `hep-ph` → ứng cử viên là `hep`.
    *   `cond-mat.str-el` → ứng cử viên là `cond-mat`.

*   **Bước 2: Phân tích dựa trên dữ liệu:**
    Chúng tôi đếm tần suất xuất hiện của mỗi ứng cử viên. Điều này cho thấy "thị phần" của mỗi lĩnh vực trong toàn bộ không gian tri thức của ArXiv.

*   **Bước 3: Đặt ngưỡng và Quyết định:**
    Chúng tôi đặt ra một ngưỡng khách quan: một ứng cử viên phải chiếm **ít nhất 0.1%** tổng số lượt gán nhãn thì mới được công nhận là một "Nhãn Cha" thực sự. Kết quả là 17 lĩnh vực lớn đã được xác định.

### **2.2. Script Phân Tích & Bằng Chứng Lựa Chọn**
<details>
<summary>Nhấn vào đây để xem script phân tích lựa chọn Nhãn Cha</summary>

```python
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm

# Tải dữ liệu
df = pd.read_csv("data/arxiv_dataset_with_hierarchical_labels.csv")

# Quét và đếm ứng cử viên
parent_candidate_counts = Counter()
total_label_instances = 0
for cat_string in tqdm(df['categories'].dropna(), desc="Đang quét nhãn"):
    child_labels = cat_string.split(' ')
    total_label_instances += len(child_labels)
    for label in child_labels:
        candidate = None
        if '.' in label:
            candidate = label.split('.', 1)[0]
        elif '-' in label:
            candidate = label.split('-', 1)[0]
        if candidate:
            parent_candidate_counts[candidate] += 1

# Tạo bảng thống kê
stats_df = pd.DataFrame(parent_candidate_counts.items(), columns=['Ứng cử viên (Prefix)', 'Số lần xuất hiện'])
stats_df = stats_df.sort_values(by='Số lần xuất hiện', ascending=False).reset_index(drop=True)
stats_df['Tỷ lệ trên tổng số nhãn (%)'] = (stats_df['Số lần xuất hiện'] / total_label_instances) * 100

# Áp dụng ngưỡng
PARENT_THRESHOLD_PERCENTAGE = 0.1
stats_df['Được chọn làm Nhãn Cha?'] = stats_df['Tỷ lệ trên tổng số nhãn (%)'] >= PARENT_THRESHOLD_PERCENTAGE
stats_df['Được chọn làm Nhãn Cha?'] = stats_df['Được chọn làm Nhãn Cha?'].map({True: '✅ Có', False: '❌ Không'})

# In kết quả
print(f"Tổng số lượt gán nhãn: {total_label_instances:,}")
print(stats_df.head(20).to_string())
```
</details>

**Bảng kết quả phân tích:** Bảng dưới đây là bằng chứng cho thấy việc lựa chọn 17 nhãn cha là hoàn toàn dựa trên dữ liệu.

```
   Ứng cử viên (Prefix)  Số lần xuất hiện  Tỷ lệ trên tổng số nhãn (%) Được chọn làm Nhãn Cha?
0                  math            892053                      23.0613                    ✅ Có
1                    cs            780865                      20.1869                    ✅ Có
2              cond-mat            445469                      11.5163                    ✅ Có
3                   hep            409994                      10.5992                    ✅ Có
4               physics            285469                       7.3799                    ✅ Có
5              astro-ph            283171                       7.3205                    ✅ Có
...                 ...               ...                          ...                     ...
16                 econ              8196                       0.2119                    ✅ Có
17                    q              2934                       0.0758                 ❌ Không
...                 ...               ...                          ...                     ...
```

### **2.3. Kết quả Phân cấp**
Sau quá trình này, mỗi bài báo trong dataset được bổ sung thêm hai cột mới, ví dụ:

| categories | parent_labels | child_labels |
| :--- | :--- | :--- |
| `hep-ph` | `['hep']` | `['hep-ph']` |
| `math.CO cs.CG` | `['cs', 'math']` | `['cs.CG', 'math.CO']` |

---

## **3. Giai Đoạn 2: Tạo Dataset Con Cân Bằng Tối Ưu (30,000 mẫu)**

### **3.1. Thách thức và Mục tiêu kép**
Mục tiêu là tạo ra một tập dữ liệu nhỏ (~30,000 mẫu) để giải quyết hai vấn đề cùng lúc:
1.  **Cân bằng Cấu trúc:** Tỷ lệ bài báo **đơn nhãn** và **đa nhãn** phải là 50-50.
2.  **Cân bằng Lớp:** Sự chênh lệch về số lượng mẫu giữa 17 nhãn cha phải được giảm thiểu tối đa.

### **3.2. Quy trình lấy mẫu**
Chúng tôi đã áp dụng một chiến lược lấy mẫu hai giai đoạn có chủ đích:

*   **Bước 1 (Lấy mẫu Đơn nhãn):**
    Chúng tôi tạo ra một "ngân sách" 15,000 mẫu cho các bài báo đơn nhãn. Ngân sách này được chia đều cho 17 lớp cha, mỗi lớp khoảng **882 mẫu**. Chúng tôi đã lọc và lấy ngẫu nhiên chính xác số lượng mẫu này cho từng lớp.

*   **Bước 2 (Lấy mẫu Đa nhãn):**
    Chúng tôi tạo một "ngân sách" 15,000 mẫu khác cho các bài báo đa nhãn.
    1.  Với mỗi trong 17 lớp, chúng tôi lấy một lượng lớn ứng cử viên đa nhãn có chứa lớp đó.
    2.  Sau đó, chúng tôi gộp tất cả các ứng cử viên này lại và loại bỏ các bài báo trùng lặp.
    3.  Cuối cùng, chúng tôi lấy ngẫu nhiên **15,000** mẫu từ tập hợp đa nhãn duy nhất này.

### **3.3. Script Lấy Mẫu Cân Bằng**
<details>
<summary>Nhấn vào đây để xem script tạo dataset cân bằng</summary>

```python
# Script Lấy Mẫu ArXiv CÂN BẰNG
import pandas as pd
from collections import Counter
import ast
import numpy as np
from tqdm.auto import tqdm

# THAM SỐ CẤU HÌNH
TARGET_TOTAL_SAMPLES = 30000
SINGLE_MULTI_RATIO = 0.5
N_TARGET_LABELS = 17
RANDOM_STATE = 42
TOTAL_SINGLE_TARGET = int(TARGET_TOTAL_SAMPLES * SINGLE_MULTI_RATIO)
TOTAL_MULTI_TARGET = TARGET_TOTAL_SAMPLES - TOTAL_SINGLE_TARGET
SINGLE_PER_CLASS = TOTAL_SINGLE_TARGET // N_TARGET_LABELS
MULTI_PER_CLASS_INITIAL = int(TOTAL_MULTI_TARGET * 1.8 // N_TARGET_LABELS)

# Tải và xử lý dữ liệu
df = pd.read_csv("data/arxiv_dataset_with_hierarchical_labels.csv", low_memory=False)
df['parent_labels_parsed'] = df['parent_labels'].apply(ast.literal_eval)
df['num_labels'] = df['parent_labels_parsed'].apply(len)

# Xác định nhãn mục tiêu
label_counts = Counter([item for sublist in df['parent_labels_parsed'] for item in sublist])
target_parents = [label for label, count in label_counts.most_common(N_TARGET_LABELS)]

# Giai đoạn 1: Lấy mẫu đơn nhãn
df_single = df[df['num_labels'] == 1].copy()
single_samples = []
for label in tqdm(target_parents, desc="Lấy mẫu đơn nhãn"):
    mask = df_single['parent_labels_parsed'].apply(lambda x: x == [label] if x else False)
    subset = df_single[mask]
    n_to_take = min(len(subset), SINGLE_PER_CLASS)
    if n_to_take > 0:
        single_samples.append(subset.sample(n=n_to_take, random_state=RANDOM_STATE))
df_single_balanced = pd.concat(single_samples, ignore_index=True)

# Giai đoạn 2: Lấy mẫu đa nhãn
df_multi = df[df['num_labels'] > 1].copy()
multi_samples_raw = []
for label in target_parents:
    mask = df_multi['parent_labels_parsed'].apply(lambda x: label in x if x else False)
    candidates = df_multi[mask]
    n_to_take = min(len(candidates), MULTI_PER_CLASS_INITIAL)
    if n_to_take > 0:
        multi_samples_raw.append(candidates.sample(n=n_to_take, random_state=RANDOM_STATE))
df_multi_raw = pd.concat(multi_samples_raw, ignore_index=True)
df_multi_unique = df_multi_raw.drop_duplicates(subset=['id'])

if len(df_multi_unique) > TOTAL_MULTI_TARGET:
    df_multi_balanced = df_multi_unique.sample(n=TOTAL_MULTI_TARGET, random_state=RANDOM_STATE)
else:
    df_multi_balanced = df_multi_unique

# Hợp nhất và lưu
df_final = pd.concat([df_single_balanced, df_multi_balanced], ignore_index=True)
df_final = df_final.drop_duplicates(subset=['id']).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
df_final.drop(columns=['parent_labels_parsed', 'num_labels'], inplace=True, errors='ignore')
df_final.to_csv("data/arxiv_perfectly_balanced.csv", index=False)
print(f"Đã tạo file với {len(df_final)} mẫu.")
```
</details>

---

## **4. Phân Tích Chi Tiết Bộ Dữ Liệu Cuối Cùng**

### **4.1. Kết quả Tổng quan**

| Thống kê | Mục tiêu | Kết quả Thực tế | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Tổng số mẫu** | 30,000 | **29,994** | ✅ Rất gần |
| **Tỷ lệ Đơn nhãn** | 50.0% | **50.0%** (14,994 mẫu) | ✅ Hoàn hảo |
| **Tỷ lệ Đa nhãn** | 50.0% | **50.0%** (15,000 mẫu) | ✅ Hoàn hảo |
| **Độ cân bằng Lớp** | Cân bằng | **Trung bình** (CV=0.349) | ⚠️ Cần chú ý & giải quyết |

**Phân bố theo Lớp Cha:**
```
     Lớp  Số mẫu
     hep     5237
      cs     4865
    math     4223
 physics     4062
 ...         ...
    econ     2004
    cond     1819
```

### **4.2. Giải thích về sự Cân Bằng "Không Hoàn Hảo"**

Phân tích cuối cùng cho thấy mặc dù tỷ lệ đơn/đa nhãn đã cân bằng hoàn hảo, sự phân bố giữa các lớp cha vẫn chỉ ở mức "trung bình". Nguyên nhân của sự chênh lệch còn lại này là do **bản chất cố hữu của dữ liệu đa nhãn**:

1.  **Sự "Nổi Tiếng" của các Hub:** Một số nhãn như `hep`, `cs`, `math` có xu hướng là các "hub" liên ngành, chúng xuất hiện trong rất nhiều tổ hợp đa nhãn khác nhau. Khi chúng ta lấy mẫu đa nhãn, các nhãn này có xác suất được "tuyển dụng" vào bộ dữ liệu cao hơn một cách tự nhiên.
2.  **Sự Hiếm Hoi của các Lớp "Cô Đơn":** Ngược lại, các nhãn như `cond` hay `econ` ít kết hợp với các lĩnh vực khác hơn. Do đó, chúng có ít "cơ hội" hơn để được chọn trong giai đoạn lấy mẫu đa nhãn.
3.  **Ràng buộc không thể tránh khỏi:** Việc chọn một bài báo đa nhãn `['hep', 'gr']` sẽ đồng thời tăng số lượng cho cả `hep` và `gr`. Chúng ta không thể tăng số lượng cho `gr` mà không vô tình tăng thêm cho `hep`. Điều này tạo ra một bài toán tối ưu hóa tổ hợp phức tạp.

**Kết luận:** Bộ dữ liệu 30,000 mẫu này là một **sự thỏa hiệp tối ưu**. Nó đã giải quyết thành công vấn đề mất cân bằng ở cả hai cấp độ đến mức tốt nhất có thể, tạo ra một nền tảng chất lượng cao và ít thiên vị hơn đáng kể để huấn luyện và đánh giá các mô hình phân loại.

---

## **5. Luận Cứ Bảo Vệ Chiến Lược Lấy Mẫu: Tại Sao Đây Là Cách Tiếp Cận Tối Ưu?**

Đây là phần quan trọng nhất để trả lời cho câu hỏi: "Tại sao chúng ta không cân bằng tuyệt đối các lớp cha?" và "Liệu sự mất cân bằng còn lại có làm ảnh hưởng đến kết quả không?". Câu trả lời ngắn gọn là: **chiến lược này là tối ưu vì nó giải quyết vấn đề cốt lõi hơn và vấn đề còn lại đã được xử lý một cách hiệu quả ở giai đoạn huấn luyện.**

### **5.1. Ưu Tiên Chiến Lược: Cân Bằng "Kỹ Năng" Quan Trọng Hơn Cân Bằng "Tần Suất"**

Bài toán của chúng ta có hai thách thức chính:

1.  **Thách thức về Cấu trúc (Đơn nhãn vs. Đa nhãn):** Mô hình cần học hai "kỹ năng" tư duy khác nhau: nhận diện đặc điểm chuyên sâu của một lĩnh vực (đơn nhãn) và nhận diện sự giao thoa, kết hợp giữa các lĩnh vực (đa nhãn).
2.  **Thách thức về Tần suất (Lớp cha phổ biến vs. Lớp cha hiếm):** Mô hình có xu hướng học tốt hơn về các lĩnh vực phổ biến (`math`, `cs`) và bỏ qua các lĩnh vực hiếm (`econ`, `cond`).

Trong hai thách thức trên, **thách thức về cấu trúc là nền tảng và khó giải quyết hơn**. Một mô hình không được dạy cách xử lý các trường hợp đa nhãn một cách đầy đủ sẽ thất bại hoàn toàn ở nhiệm vụ cốt lõi, bất kể nó có giỏi nhận diện lớp `math` đến đâu.

Do đó, chúng tôi đã đưa ra quyết định chiến lược: **Ưu tiên giải quyết vấn đề cấu trúc một cách triệt để ở giai đoạn chuẩn bị dữ liệu** bằng cách tạo ra một bộ dữ liệu cân bằng hoàn hảo 50/50. Vấn đề về tần suất, dù quan trọng, nhưng có thể được xử lý hiệu quả bằng các công cụ kỹ thuật ở giai đoạn huấn luyện.

### **5.2. Tại Sao Cân Bằng Cả Hai Thứ Cùng Lúc Là Bất Khả Thi? Một Ví Dụ Cụ Thể**

Hãy giả sử chúng ta muốn mỗi trong 17 lớp cha đều có chính xác 2,000 mẫu và tỷ lệ đơn/đa nhãn cũng là 50/50 (1,000 đơn nhãn, 1,000 đa nhãn cho mỗi lớp).

*   **Với lớp `econ`:** Chúng ta có thể chỉ tìm thấy 800 bài báo đơn nhãn `['econ']` trong toàn bộ 2.2 triệu mẫu. Chúng ta không thể đạt được mục tiêu 1,000 mẫu đơn nhãn mà không tạo ra dữ liệu giả. **Mục tiêu thất bại.**
*   **Với lớp `hep`:** Chúng ta dễ dàng tìm được 1,000 mẫu đơn nhãn. Sau đó, chúng ta cần tìm 1,000 mẫu đa nhãn. Giả sử chúng ta tìm được 500 bài báo `['hep', 'gr']` và 500 bài báo `['hep', 'physics']`.
    *   Lúc này, lớp `hep` đã đủ 2,000 mẫu.
    *   **Nhưng lớp `gr` và `physics`** giờ đây đã vô tình nhận thêm 500 mẫu đa nhãn. Điều này sẽ phá vỡ mục tiêu 2,000 mẫu của chúng.

**Kết luận:** Bản chất liên kết của dữ liệu đa nhãn tạo ra một hiệu ứng "domino", khiến việc cân bằng đồng thời cả hai mục tiêu là một bài toán tối ưu hóa gần như không thể giải quyết mà không hy sinh một lượng lớn dữ liệu.

### **5.3. Bằng Chứng Thuyết Phục Từ Kết Quả Huấn Luyện**

Nếu sự mất cân bằng còn lại ở các lớp cha là một "lỗi" nghiêm trọng, chỉ số **Macro F1-Score** (đối xử công bằng với mọi lớp) sẽ thấp hơn đáng kể so với **Weighted F1-Score** (ưu tiên các lớp nhiều mẫu). Tuy nhiên, kết quả thực tế lại cho thấy điều ngược lại:

**Kết quả Tầng 1 (Dự đoán 17 Nhãn Cha):**
*   **F1-Score (Weighted Avg): 0.6483**
*   **F1-Score (Macro Avg):    0.6474**

Sự chênh lệch không đáng kể (**chỉ 0.0009**) là một **bằng chứng mạnh mẽ** cho thấy mô hình đang hoạt động tốt trên **cả các lớp đa số và thiểu số**. Lý do là vì trong script huấn luyện, chúng tôi đã sử dụng tham số `class_weight='balanced'`. Tham số này tự động "phạt" mô hình nặng hơn khi nó dự đoán sai một mẫu thuộc lớp hiếm, buộc nó phải học một cách công bằng.

---

## **6. Phân Tích Hiệu Suất và Nguyên Nhân Kết Quả Chưa Cao Hơn**

Với F1-score ~0.65 cho nhãn cha và ~0.40 cho nhãn con, kết quả này là một **baseline rất tốt và thực tế**, không phải là "thấp". Hiệu suất chưa cao hơn không phải do lỗi chuẩn bị dữ liệu, mà do hai nguyên nhân chính: **độ khó cố hữu của bài toán** và **giới hạn của kiến trúc mô hình hiện tại**.

### **6.1. Độ Khó Cố Hữu Của Bài Toán**

1.  **Sự Mơ Hồ và Giao Thoa Giữa Các Lĩnh Vực Khoa Học:** Ranh giới giữa các lĩnh vực học thuật thường không rõ ràng. Một bài báo về "mô phỏng va chạm hạt nhân" có thể được phân loại là `nucl-th` (Lý thuyết hạt nhân) hoặc `hep-ph` (Vật lý hạt năng lượng cao). Việc mô hình dự đoán một nhãn thay vì nhãn còn lại không hẳn là một lỗi, mà là một sự diễn giải hợp lý khác. Con người cũng có thể bất đồng trong những trường hợp này.

2.  **Thách Thức Của Phân Loại Đa Nhãn Chi Tiết (Fine-Grained):** Bài toán thực sự là dự đoán các nhãn con, với số lượng lên tới hàng trăm lớp. Đây là một bài toán cực kỳ khó. Bằng chứng là sự sụt giảm hiệu suất từ Tầng 1 xuống Tầng 2:
    *   **F1-macro (Nhãn Cha): 0.6474**
    *   **F1-macro (Nhãn Con): 0.2543**
    Chỉ số F1-macro của nhãn con thấp cho thấy mô hình đang gặp khó khăn rất lớn với các lớp con hiếm (long-tail problem), vốn chỉ có vài chục mẫu để học.

### **6.2. Giới Hạn Của Kiến Trúc Mô Hình Hiện Tại**

1.  **Điểm Mù Của TF-IDF: Thiếu Ngữ Nghĩa:** Kỹ thuật `TF-IDF` là một phương pháp "túi từ" (bag-of-words). Nó chỉ quan tâm đến tần suất của từ, chứ **không hiểu được ngữ nghĩa hay mối quan hệ giữa các từ**.
    *   **Ví dụ:** TF-IDF coi "neural network", "deep learning", và "backpropagation" là ba khái niệm hoàn toàn riêng biệt. Nó không biết rằng chúng thuộc cùng một hệ sinh thái khái niệm. Điều này hạn chế rất lớn khả năng "hiểu" sâu nội dung văn bản của mô hình.

2.  **Mô Hình Là Baseline, Chưa Được Tinh Chỉnh Sâu:** Các tham số của LightGBM (`n_estimators=100`, etc.) mới chỉ là các giá trị ban đầu để có kết quả nhanh. Để đạt hiệu suất tối đa, các mô hình cần trải qua quá trình tinh chỉnh siêu tham số (Hyperparameter Tuning) phức tạp, có thể mất hàng giờ hoặc hàng ngày để tìm ra bộ tham số tốt nhất.

### **6.3. Lộ Trình Cải Thiện: Từ Baseline Đến Hiệu Suất Cao Hơn**

Nền tảng dữ liệu đã vững chắc. Để cải thiện điểm số, các bước tiếp theo nên tập trung vào việc nâng cấp mô hình, chứ không phải lấy mẫu lại dữ liệu:

1.  **Nâng Cấp Kỹ Thuật Trích Xuất Đặc Trưng (Quan trọng nhất):** Thay thế TF-IDF bằng các kỹ thuật hiểu ngữ nghĩa như:
    *   **Word Embeddings (Word2Vec, FastText):** Đại diện mỗi từ bằng một vector, nắm bắt được quan hệ giữa các từ.
    *   **Contextual Embeddings (BERT, SciBERT):** Các mô hình Transformer có khả năng "hiểu" một từ dựa trên ngữ cảnh của cả câu, mang lại hiệu quả vượt trội cho các bài toán NLP.

2.  **Tinh Chỉnh Siêu Tham Số (Hyperparameter Tuning):** Sử dụng các công cụ như Optuna, Hyperopt hoặc GridSearch để tự động tìm ra bộ tham số tốt nhất cho LightGBM.

3.  **Tối Ưu Hóa Ngưỡng Quyết Định (Threshold Optimization):** Thay vì dùng ngưỡng 0.5 mặc định, có thể tìm ra một ngưỡng riêng cho mỗi lớp để tối đa hóa F1-score.

**Kết luận:** Bộ dữ liệu hiện tại là nền tảng tối ưu. Hiệu suất hiện tại là một baseline mạnh mẽ, phản ánh đúng độ khó của bài toán và giới hạn của kiến trúc mô hình ban đầu. Các cải tiến trong tương lai sẽ đến từ việc xây dựng một mô hình "thông minh" hơn trên nền tảng dữ liệu vững chắc này.

---

## **7. Cách Sử Dụng File Dữ Liệu Cuối Cùng**

File `arxiv_perfectly_balanced.csv` chứa dữ liệu đã được xử lý. Khi đọc file này bằng Pandas, cột `parent_labels` và `child_labels` sẽ ở dạng chuỗi. Cần sử dụng thư viện `ast` để chuyển đổi chúng trở lại thành dạng `list` trong Python trước khi sử dụng.

**Ví dụ:**```python
import pandas as pd
import ast

df = pd.read_csv("data/arxiv_perfectly_balanced.csv")
df['parent_labels'] = df['parent_labels'].apply(ast.literal_eval)
df['child_labels'] = df['child_labels'].apply(ast.literal_eval)```

# Ref:
- https://aistudio.google.com/prompts/10hvWKjqxJPZW-vmZDCRVDaC9OkHW0UBR
- https://aistudio.google.com/prompts/1Wx081g1dLhv2yotR0LhVG0D2Cji1t2pM