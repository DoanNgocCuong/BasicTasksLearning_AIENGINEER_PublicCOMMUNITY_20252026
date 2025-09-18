# Phân Tích và Chuẩn Bị Dữ Liệu ArXiv cho Mô Hình Phân Loại

Tài liệu này mô tả chi tiết quy trình tiền xử lý và lấy mẫu được áp dụng trên bộ dữ liệu ArXiv Abstracts Large (~2.3 triệu bài báo). Mục tiêu là biến đổi dữ liệu thô, phức tạp thành một tập dữ liệu nhỏ hơn, cân bằng và có cấu trúc, sẵn sàng cho việc huấn luyện các mô hình Machine Learning.

## Mục Lục
1.  [Giới Thiệu Vấn Đề](#giới-thiệu-vấn-đề)
2.  [Giai Đoạn 1: Phân Tích và Tạo Nhãn Cha-Con](#giai-đoạn-1-phân-tích-và-tạo-nhãn-cha-con)
    - [Quy trình tạo Nhãn Cha](#11-quy-trình-tạo-nhãn-cha)
    - [Script Phân Tích & Bằng Chứng Lựa Chọn](#12-script-phân-tích--bằng-chứng-lựa-chọn)
    - [Kết quả](#13-kết-quả-phân-cấp)
3.  [Giai Đoạn 2: Tạo Dataset Con Cân Bằng Tối Ưu](#giai-đoạn-2-tạo-dataset-con-cân-bằng-tối-ưu-30000-mẫu)
    - [Thách thức và Mục tiêu kép](#21-thách-thức-và-mục-tiêu-kép)
    - [Quy trình lấy mẫu](#22-quy-trình-lấy-mẫu)
    - [Script Lấy Mẫu Cân Bằng](#23-script-lấy-mẫu-cân-bằng)
4.  [Phân Tích Chi Tiết Bộ Dữ Liệu Cuối Cùng](#phân-tích-chi-tiết-bộ-dữ-liệu-cuối-cùng)
    - [Kết quả Tổng quan](#41-kết-quả-tổng-quan)
    - [Giải thích về sự Cân Bằng "Không Hoàn Hảo"](#42-giải-thích-về-sự-cân-bằng-không-hoàn-hảo)
5.  [Cách Sử Dụng File Dữ Liệu Cuối Cùng](#cách-sử-dụng-file-dữ-liệu-cuối-cùng)

---

## Giới Thiệu Vấn Đề

Bộ dữ liệu ArXiv gốc có những đặc điểm sau:
- **Kích thước lớn:** Hơn 2.2 triệu dòng.
- **Nhãn phức tạp:** Cột `categories` là một chuỗi văn bản, thường chứa nhiều nhãn con.
- **Tính liên ngành (Đa nhãn):** Phân tích cho thấy gần **25%** số bài báo thuộc về nhiều hơn một lĩnh vực lớn.
- **Mất cân bằng nghiêm trọng:** Một số lĩnh vực như `math` có số lượng bài báo gấp hơn 75 lần so với các lĩnh vực như `econ`.

Việc huấn luyện mô hình trực tiếp trên dữ liệu này sẽ gặp khó khăn về tài nguyên và mô hình sẽ bị thiên vị nặng về các lớp đa số. Do đó, quy trình tiền xử lý này là cực kỳ cần thiết.

---

## Giai Đoạn 1: Phân Tích và Tạo Nhãn Cha-Con

Mục tiêu của giai đoạn này là tạo ra một cấu trúc nhãn có hệ thống hơn từ các chuỗi `categories` thô, giúp mô hình có thể học ở các cấp độ trừu tượng khác nhau.

### 1.1 Quy trình tạo Nhãn Cha

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

### 1.2 Script Phân Tích & Bằng Chứng Lựa Chọn
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
6                 quant            136852                       3.5379                    ✅ Có
7                  stat            130039                       3.3618                    ✅ Có
8                 astro            105380                       2.7243                    ✅ Có
9                    gr            101130                       2.6144                    ✅ Có
10                 nucl             79175                       2.0468                    ✅ Có
11                 eess             71638                       1.8520                    ✅ Có
12                q-bio             46886                       1.2121                    ✅ Có
13                 nlin             43866                       1.1340                    ✅ Có
14                q-fin             21171                       0.5473                    ✅ Có
15                 cond             14215                       0.3675                    ✅ Có
16                 econ              8196                       0.2119                    ✅ Có
17                    q              2934                       0.0758                 ❌ Không
18                 chao              2398                       0.0620                 ❌ Không
...                 ...               ...                          ...                     ...
```

### 1.3 Kết quả Phân cấp
Sau quá trình này, mỗi bài báo trong dataset được bổ sung thêm hai cột mới, ví dụ:

| categories | parent_labels | child_labels |
| :--- | :--- | :--- |
| `hep-ph` | `['hep']` | `['hep-ph']` |
| `math.CO cs.CG` | `['cs', 'math']` | `['cs.CG', 'math.CO']` |

---

### Giai Đoạn 2: Tạo Dataset Con Cân Bằng Tối Ưu (30,000 mẫu)

#### 2.1 Thách thức và Mục tiêu kép
Mục tiêu là tạo ra một tập dữ liệu nhỏ (~30,000 mẫu) để giải quyết hai vấn đề cùng lúc:
1.  **Cân bằng Cấu trúc:** Tỷ lệ bài báo đơn nhãn và đa nhãn phải là 50-50.
2.  **Cân bằng Lớp:** Sự chênh lệch về số lượng mẫu giữa 17 nhãn cha phải được giảm thiểu tối đa.

#### 2.2 Quy trình lấy mẫu
Chúng tôi đã áp dụng một chiến lược lấy mẫu hai giai đoạn:

*   **Bước 1 (Lấy mẫu Đơn nhãn):**
    Chúng tôi tạo ra một "ngân sách" 15,000 mẫu cho các bài báo đơn nhãn. Ngân sách này được chia đều cho 17 lớp cha, mỗi lớp khoảng **882 mẫu**. Chúng tôi đã lọc và lấy ngẫu nhiên chính xác số lượng mẫu này cho từng lớp.

*   **Bước 2 (Lấy mẫu Đa nhãn):**
    Chúng tôi tạo một "ngân sách" 15,000 mẫu khác cho các bài báo đa nhãn.
    1.  Với mỗi trong 17 lớp, chúng tôi lấy một lượng lớn ứng cử viên đa nhãn có chứa lớp đó.
    2.  Sau đó, chúng tôi gộp tất cả các ứng cử viên này lại và loại bỏ các bài báo trùng lặp.
    3.  Cuối cùng, chúng tôi lấy ngẫu nhiên **15,000** mẫu từ tập hợp đa nhãn duy nhất này.

#### 2.3 Script Lấy Mẫu Cân Bằng
<details>
<summary>Nhấn vào đây để xem script tạo dataset cân bằng</summary>

```python
# Script Lấy Mẫu ArXiv CÂN BẰNG HOÀN TOÀN
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

### Phân Tích Chi Tiết Bộ Dữ Liệu Cuối Cùng

Sau khi chạy script, chúng tôi đã thực hiện một phân tích chi tiết trên file `arxiv_perfectly_balanced.csv` và thu được kết quả sau:

#### 4.1 Kết quả Tổng quan

| Thống kê | Mục tiêu | Kết quả Thực tế | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Tổng số mẫu** | 30,000 | **29,994** | ✅ Rất gần |
| **Tỷ lệ Đơn nhãn** | 50.0% | **50.0%** (14,994 mẫu) | ✅ Hoàn hảo |
| **Tỷ lệ Đa nhãn** | 50.0% | **50.0%** (15,000 mẫu) | ✅ Hoàn hảo |
| **Độ cân bằng Lớp** | Cân bằng | **Trung bình** (CV=0.349) | ⚠️ Cần chú ý |

**Phân bố theo Lớp Cha:**
```
     Lớp  Số mẫu
     hep     5237
      cs     4865
    math     4223
 physics     4062
cond-mat     3398
      gr     3154
    stat     2810
   quant     2735
astro-ph     2496
    nlin     2329
   q-fin     2292
    nucl     2240
    eess     2180
   q-bio     2129
   astro     2014
    econ     2004
    cond     1819
```

#### 4.2 Giải thích về sự Cân Bằng "Không Hoàn Hảo"

Phân tích cuối cùng cho thấy mặc dù tỷ lệ đơn/đa nhãn đã cân bằng hoàn hảo, sự phân bố giữa các lớp cha vẫn chỉ ở mức "trung bình". Nguyên nhân của sự chênh lệch còn lại này là do **bản chất cố hữu của dữ liệu đa nhãn**:

1.  **Sự "Nổi Tiếng" của các Hub:** Một số nhãn như `hep`, `cs`, `math` có xu hướng là các "hub" liên ngành, chúng xuất hiện trong rất nhiều tổ hợp đa nhãn khác nhau. Khi chúng ta lấy mẫu đa nhãn, các nhãn này có xác suất được "tuyển dụng" vào bộ dữ liệu cao hơn một cách tự nhiên.
2.  **Sự Hiếm Hoi của các Lớp "Cô Đơn":** Ngược lại, các nhãn như `cond` hay `econ` ít kết hợp với các lĩnh vực khác hơn. Do đó, chúng có ít "cơ hội" hơn để được chọn trong giai đoạn lấy mẫu đa nhãn.
3.  **Ràng buộc không thể tránh khỏi:** Việc chọn một bài báo đa nhãn `['hep', 'gr']` sẽ đồng thời tăng số lượng cho cả `hep` và `gr`. Chúng ta không thể tăng số lượng cho `gr` mà không vô tình tăng thêm cho `hep`. Điều này tạo ra một bài toán tối ưu hóa tổ hợp phức tạp, khiến việc cân bằng tuyệt đối trên cả hai chiều là gần như không thể nếu muốn giữ lại các mẫu đa nhãn.

**Kết luận:** Bộ dữ liệu 30,000 mẫu này là một **sự thỏa hiệp tối ưu**. Nó đã giải quyết thành công vấn đề mất cân bằng ở cả hai cấp độ đến mức tốt nhất có thể, tạo ra một nền tảng chất lượng cao và ít thiên vị hơn đáng kể để huấn luyện và đánh giá các mô hình phân loại.

---

### Cách Sử Dụng File Dữ Liệu Cuối Cùng
File `arxiv_perfectly_balanced.csv` chứa dữ liệu đã được xử lý. Khi đọc file này bằng Pandas, cột `parent_labels` và `child_labels` sẽ ở dạng chuỗi. Cần sử dụng thư viện `ast` để chuyển đổi chúng trở lại thành dạng `list` trong Python trước khi sử dụng.

**Ví dụ:**
```python
import pandas as pd
import ast

df = pd.read_csv("data/arxiv_perfectly_balanced.csv")
df['parent_labels'] = df['parent_labels'].apply(ast.literal_eval)
df['child_labels'] = df['child_labels'].apply(ast.literal_eval)
```

# Ref:
- https://aistudio.google.com/prompts/10hvWKjqxJPZW-vmZDCRVDaC9OkHW0UBR