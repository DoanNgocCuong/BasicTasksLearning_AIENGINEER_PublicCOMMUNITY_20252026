### Tóm Tắt Chi Tiết Quy Trình và Kết Quả Dự Án

Tài liệu này ghi lại quá trình cải tiến bài toán phân loại chủ đề bài báo ArXiv, chuyển từ một phương pháp tiếp cận đơn giản sang một hệ thống phân cấp đa nhãn tinh vi hơn. Mục tiêu là xây dựng một mô hình không chỉ dự đoán đúng lĩnh vực mà còn có khả năng nhận diện tính liên ngành của khoa học.

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