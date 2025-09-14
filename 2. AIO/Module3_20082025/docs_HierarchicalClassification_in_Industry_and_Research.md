# Bảng So Sánh Chi Tiết Các Approaches trong Hierarchical Classification

## 1. So Sánh Kiến Trúc Chính




### 1.1. Flat Classification

| Đặc điểm | Mô tả chi tiết |
|----------|-----------------|
| **Mô tả** | Trong phương pháp này, mỗi lớp trong hệ thống phân cấp được coi là một lớp độc lập, không có mối quan hệ phân cấp nào được tính đến. Một bộ phân loại duy nhất được huấn luyện để phân loại các mẫu vào một trong số các lớp này. Đây là cách tiếp cận đơn giản nhất và thường được dùng làm đường cơ sở (baseline) để so sánh hiệu suất với các phương pháp phân cấp phức tạp hơn. |
| **Ưu điểm** | - **Đơn giản trong triển khai:** Dễ dàng cài đặt và sử dụng với các thuật toán phân loại tiêu chuẩn (ví dụ: SVM, Logistic Regression, Neural Networks).<br>- **Tận dụng các mô hình chuẩn:** Có thể sử dụng bất kỳ mô hình phân loại đa lớp nào mà không cần sửa đổi đặc biệt.<br>- **Tốc độ huấn luyện nhanh:** Đối với các tập dữ liệu không quá lớn, quá trình huấn luyện thường nhanh chóng vì chỉ có một mô hình cần được tối ưu hóa. |
| **Nhược điểm** | - **Bỏ qua thông tin phân cấp:** Không tận dụng được mối quan hệ giữa các lớp, dẫn đến việc mất mát thông tin quan trọng và có thể tạo ra các dự đoán không nhất quán (ví dụ: phân loại một đối tượng là 'mèo' nhưng không phải là 'động vật').<br>- **Hiệu suất kém:** Thường cho hiệu suất thấp hơn đáng kể so với các phương pháp phân cấp, đặc biệt khi số lượng lớp lớn và có cấu trúc phân cấp rõ ràng.<br>- **Khó mở rộng:** Khi số lượng lớp tăng lên, độ phức tạp của mô hình tăng theo, có thể dẫn đến vấn đề về bộ nhớ và thời gian tính toán. |
| **Trường hợp sử dụng** | - **So sánh đường cơ sở (Baseline comparison):** Được sử dụng để thiết lập một điểm tham chiếu hiệu suất tối thiểu.<br>- **Phân loại với hệ thống phân cấp nông:** Khi hệ thống phân cấp chỉ có một hoặc hai cấp độ và mối quan hệ giữa các lớp không quá phức tạp.<br>- **Các bài toán phân loại đơn giản:** Khi yêu cầu về độ chính xác không quá cao và sự đơn giản là ưu tiên hàng đầu. |
| **Hiệu suất** | Trong nghiên cứu của Stanford về phân loại sản phẩm Amazon, Flat Classification đạt độ chính xác từ 2.8-3.01%. Điều này cho thấy sự kém hiệu quả rõ rệt khi bỏ qua cấu trúc phân cấp. |
| **Nguồn** | [Stanford Amazon Study](https://cs229.stanford.edu/proj2014/Bin%20Wang,%20Shaoming%20Feng,%20Hierarchical%20Classification%20of%20Amazon%20Products.pdf) - Nghiên cứu này phân tích các phương pháp phân loại phân cấp cho sản phẩm của Amazon, cung cấp cái nhìn sâu sắc về hiệu suất của Flat Classification trong bối cảnh thực tế. |




### 1.2. LCPN (Top-down) - Local Classifier Per Node

| Đặc điểm | Mô tả chi tiết |
|----------|-----------------|
| **Mô tả** | LCPN là một phương pháp phân loại phân cấp theo hướng từ trên xuống (top-down). Trong cách tiếp cận này, một bộ phân loại riêng biệt được huấn luyện cho mỗi nút (node) trong cây phân cấp. Khi một mẫu được phân loại, nó sẽ đi qua từng cấp độ của cây, bắt đầu từ gốc. Tại mỗi nút, bộ phân loại cục bộ sẽ quyết định đường dẫn tiếp theo xuống các nút con. Quá trình này tiếp tục cho đến khi đạt đến một nút lá (leaf node) hoặc một cấp độ mong muốn. |
| **Ưu điểm** | - **Dễ dàng gỡ lỗi (debug) và mở rộng (scale):** Do mỗi bộ phân loại hoạt động độc lập, việc xác định lỗi và mở rộng hệ thống (thêm hoặc bớt các lớp/nút) trở nên dễ dàng hơn.<br>- **Dễ hiểu (Interpretable):** Quá trình phân loại có thể được theo dõi từng bước, giúp hiểu rõ hơn về cách mô hình đưa ra quyết định.<br>- **Đã được chứng minh trong ngành (Industry proven):** Phương pháp này đã được áp dụng rộng rãi và thành công trong nhiều ứng dụng thực tế, đặc biệt là trong các hệ thống phân loại lớn. |
| **Nhược điểm** | - **Lan truyền lỗi (Error propagation):** Một lỗi phân loại ở cấp độ cao hơn sẽ ảnh hưởng đến tất cả các cấp độ thấp hơn. Nếu một mẫu bị phân loại sai ở một nút cha, nó sẽ không bao giờ có thể được phân loại đúng ở các nút con, bất kể các bộ phân loại con có chính xác đến đâu.<br>- **Nhiều mô hình:** Yêu cầu huấn luyện và duy trì nhiều mô hình phân loại (một cho mỗi nút), điều này có thể tốn kém về tài nguyên tính toán và thời gian. |
| **Trường hợp sử dụng** | - **Phân loại sản phẩm thương mại điện tử (E-commerce taxonomy):** Rất phù hợp cho việc tổ chức và phân loại hàng triệu sản phẩm vào các danh mục phức tạp.<br>- **Phân loại tài liệu:** Sắp xếp các tài liệu vào các chủ đề và chủ đề phụ.<br>- **Phân loại sinh học:** Phân loại các loài sinh vật theo hệ thống phân loại Linnaean. |
| **Hiệu suất** | Trong nghiên cứu về phân loại sản phẩm Amazon, LCPN (Greedy Search) cho thấy độ chính xác giảm dần từ 55.32% ở cấp độ 1 (L1) xuống 11.79% ở cấp độ 8 (L8). Điều này minh họa rõ ràng vấn đề lan truyền lỗi khi đi sâu vào hệ thống phân cấp. |
| **Nguồn** | [Stanford Amazon Study](https://cs229.stanford.edu/proj2014/Bin%20Wang,%20Shaoming%20Feng,%20Hierarchical%20Classification%20of%20Amazon%20Products.pdf) - Nghiên cứu này cung cấp phân tích chi tiết về hiệu suất của LCPN trong bối cảnh phân loại sản phẩm quy mô lớn. |




### 1.3. Multi-task Learning (MTL)

| Đặc điểm | Mô tả chi tiết |
|----------|-----------------|
| **Mô tả** | Multi-task Learning là một phương pháp học máy trong đó nhiều nhiệm vụ liên quan được học đồng thời. Trong bối cảnh phân loại phân cấp, các nhiệm vụ có thể là phân loại ở các cấp độ khác nhau của hệ thống phân cấp hoặc phân loại các thuộc tính liên quan. Bằng cách chia sẻ các biểu diễn (representations) giữa các nhiệm vụ, MTL có thể tận dụng thông tin chung và cải thiện hiệu suất tổng thể, đặc biệt là đối với các nhiệm vụ có ít dữ liệu hơn. |
| **Ưu điểm** | - **Huấn luyện chung (Joint training):** Các nhiệm vụ được huấn luyện cùng lúc, cho phép mô hình học các đặc trưng chung hữu ích cho tất cả các nhiệm vụ.<br>- **Giảm lan truyền lỗi:** So với các phương pháp top-down, MTL có thể giảm thiểu vấn đề lan truyền lỗi vì các quyết định phân loại ở các cấp độ khác nhau được đưa ra đồng thời hoặc có sự ảnh hưởng lẫn nhau, thay vì phụ thuộc hoàn toàn vào quyết định của cấp trên.<br>- **Biểu diễn chia sẻ (Shared representation):** Mô hình học được một biểu diễn dữ liệu mạnh mẽ hơn bằng cách tổng hợp thông tin từ nhiều nhiệm vụ, giúp cải thiện khả năng khái quát hóa. |
| **Nhược điểm** | - **Kiến trúc phức tạp:** Thiết kế và triển khai mô hình MTL có thể phức tạp hơn so với các mô hình đơn nhiệm, đòi hỏi sự hiểu biết sâu sắc về mối quan hệ giữa các nhiệm vụ.<br>- **Khó điều chỉnh (Tuning difficulty):** Việc tìm kiếm các siêu tham số (hyperparameters) tối ưu và cân bằng trọng số giữa các nhiệm vụ có thể rất thách thức và tốn thời gian. |
| **Trường hợp sử dụng** | - **Chăm sóc sức khỏe (Healthcare):** Phát hiện nhiều loại bệnh hoặc tình trạng y tế từ cùng một bộ dữ liệu.<br>- **Xử lý ngôn ngữ tự nhiên (NLP):** Các nhiệm vụ như gắn thẻ từ loại (POS tagging), nhận dạng thực thể có tên (NER) và phân tích cảm xúc có thể được học cùng lúc.<br>- **Phân loại hình ảnh:** Phân loại đối tượng và thuộc tính của đối tượng trong cùng một hình ảnh. |
| **Hiệu suất** | Trong một nghiên cứu về phát hiện rối loạn nhịp tim (Arrhythmia Detection), MTL đạt độ chính xác 95% cho phân loại nhị phân và 88% cho phân loại đa nhãn, cho thấy hiệu quả cao trong các bài toán y tế phức tạp. |
| **Nguồn** | [Arrhythmia Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC11481069/) - Bài báo này minh họa ứng dụng của MTL trong việc phân tích dữ liệu điện tâm đồ (ECG) để phát hiện các bất thường về nhịp tim. |




### 1.4. Hierarchical SVM

| Đặc điểm | Mô tả chi tiết |
|----------|-----------------|
| **Mô tả** | Hierarchical Support Vector Machine (HSVM) là một mở rộng của máy vector hỗ trợ (SVM) truyền thống, được thiết kế để tận dụng cấu trúc phân cấp của dữ liệu. Thay vì huấn luyện một SVM duy nhất cho tất cả các lớp, HSVM thường xây dựng một cây các bộ phân loại SVM, trong đó mỗi bộ phân loại giải quyết một nhiệm vụ phân loại nhị phân hoặc đa lớp cục bộ tại một nút hoặc một cấp độ cụ thể trong hệ thống phân cấp. Phương pháp này tích hợp thông tin phân cấp vào quá trình tối ưu hóa, thường thông qua các ràng buộc hoặc điều chỉnh (regularization) để đảm bảo tính nhất quán của các dự đoán theo cấu trúc phân cấp. |
| **Ưu điểm** | - **Tối ưu hóa lồi (Convex optimization):** Giống như SVM truyền thống, HSVM thường dựa trên các bài toán tối ưu hóa lồi, đảm bảo tìm được nghiệm tối ưu toàn cục và có cơ sở lý thuyết vững chắc.<br>- **Đảm bảo lý thuyết (Theoretical guarantees):** Có các đảm bảo về hiệu suất và khả năng khái quát hóa, đặc biệt trong các trường hợp dữ liệu có thể phân tách tuyến tính hoặc phi tuyến tính.<br>- **Chuyển giao trực giao (Orthogonal transfer):** Một số biến thể của HSVM, như Orthogonal Transfer SVM, cho phép chuyển giao kiến thức giữa các cấp độ phân cấp một cách hiệu quả, giúp cải thiện hiệu suất ở các cấp độ thấp hơn mà không làm suy giảm hiệu suất ở các cấp độ cao hơn. |
| **Nhược điểm** | - **Giới hạn ở SVM:** Phương pháp này bị ràng buộc bởi các đặc điểm của SVM, có thể không phải là lựa chọn tốt nhất cho mọi loại dữ liệu hoặc cấu trúc phân cấp. Việc mở rộng sang các mô hình học sâu có thể phức tạp.<br>- **Điều chỉnh phức tạp (Complex regularization):** Việc thiết kế và điều chỉnh các hàm điều chỉnh để tích hợp thông tin phân cấp một cách hiệu quả có thể rất phức tạp và đòi hỏi kiến thức chuyên sâu. |
| **Trường hợp sử dụng** | - **Phân loại tài liệu (Document classification):** Đặc biệt hiệu quả trong việc tổ chức các bộ sưu tập tài liệu lớn theo các chủ đề và chủ đề phụ.<br>- **Phân loại sinh học:** Phân loại các loài sinh vật hoặc gen theo hệ thống phân loại.<br>- **Các bài toán phân loại có cấu trúc:** Khi có một cấu trúc rõ ràng và chặt chẽ giữa các lớp. |
| **Hiệu suất** | Orthogonal Transfer SVM đã đạt được hiệu suất 


được coi là state-of-the-art (SOTA) trên bộ dữ liệu RCV1, một bộ dữ liệu phân loại văn bản lớn. Điều này nhấn mạnh khả năng của nó trong việc xử lý các tác vụ phân loại phân cấp phức tạp với hiệu quả cao. |
| **Nguồn** | [Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ohsvm_techreport.pdf) - Nghiên cứu này từ Microsoft trình bày chi tiết về Orthogonal Transfer SVM và các ứng dụng của nó. |




### 1.5. Graph-based (GNN+Transformer)

| Đặc điểm | Mô tả chi tiết |
|----------|-----------------|
| **Mô tả** | Phương pháp này kết hợp sức mạnh của Mạng nơ-ron đồ thị (Graph Neural Networks - GNNs) và kiến trúc Transformer để xử lý các bài toán phân loại phân cấp trên dữ liệu có cấu trúc đồ thị. GNNs xuất sắc trong việc học biểu diễn từ dữ liệu đồ thị bằng cách tổng hợp thông tin từ các nút lân cận, trong khi Transformer nổi tiếng với khả năng nắm bắt các phụ thuộc dài hạn và xử lý dữ liệu tuần tự hoặc không tuần tự một cách hiệu quả. Sự kết hợp này cho phép mô hình tận dụng cả cấu trúc phân cấp rõ ràng (từ GNN) và các mối quan hệ phức tạp, phi tuyến tính giữa các lớp (từ Transformer). |
| **Ưu điểm** | - **State-of-the-art (SOTA):** Thường đạt được hiệu suất vượt trội trên nhiều bộ dữ liệu phân loại phân cấp phức tạp, đặc biệt là khi dữ liệu có thể được biểu diễn dưới dạng đồ thị.<br>- **Biểu diễn phong phú:** Có khả năng học các biểu diễn ngữ cảnh và cấu trúc rất phong phú từ dữ liệu, giúp nắm bắt các mối quan hệ phức tạp giữa các lớp và các mẫu.<br>- **Khả năng mở rộng (Scalable):** Các kiến trúc GNN và Transformer hiện đại đã được tối ưu hóa để xử lý các đồ thị và tập dữ liệu lớn, cho phép áp dụng cho các hệ thống phân loại quy mô lớn. |
| **Nhược điểm** | - **Độ phức tạp cao:** Kiến trúc mô hình rất phức tạp, đòi hỏi tài nguyên tính toán lớn để huấn luyện và tinh chỉnh.<br>- **Khó triển khai:** Việc thiết kế, huấn luyện và triển khai các mô hình kết hợp GNN và Transformer đòi hỏi kiến thức chuyên sâu về cả hai lĩnh vực và kỹ năng kỹ thuật cao. |
| **Trường hợp sử dụng** | - **Đồ thị quy mô lớn (Large-scale graphs):** Phân loại các nút trong các đồ thị mạng xã hội, mạng tri thức, hoặc hệ thống khuyến nghị.<br>- **Phân loại văn bản phân cấp:** Khi các tài liệu có thể được biểu diễn dưới dạng đồ thị từ khóa hoặc đồ thị trích dẫn.<br>- **Phân loại sinh học:** Phân tích các mạng tương tác protein hoặc gen. |
| **Hiệu suất** | Các phương pháp kết hợp GNN và Transformer đã đạt được hiệu suất SOTA trên các đồ thị với hàng triệu nút, chứng tỏ khả năng xử lý các bài toán phân loại phân cấp cực kỳ lớn và phức tạp. |
| **Nguồn** | [IJCAI 2023](https://dl.acm.org/doi/10.24963/ijcai.2023/523) - Hội nghị IJCAI thường xuyên công bố các nghiên cứu tiên tiến về AI, bao gồm các ứng dụng của GNN và Transformer trong phân loại phân cấp. |




## 2. So Sánh Industry vs Research

| Khía cạnh | Industry (Công nghiệp) | Research (Nghiên cứu) |
|----------|------------------------|-----------------------|
| **Ưu tiên hàng đầu** | - **Tính ổn định (Stability):** Các giải pháp phải hoạt động đáng tin cậy trong môi trường sản xuất, ít lỗi và dễ bảo trì.<br>- **Tốc độ ra thị trường (Speed-to-market):** Khả năng triển khai nhanh chóng để đáp ứng nhu cầu kinh doanh và cạnh tranh.<br>- **Hiệu quả chi phí (Cost-effectiveness):** Giải pháp phải tối ưu về chi phí vận hành và tài nguyên.<br>- **Khả năng mở rộng (Scalability):** Có thể xử lý lượng dữ liệu và người dùng lớn mà không ảnh hưởng đến hiệu suất. | - **Độ chính xác (Accuracy):** Mục tiêu chính là đạt được hiệu suất cao nhất có thể, thường vượt qua các phương pháp hiện có.<br>- **Tính đổi mới (Innovation):** Phát triển các thuật toán, mô hình hoặc phương pháp tiếp cận mới để giải quyết các vấn đề chưa được giải quyết hoặc cải thiện đáng kể các giải pháp hiện có.<br>- **Khám phá kiến thức mới:** Mở rộng hiểu biết về lĩnh vực, tìm ra các nguyên lý cơ bản hoặc các mối quan hệ mới. |
| **Phương pháp ưa thích** | - **LCPN (Local Classifier Per Node):** Do tính dễ hiểu, dễ gỡ lỗi và khả năng mở rộng trong các hệ thống phân cấp lớn.<br>- **Flat Classification + Hậu xử lý (post-processing):** Sử dụng các mô hình đơn giản kết hợp với các quy tắc nghiệp vụ hoặc thuật toán hậu xử lý để đảm bảo tính nhất quán phân cấp và đáp ứng yêu cầu kinh doanh.<br>- **Các mô hình đã được chứng minh:** Ưu tiên các phương pháp đã được kiểm chứng và có tài liệu tốt, dễ dàng tích hợp vào hệ thống hiện có. | - **GNN+Transformer:** Các kiến trúc học sâu tiên tiến, đặc biệt là khi dữ liệu có cấu trúc đồ thị hoặc cần nắm bắt các mối quan hệ phức tạp.<br>- **Multi-task Learning (MTL):** Để tận dụng thông tin chung giữa các nhiệm vụ và cải thiện hiệu suất tổng thể.<br>- **Các phương pháp mới nổi:** Luôn tìm kiếm và thử nghiệm các kỹ thuật mới nhất để đẩy giới hạn của công nghệ. |
| **Các chỉ số đánh giá** | - **Tác động kinh doanh (Business impact):** Các chỉ số trực tiếp liên quan đến doanh thu, lợi nhuận, trải nghiệm người dùng (ví dụ: tỷ lệ chuyển đổi, giảm tỷ lệ trả hàng).<br>- **Độ trễ (Latency):** Thời gian phản hồi của hệ thống, đặc biệt quan trọng trong các ứng dụng thời gian thực.<br>- **Tỷ lệ lỗi (Error rate):** Số lượng lỗi phân loại, đặc biệt là các lỗi nghiêm trọng ảnh hưởng đến trải nghiệm người dùng.<br>- **Chi phí vận hành (Operational cost):** Chi phí tính toán và lưu trữ để duy trì hệ thống. | - **Hierarchical F1-score:** Một biến thể của F1-score có tính đến cấu trúc phân cấp, đánh giá độ chính xác của phân loại ở các cấp độ khác nhau.<br>- **Tree distance / Path length:** Đo lường khoảng cách giữa dự đoán và nhãn thực tế trên cây phân cấp.<br>- **Độ chính xác (Accuracy), Độ thu hồi (Recall), Độ chính xác (Precision):** Các chỉ số truyền thống, nhưng thường được phân tích theo từng cấp độ hoặc theo các lớp cụ thể.<br>- **Khả năng khái quát hóa (Generalization ability):** Hiệu suất trên các tập dữ liệu mới, chưa từng thấy. |
| **Dữ liệu** | - **Độc quyền (Proprietary):** Thường làm việc với các bộ dữ liệu lớn, độc quyền của công ty, có thể chứa thông tin nhạy cảm và yêu cầu bảo mật cao.<br>- **Quy mô lớn (Large-scale):** Dữ liệu thường rất lớn, đòi hỏi các giải pháp có khả năng xử lý phân tán và hiệu quả. | - **Bộ dữ liệu công khai (Public benchmarks):** Sử dụng các bộ dữ liệu tiêu chuẩn được công nhận trong cộng đồng nghiên cứu để so sánh công bằng với các phương pháp khác.<br>- **Dữ liệu tổng hợp (Synthetic data):** Có thể tạo dữ liệu tổng hợp để thử nghiệm các ý tưởng mới hoặc khi dữ liệu thực tế khan hiếm. |
| **Ràng buộc** | - **Chi phí (Cost):** Ngân sách cho phát triển, triển khai và bảo trì.<br>- **Bảo trì (Maintenance):** Khả năng dễ dàng cập nhật, sửa lỗi và duy trì hệ thống trong thời gian dài.<br>- **Tuân thủ quy định (Regulatory compliance):** Đặc biệt trong các ngành như y tế, tài chính, cần tuân thủ các quy định pháp lý nghiêm ngặt.<br>- **Tài nguyên hạn chế:** Có thể bị giới hạn bởi tài nguyên phần cứng hoặc đội ngũ kỹ sư. | - **Tài nguyên tính toán (Computational resources):** Các mô hình phức tạp thường đòi hỏi GPU mạnh mẽ và thời gian huấn luyện dài.<br>- **Thời gian (Time):** Các dự án nghiên cứu có thể kéo dài để khám phá và thử nghiệm các ý tưởng mới.<br>- **Khả năng tái tạo (Reproducibility):** Đảm bảo rằng kết quả có thể được tái tạo bởi các nhà nghiên cứu khác. |




## 3. So Sánh Theo Company

| Công ty | Phương pháp tiếp cận | Kiến trúc | Đổi mới chính | Kết quả | Liên kết |
|---------|----------------------|------------|---------------|---------|------|
| **Amazon** | **Greedy + K-beam Search** | **Top-down recursive (Đệ quy từ trên xuống)** | **Chiến lược kết hợp:** Amazon đã phát triển một chiến lược phân loại phân cấp kết hợp giữa thuật toán tham lam (Greedy) và tìm kiếm chùm K (K-beam Search). Thay vì chỉ chọn một đường dẫn duy nhất tại mỗi cấp độ (như trong Greedy Search thuần túy), K-beam Search giữ lại K đường dẫn tốt nhất, giúp giảm thiểu vấn đề lan truyền lỗi và tăng khả năng tìm ra đường dẫn phân loại chính xác hơn. | **Độ chính xác:** Trong nghiên cứu của Stanford, phương pháp này cho thấy độ chính xác giảm dần từ 55.32% ở cấp độ 1 (L1) xuống 11.79% ở cấp độ 8 (L8) khi sử dụng chiến lược tham lam. Tuy nhiên, chiến lược kết hợp đã cải thiện hiệu suất đáng kể ở các cấp độ sâu hơn. | [Stanford Study](https://cs229.stanford.edu/proj2014/Bin%20Wang,%20Shaoming%20Feng,%20Hierarchical%20Classification%20of%20Amazon%20Products.pdf) - Nghiên cứu này phân tích chi tiết các phương pháp phân loại phân cấp được Amazon sử dụng cho hàng triệu sản phẩm của họ. |
| **Amazon (2024)** | **Instance + Label Hierarchy (Phân cấp thực thể và nhãn)** | **Contrastive Learning (Học tương phản)** | **Lấy mẫu lân cận (Neighborhood sampling):** Phương pháp này tập trung vào việc học các biểu diễn mạnh mẽ bằng cách sử dụng học tương phản, trong đó các cặp thực thể và nhãn liên quan được kéo lại gần nhau trong không gian nhúng, trong khi các cặp không liên quan bị đẩy ra xa. Kỹ thuật lấy mẫu lân cận giúp chọn ra các cặp tích cực và tiêu cực hiệu quả, đặc biệt hữu ích trong việc xử lý cấu trúc phân cấp phức tạp. | **Vượt trội hơn các đường cơ sở:** Phương pháp này đã được chứng minh là vượt trội hơn các phương pháp phân loại phân cấp truyền thống và các đường cơ sở khác, cho thấy tiềm năng lớn trong việc cải thiện độ chính xác và khả năng khái quát hóa. | [arXiv 2024](https://arxiv.org/abs/2403.06021) - Bài báo mới nhất từ Amazon trình bày về cách tiếp cận tiên tiến này. |
| **Meta/Facebook** | **Few-Shot Learner (Học với ít mẫu)** | **Zero/Few/Low-shot (Học với không/ít/rất ít mẫu)** | **Thích ứng nhanh chóng (Rapid adaptation):** Meta đã phát triển các hệ thống AI có khả năng học và thích ứng nhanh chóng với các nhiệm vụ mới hoặc các loại nội dung mới chỉ với rất ít hoặc thậm chí không có dữ liệu huấn luyện. Điều này đặc biệt quan trọng trong việc phát hiện nội dung có hại hoặc phân loại các xu hướng mới nổi trên nền tảng của họ. | **Từ vài tuần xuống vài tháng:** Khả năng thích ứng nhanh giúp giảm thời gian cần thiết để triển khai các bộ phân loại mới từ vài tháng xuống chỉ còn vài tuần, mang lại lợi thế cạnh tranh đáng kể trong việc phản ứng với các mối đe dọa hoặc xu hướng mới. | [Meta Blog](https://about.fb.com/news/2021/12/metas-new-ai-system-tackles-harmful-content/) - Bài viết trên blog của Meta mô tả cách họ sử dụng AI để giải quyết vấn đề nội dung có hại. |
| **Microsoft** | **Orthogonal Transfer SVM** | **Hierarchical regularization (Điều chỉnh phân cấp)** | **Ràng buộc trực giao (Orthogonal constraints):** Microsoft đã phát triển một biến thể của SVM gọi là Orthogonal Transfer SVM, trong đó các ràng buộc trực giao được áp dụng để đảm bảo rằng các bộ phân loại ở các cấp độ khác nhau của hệ thống phân cấp học được các đặc trưng độc lập và bổ sung cho nhau. Điều này giúp cải thiện hiệu suất tổng thể và tính nhất quán của các dự đoán. | **State-of-the-art (SOTA) trên RCV1:** Phương pháp này đã đạt được hiệu suất SOTA trên bộ dữ liệu RCV1-v2, một bộ dữ liệu phân loại văn bản lớn và phức tạp, chứng tỏ hiệu quả của nó trong việc xử lý các bài toán phân loại phân cấp. | [MS Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ohsvm_techreport.pdf) - Nghiên cứu từ Microsoft cung cấp thông tin chi tiết về Orthogonal Transfer SVM. |




## 4. So Sánh Hiệu Suất

| Phương pháp | Bộ dữ liệu | Chỉ số | Hiệu suất | Năm | Nguồn |
|-------------|------------|--------|-----------|-----|-------|
| **Flat Classification** | **Amazon Products** | **Độ chính xác (Accuracy)** | **2.8-3.01%** | **2014** | **Stanford**<br>Phương pháp này được sử dụng làm đường cơ sở trong nghiên cứu của Stanford về phân loại sản phẩm Amazon. Kết quả cho thấy hiệu suất rất thấp khi bỏ qua hoàn toàn cấu trúc phân cấp, minh chứng cho sự cần thiết của các phương pháp phân loại phân cấp. |
| **Greedy Search** | **Amazon Products** | **Độ chính xác L1 (Accuracy L1)** | **55.32%** | **2014** | **Stanford**<br>Đây là một biến thể của phương pháp LCPN (Local Classifier Per Node) theo hướng top-down. Greedy Search chọn đường dẫn có xác suất cao nhất tại mỗi cấp độ. Độ chính xác 55.32% ở cấp độ 1 (L1) cho thấy hiệu quả ban đầu, nhưng hiệu suất giảm nhanh ở các cấp độ sâu hơn do vấn đề lan truyền lỗi. |
| **Combined Strategy (Amazon)** | **Amazon Products** | **Độ chính xác L1 (Accuracy L1)** | **Tốt nhất theo từng cấp độ** | **2014** | **Stanford**<br>Chiến lược kết hợp của Amazon (Greedy + K-beam Search) đã cải thiện đáng kể hiệu suất so với Greedy Search thuần túy. Bằng cách giữ lại nhiều đường dẫn tiềm năng (K-beam), nó giảm thiểu tác động của lỗi ở các cấp độ cao hơn và đạt được hiệu suất tốt nhất ở mỗi cấp độ của hệ thống phân cấp sản phẩm. |
| **Orthogonal Transfer SVM** | **RCV1-v2** | **Nhiều chỉ số (Multiple)** | **State-of-the-art (SOTA)** | **2011** | **Microsoft**<br>Phương pháp này của Microsoft đã đạt được hiệu suất SOTA trên bộ dữ liệu RCV1-v2, một bộ dữ liệu phân loại văn bản lớn với cấu trúc phân cấp phức tạp. Điều này cho thấy khả năng của HSVM trong việc xử lý các tác vụ phân loại tài liệu quy mô lớn với độ chính xác cao. |
| **CNN+BiLSTM+Attention** | **Wearable ECG** | **Độ chính xác (Accuracy)** | **95% nhị phân, 88% đa nhãn** | **2024** | **Healthcare**<br>Đây là một kiến trúc học sâu kết hợp Mạng nơ-ron tích chập (CNN) để trích xuất đặc trưng cục bộ, Mạng bộ nhớ dài ngắn hai chiều (BiLSTM) để nắm bắt phụ thuộc tuần tự, và cơ chế Attention để tập trung vào các phần quan trọng của dữ liệu. Ứng dụng trong phát hiện rối loạn nhịp tim từ dữ liệu ECG, đạt hiệu suất cao cho cả phân loại nhị phân (có/không có rối loạn nhịp) và đa nhãn (phân loại các loại rối loạn nhịp khác nhau). |
| **HSGT (Hierarchical Graph Transformer)** | **Large graphs (Đồ thị lớn)** | **Nhiều chỉ số (Multiple)** | **SOTA trên hàng triệu nút** | **2023** | **IJCAI**<br>HSGT là một ví dụ về phương pháp kết hợp GNN và Transformer, được thiết kế để xử lý các bài toán phân loại trên đồ thị quy mô lớn. Nó đã chứng minh khả năng đạt hiệu suất SOTA trên các đồ thị có hàng triệu nút, cho thấy tiềm năng trong các ứng dụng như mạng xã hội, hệ thống khuyến nghị và mạng tri thức. |




## 5. Ma trận khuyến nghị

| Kịch bản | Phương pháp tiếp cận được khuyến nghị | Lý do | Mức độ ưu tiên triển khai |
|----------|--------------------------------------|-------|--------------------------|
| **Khởi nghiệp Thương mại điện tử (E-commerce Startup)** | **LCPN → MTL (Local Classifier Per Node sau đó Multi-task Learning)** | - **Cân bằng giữa sự đơn giản và hiệu suất:** Bắt đầu với LCPN vì nó dễ triển khai và gỡ lỗi, phù hợp với nguồn lực hạn chế của một startup. Khi hệ thống phát triển và dữ liệu trở nên phong phú hơn, chuyển sang MTL để tận dụng các mối quan hệ giữa các nhiệm vụ và cải thiện hiệu suất tổng thể, giảm thiểu lan truyền lỗi.<br>- **Khả năng mở rộng:** LCPN cho phép mở rộng hệ thống phân loại một cách có kiểm soát, trong khi MTL giúp tối ưu hóa việc học từ dữ liệu đa dạng. | **Cao**<br>Vì đây là yếu tố cốt lõi để tổ chức sản phẩm và nâng cao trải nghiệm người dùng, ảnh hưởng trực tiếp đến doanh thu. |
| **Công ty Công nghệ lớn (Large Tech Company)** | **Các phương pháp dựa trên đồ thị (Graph-based methods)** | - **Tài nguyên cho sự phức tạp:** Các công ty lớn có đủ tài nguyên tính toán và đội ngũ kỹ sư chuyên môn để triển khai và duy trì các mô hình phức tạp như GNN+Transformer.<br>- **Xử lý dữ liệu quy mô lớn:** Các phương pháp dựa trên đồ thị đặc biệt hiệu quả với dữ liệu có cấu trúc phức tạp và quy mô lớn, điển hình là các mạng xã hội, hệ thống khuyến nghị, hoặc cơ sở tri thức.<br>- **Đạt hiệu suất SOTA:** Các phương pháp này thường mang lại hiệu suất tốt nhất, phù hợp với mục tiêu dẫn đầu công nghệ của các công ty lớn. | **Trung bình**<br>Mặc dù quan trọng, nhưng việc triển khai có thể mất thời gian và đòi hỏi đầu tư lớn, nên có thể không phải là ưu tiên hàng đầu ngay lập tức nếu các giải pháp hiện có vẫn đáp ứng được. |
| **Chăm sóc sức khỏe (Healthcare)** | **Multi-task Learning (Học đa nhiệm)** | - **Yêu cầu quy định (Regulatory requirements):** Trong lĩnh vực y tế, việc chẩn đoán và phân loại thường liên quan đến nhiều yếu tố và cần độ chính xác cao. MTL cho phép mô hình học các mối quan hệ phức tạp giữa các triệu chứng, bệnh lý và kết quả điều trị, đồng thời cải thiện khả năng khái quát hóa.<br>- **Tận dụng dữ liệu đa dạng:** Dữ liệu y tế thường rất đa dạng (hình ảnh, văn bản, tín hiệu sinh học), MTL có thể tích hợp thông tin từ nhiều nguồn để đưa ra quyết định chính xác hơn.<br>- **Giảm thiểu lỗi:** Khả năng giảm lan truyền lỗi của MTL là rất quan trọng trong các ứng dụng y tế, nơi một lỗi nhỏ có thể có hậu quả nghiêm trọng. | **Cao**<br>Độ chính xác và độ tin cậy là tối quan trọng trong y tế, và MTL cung cấp một khung làm việc mạnh mẽ để đạt được điều đó. |
| **Dự án Nghiên cứu (Research Project)** | **GNN + Transformer** | - **Thúc đẩy trạng thái nghệ thuật (Push state-of-the-art):** Các dự án nghiên cứu thường tập trung vào việc khám phá các giới hạn của công nghệ và phát triển các phương pháp mới. GNN và Transformer là những kiến trúc tiên tiến nhất hiện nay, mang lại tiềm năng lớn để đạt được hiệu suất vượt trội.<br>- **Khám phá các ý tưởng mới:** Đây là nền tảng tốt để thử nghiệm các ý tưởng mới về biểu diễn đồ thị, cơ chế chú ý (attention) và học sâu. | **Thấp**<br>Mặc dù quan trọng về mặt học thuật, nhưng mức độ ưu tiên triển khai trong thực tế có thể thấp hơn do độ phức tạp và chi phí cao. |
| **Nguyên mẫu nhanh (Quick Prototype)** | **Flat Classification + Hậu xử lý (post-processing)** | - **Triển khai nhanh chóng (Fast implementation):** Phương pháp Flat Classification rất đơn giản và nhanh chóng để triển khai, cho phép tạo ra một nguyên mẫu hoạt động trong thời gian ngắn.<br>- **Kiểm tra ý tưởng:** Phù hợp để nhanh chóng kiểm tra tính khả thi của một ý tưởng hoặc thu thập phản hồi ban đầu mà không cần đầu tư quá nhiều vào kiến trúc phức tạp.<br>- **Dễ dàng điều chỉnh:** Hậu xử lý có thể được sử dụng để điều chỉnh kết quả và đảm bảo tính nhất quán phân cấp ở mức độ cơ bản. | **Cao**<br>Khi cần một giải pháp nhanh chóng để chứng minh khái niệm hoặc thử nghiệm thị trường. |




## 6. Tóm tắt các Thực hành Tốt nhất

### 6.1. Thực hành Tốt nhất trong Công nghiệp (Industry Best Practices)

1.  **Bắt đầu đơn giản (Start Simple):**
    *   **Mô tả:** Luôn bắt đầu với một đường cơ sở (baseline) đơn giản như Flat Classification. Sau đó, dần dần nâng cấp lên các phương pháp phức tạp hơn như LCPN (Local Classifier Per Node) khi nhu cầu và dữ liệu phát triển.
    *   **Lý do:** Giúp nhanh chóng có được một hệ thống hoạt động, dễ dàng xác định các vấn đề và đo lường mức độ cải thiện của các phương pháp phức tạp hơn.

2.  **Ngưỡng tin cậy (Confidence Thresholding):**
    *   **Mô tả:** Áp dụng các ngưỡng tin cậy cho các dự đoán của mô hình. Nếu độ tin cậy của một dự đoán thấp, có thể chuyển giao cho con người xem xét hoặc sử dụng các phương pháp dự phòng.
    *   **Lý do:** Giảm thiểu sự lan truyền lỗi trong các hệ thống phân cấp từ trên xuống và đảm bảo chất lượng đầu ra, đặc biệt quan trọng trong các ứng dụng nhạy cảm.

3.  **Học chuyển giao (Transfer Learning):**
    *   **Mô tả:** Tận dụng các mô hình đã được huấn luyện trước (pretrained models) trên các tập dữ liệu lớn. Sau đó, tinh chỉnh (fine-tune) chúng trên dữ liệu cụ thể của bạn.
    *   **Lý do:** Giảm đáng kể thời gian và tài nguyên huấn luyện, đồng thời cải thiện hiệu suất, đặc biệt khi dữ liệu của bạn hạn chế.

4.  **Các chỉ số kinh doanh (Business Metrics):**
    *   **Mô tả:** Tập trung vào các chỉ số đánh giá có ý nghĩa kinh doanh trực tiếp (ví dụ: tỷ lệ chuyển đổi, giảm chi phí vận hành, cải thiện trải nghiệm khách hàng) thay vì chỉ các chỉ số kỹ thuật thuần túy.
    *   **Lý do:** Đảm bảo rằng các cải tiến về mặt kỹ thuật thực sự mang lại giá trị cho doanh nghiệp.

5.  **Kiểm thử A/B (A/B Testing):**
    *   **Mô tả:** Triển khai các phương pháp mới song song với phương pháp cũ và so sánh hiệu suất của chúng trong môi trường thực tế với một nhóm người dùng nhỏ.
    *   **Lý do:** Xác thực các cải tiến một cách khách quan và giảm thiểu rủi ro khi triển khai rộng rãi.

### 6.2. Thực hành Tốt nhất trong Nghiên cứu (Research Best Practices)

1.  **Thiết kế nhận biết phân cấp (Hierarchy-aware Design):**
    *   **Mô tả:** Thiết kế các mô hình và thuật toán có khả năng mã hóa và tận dụng cấu trúc phân loại một cách tự nhiên.
    *   **Lý do:** Đảm bảo rằng thông tin phân cấp được sử dụng hiệu quả để cải thiện độ chính xác và tính nhất quán của các dự đoán.

2.  **Học đa nhiệm (Multi-task Learning):**
    *   **Mô tả:** Huấn luyện mô hình để giải quyết nhiều nhiệm vụ liên quan đồng thời, trong đó các nhiệm vụ có thể tương ứng với các cấp độ khác nhau của hệ thống phân cấp.
    *   **Lý do:** Tối ưu hóa chung giúp mô hình học các biểu diễn mạnh mẽ hơn, giảm lan truyền lỗi và cải thiện hiệu suất tổng thể.

3.  **Phương pháp đồ thị (Graph Methods):**
    *   **Mô tả:** Sử dụng các phương pháp dựa trên đồ thị như GNNs để học biểu diễn từ dữ liệu có cấu trúc đồ thị, đặc biệt khi các mối quan hệ giữa các lớp hoặc thực thể có thể được mô hình hóa dưới dạng đồ thị.
    *   **Lý do:** Cung cấp khả năng học biểu diễn phong phú và mạnh mẽ cho các cấu trúc dữ liệu phức tạp, thường vượt trội so với các phương pháp truyền thống.

4.  **Đánh giá toàn diện (Comprehensive Evaluation):**
    *   **Mô tả:** Sử dụng các chỉ số đánh giá chuyên biệt cho phân loại phân cấp (ví dụ: Hierarchical F1, Tree distance) bên cạnh các chỉ số truyền thống.
    *   **Lý do:** Cung cấp cái nhìn đầy đủ và chính xác hơn về hiệu suất của mô hình trong bối cảnh phân cấp.

5.  **Phân tích lý thuyết (Theoretical Analysis):**
    *   **Mô tả:** Nghiên cứu và hiểu rõ các thuộc tính toán học, giới hạn và đảm bảo của các thuật toán được đề xuất.
    *   **Lý do:** Cung cấp cơ sở khoa học vững chắc cho các phương pháp, giúp giải thích hành vi của mô hình và hướng dẫn phát triển trong tương lai.

### 6.3. Các Nguyên tắc Chung (Universal Principles)

1.  **Chất lượng dữ liệu (Data Quality):**
    *   **Mô tả:** Đảm bảo dữ liệu huấn luyện sạch, được gắn nhãn chính xác và có cấu trúc phân loại rõ ràng, nhất quán.
    *   **Lý do:** Dữ liệu chất lượng cao là nền tảng cho mọi mô hình học máy hiệu quả. Dữ liệu kém chất lượng sẽ dẫn đến hiệu suất kém, bất kể mô hình có phức tạp đến đâu.

2.  **Chiến lược đánh giá (Evaluation Strategy):**
    *   **Mô tả:** Thiết lập một chiến lược đánh giá mạnh mẽ, bao gồm nhiều chỉ số, kiểm thử trên các tập dữ liệu độc lập và kiểm thử trong môi trường thực tế.
    *   **Lý do:** Đảm bảo rằng mô hình không chỉ hoạt động tốt trên dữ liệu huấn luyện mà còn có khả năng khái quát hóa tốt trong các tình huống thực tế.

3.  **Khả năng mở rộng (Scalability):**
    *   **Mô tả:** Thiết kế các giải pháp có khả năng xử lý lượng dữ liệu và số lượng lớp tăng lên mà không làm giảm đáng kể hiệu suất hoặc tăng chi phí quá mức.
    *   **Lý do:** Các hệ thống phân loại phân cấp thường phải đối mặt với dữ liệu lớn và cấu trúc phức tạp, do đó khả năng mở rộng là yếu tố then chốt cho sự thành công lâu dài.

4.  **Khả năng giải thích (Interpretability):**
    *   **Mô tả:** Cố gắng hiểu cách mô hình đưa ra quyết định, đặc biệt là trong các ứng dụng quan trọng như y tế hoặc tài chính.
    *   **Lý do:** Giúp xây dựng niềm tin vào mô hình, gỡ lỗi khi có lỗi và tuân thủ các quy định yêu cầu tính minh bạch.

5.  **Học liên tục (Continuous Learning):**
    *   **Mô tả:** Thiết lập cơ chế để mô hình có thể học và thích nghi với các danh mục mới, thay đổi trong cấu trúc phân loại hoặc dữ liệu mới theo thời gian.
    *   **Lý do:** Các hệ thống phân loại thường phải đối mặt với sự thay đổi liên tục của dữ liệu và yêu cầu, do đó khả năng học liên tục là cần thiết để duy trì hiệu suất và tính phù hợp. |



