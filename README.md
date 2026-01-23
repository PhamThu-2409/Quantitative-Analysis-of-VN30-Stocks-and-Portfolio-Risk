# Phân tích định lượng cổ phiếu VN30 và rủi ro danh mục đầu tư

## Mô tả (Description)
Dự án này thực hiện phân tích định lượng toàn diện đối với các cổ phiếu thuộc chỉ số VN30 trong giai đoạn 2020–2025, tập trung vào phân tích lợi suất, rủi ro hệ thống và xây dựng danh mục đầu tư.
Nghiên cứu kết hợp các phương pháp kinh tế lượng, lý thuyết định giá tài sản và phân tích danh mục nhằm cung cấp góc nhìn thực nghiệm về thị trường chứng khoán Việt Nam.

---

## Tính năng chính (Features)
- Tiền xử lý và chuẩn hóa dữ liệu giá cổ phiếu VN30
- Phân tích lợi suất và rủi ro
- Ước lượng hệ số beta theo mô hình CAPM
- Dự báo lợi suất và giá cổ phiếu bằng mô hình ARIMA
- Xây dựng và đánh giá danh mục đầu tư cổ phiếu

---

## Minh họa (Visuals)
Một số kết quả và hình ảnh minh họa được lưu trong thư mục `output/`, bao gồm:
- Biểu đồ xu hướng giá chuẩn hóa VN30
- Heatmap tương quan giữa các cổ phiếu
- Biểu đồ lợi suất tích lũy của danh mục
- Kết quả dự báo ARIMA
...
  
---

## Cài đặt (Installation)
### Yêu cầu
- Python 3.9 trở lên
- Hệ điều hành: Windows / macOS / Linux

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

---

## Cách sử dụng (Usage)
Chạy toàn bộ quy trình bằng lệnh:
```bash
python data.py
python analysis.py
```

Sau khi chạy, các bảng kết quả và biểu đồ sẽ được xuất ra thư mục `output/`.

---

## Dữ liệu (Data)
Dữ liệu sử dụng là dữ liệu ngày giai đoạn 2020–2025, bao gồm:
- Giá cổ phiếu các doanh nghiệp thuộc VN30
- Dữ liệu chỉ số VNINDEX
- Lãi suất phi rủi ro (đại diện bằng trái phiếu Chính phủ Việt Nam)

Tất cả dữ liệu được lấy trực tiếp thông qua API CafeF, có lưu backup trong thư mục `data/` dưới định dạng CSV.
Riêng lãi suất phi rủi ro được lấy từ Investing.com, cũng được lưu trong thư mục `data/` dưới định dạng CSV.

---

## Cấu trúc dự án (Project Structure)
```
Quantitative Analysis of VN30 Stocks and Portfolio Risk/
│
├── analysis.py                  # Code phân tích và mô hình hóa
├── data.py           # Thu thập và tiền xử lý dữ liệu
├── data/                         # Dữ liệu thô và dữ liệu đã xử lý
├── output/                       # Biểu đồ, bảng kết quả, file xuất
├── report.pdf                       # Báo cáo dự án (PDF)
└── README.md
```

---

## Báo cáo đầy đủ (Report)
Bản báo cáo hoàn chỉnh có thể xem tại đây:

📄 https://github.com/PhamThu-2409/Quantitative-Analysis-of-VN30-Stocks-and-Portfolio-Risk/blob/main/report.pdf

---

## Hỗ trợ (Support)
Nếu có câu hỏi hoặc góp ý liên quan đến dự án, vui lòng mở **Issue** trên GitHub repository.

---

## Định hướng phát triển (Roadmap)
- Cải thiện mô hình dự báo (GARCH, LSTM)
- Tối ưu danh mục theo Markowitz
- Mở rộng sang các chỉ số khác ngoài VN30

---

## Đóng góp (Contributing)
Dự án hiện được xây dựng cho mục đích học tập và trải nghiệm cá nhân. Mọi góp ý hoặc đề xuất cải thiện đều được hoan nghênh thông qua GitHub Issues.

---

## Tác giả (Author)
**Phạm Thị Anh Thư**  
Sinh viên ngành Công nghệ Tài chính  
Quan tâm: Tài chính định lượng, Quản trị rủi ro, Phân tích dữ liệu

---

## Trạng thái dự án (Project Status)
Dự án đã hoàn thành phiên bản cơ bản và có thể tiếp tục được mở rộng trong tương lai.

---

## Lưu ý
Dự án được thực hiện cho mục đích học tập và trải nghiệm cá nhân, không nhằm mục đích tư vấn đầu tư.

