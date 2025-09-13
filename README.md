BÁO CÁO: HỆ THỐNG PHÁT HIỆN RỜI KHỎI CHỖ NGỒI SỬ DỤNG BYTETRACK

1. MÔ TẢ HỆ THỐNG

Hệ thống chuyên biệt phát hiện sinh viên rời khỏi vị trí được gán ban đầu trong phòng thi.
Sử dụng ByteTrack tracking để duy trì ID người qua các frame và phát hiện vi phạm di chuyển.

2. PIPELINE HOẠT ĐỘNG

2.1 Luồng xử lý chính:
Video Input → Frame Processing → YOLO+ByteTrack → Seat Assignment → Out-of-Seat Detection → Visual Warning + Logging

2.2 Chi tiết từng bước:

Bước 1: Frame Processing
- Đọc video frame by frame
- Skip mỗi frame thứ 2 để tăng tốc xử lý
- Vẽ seat regions trên frame

Bước 2: YOLO Detection + ByteTrack
- yolo_model.track() với tracker="bytetrack.yaml"
- Persistent tracking qua persist=True
- Output: bboxes + confidences + track_ids

Bước 3: Seat Assignment
- Tính IoU giữa bbox và seat regions (FRONT_ROIS/BACK_ROIS)
- Nếu IoU > IOU_THRESHOLD → person được gán vào seat
- Lưu assigned_seat cho mỗi track_id

Bước 4: Original Seat Tracking
- Lần đầu tiên track_id xuất hiện + có seat → Lưu original_seat
- person_original_seat[track_id] = assigned_seat (không thay đổi)

Bước 5: Out-of-Seat Detection
- So sánh assigned_seat hiện tại với original_seat
- Case 1: assigned_seat != original_seat → Out of seat
- Case 2: không có assigned_seat nhưng có original_seat → Out of seat  
- Case 3: track_id mới không có seat nào → Out of seat

Bước 6: Visual Warning + Logging
- Out of seat: Bbox đỏ (0,0,255) + "ROI KHOI CHO" + "ID:X"
- In seat: Bbox xanh (0,150,150) + "ID:X"
- Log: "OUT_OF_SEAT_ID_X" → ["ROI_KHOI_CHO"]

3. CẤU TRÚC CODE CHÍNH

3.1 Class Exam_Monitoring:
```python
class Exam_Monitoring:
    def __init__(self):
        # ByteTrack tracking cho phát hiện rời khỏi chỗ ngồi
        self.person_original_seat = {}     # track_id -> seat_id ban đầu
        self.people_out_of_seat = {}       # track_id -> bbox cho những người rời khỏi chỗ
        self.log_behaviors = defaultdict(list)   # Logs cho rời khỏi chỗ ngồi
```

3.2 Hàm xử lý chính:
```python
def _process(self, frame):
    """Core processing function - chỉ xử lý phát hiện rời khỏi chỗ ngồi"""
    self.draw_cell_computer(frame)
    frame = self.detect_out_of_seat(frame)
    return frame
```

3.3 Hàm phát hiện rời khỏi chỗ:
```python
def detect_out_of_seat(self, frame):
    # Sử dụng YOLO với ByteTrack
    results = self.yolo_model.track(frame, classes=[0], device=self.device, 
                                  conf=0.3, iou=0.5, persist=True, 
                                  tracker="bytetrack.yaml", verbose=False)
    
    # Lấy dữ liệu tracking
    bboxes = boxes.xyxy.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int)
    
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        track_id = track_ids[i]
        person_box = (x1, y1, x2, y2)
        
        # Kiểm tra gán ghế hiện tại
        person_assigned = False
        assigned_seat = None
        
        # Kiểm tra IoU với các seat regions
        if self.front_cam:
            for seat_id, seat_coords in FRONT_ROIS.items():
                if calculate_iou_rotated(seat_coords, person_box) > IOU_THRESHOLD:
                    person_assigned = True
                    assigned_seat = seat_id
                    break
        
        # Gán ghế gốc cho track_id mới
        if track_id not in self.person_original_seat and person_assigned:
            self.person_original_seat[track_id] = assigned_seat
        
        # Kiểm tra rời khỏi chỗ ngồi gốc
        is_out_of_seat = False
        if track_id in self.person_original_seat:
            original_seat = self.person_original_seat[track_id]
            if not person_assigned or assigned_seat != original_seat:
                is_out_of_seat = True
                # Log behavior
                out_of_seat_id = f"OUT_OF_SEAT_ID_{track_id}"
                self.log_behaviors[out_of_seat_id].append("ROI_KHOI_CHO")
        
        # Vẽ bbox tương ứng
        if is_out_of_seat:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, 'ROI KHOI CHO', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 150), 1)
            cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
```

4. DEPENDENCIES VÀ YÊU CẦU

4.1 Thư viện cần thiết:
- opencv-python
- ultralytics (với ByteTrack support)
- torch
- numpy

4.2 Files cần thiết:
- utils/process_coords.py (chứa FRONT_ROIS, BACK_ROIS, calculate_iou_rotated)
- model/yolov8n.pt (YOLO model)
- bytetrack.yaml (ByteTracker config)

4.3 Input/Output:
- Input: Video file hoặc webcam
- Output: Processed video với cảnh báo + Log behaviors

5. CÁCH SỬ DỤNG

5.1 Chạy từ video file:
```bash
python Exam_Monitoring_OutOfSeat.py --inp_video='./demo/frontal_test2_13_9.mp4' --front_cam --display --device=cpu
```

5.2 Chạy từ webcam:
```bash
python Exam_Monitoring_OutOfSeat.py --webcam --front_cam --display --device=cpu
```

5.3 Chạy từ image:
```bash
python Exam_Monitoring_OutOfSeat.py --image='./demo/test.jpg' --front_cam --display --device=cpu
```

5.4 Tham số quan trọng:
- --front_cam / --back_cam: Chọn camera trước/sau
- --display: Hiển thị real-time
- --device: cpu hoặc cuda
- IOU_THRESHOLD = 0.1: Ngưỡng IoU cho seat assignment

6. KẾT QUẢ OUTPUT

6.1 Visual output:
- Bbox đỏ dày + "ROI KHOI CHO": Người rời khỏi chỗ
- Bbox xanh mỏng + "ID:X": Người trong ghế gốc
- Seat regions được vẽ bằng polylines đỏ

6.2 Log output:
```python
{
    'OUT_OF_SEAT_ID_1': ['ROI_KHOI_CHO'],
    'OUT_OF_SEAT_ID_3': ['ROI_KHOI_CHO'],
    'OUT_OF_SEAT_ID_5': ['ROI_KHOI_CHO']
}
```

6.3 Video output:
- Lưu tại path chỉ định trong --out_video
- Format: MP4V codec
- FPS giữ nguyên như input

7. THUẬT TOÁN BYTETRACK

7.1 Tracking pipeline:
- High-confidence detections → Kalman filter tracking
- Low-confidence detections → Associate với lost tracks
- Track management: New, Tracked, Lost, Removed states

7.2 ID persistence:
- Track ID được duy trì khi người tạm thời biến mất
- Re-identification khi người xuất hiện lại
- Robust với occlusion và motion blur

7.3 Performance:
- Real-time tracking trên CPU
- Memory efficient cho long videos
- Stable ID assignment
