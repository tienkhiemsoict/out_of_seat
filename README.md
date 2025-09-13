Video Input → Frame Processing → YOLO+ByteTrack → Seat Assignment → Out-of-Seat Detection → Visual Warning + Logging

1. YOLO Detection + ByteTrack
- yolo_model.track() với tracker="bytetrack.yaml"
- Persistent tracking
- Output: bboxes + confidences + track_ids

2. Seat Assignment
- Tính IoU giữa bbox và seat regions
- Nếu IoU > IOU_THRESHOLD → person được gán vào seat
- Lưu assigned_seat cho mỗi track_id

3. Original Seat Tracking
- Lần đầu tiên track_id xuất hiện + có seat → Lưu original_seat
- person_original_seat[track_id] = assigned_seat (không thay đổi)

4. So sánh assigned_seat hiện tại với original_seat
- Case 1: assigned_seat != original_seat → Out of seat
- Case 2: không có assigned_seat nhưng có original_seat → Out of seat  
- Case 3: track_id mới không có seat nào → Out of seat

5. Visual Warning + Logging
