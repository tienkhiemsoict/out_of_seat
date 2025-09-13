import cv2
import numpy as np
import torch
import os
import time
from collections import defaultdict

from utils.process_coords import *

from ultralytics import YOLO

from argparse import ArgumentParser


IOU_THRESHOLD = 0.1 

class Exam_Monitoring:
    def __init__(self, 
                 image, 
                 inp_video, 
                 out_video, 
                 yolo_model_path, 
                 device='cpu', 
                 display=False, 
                 webcam=False, 
                 front_cam=False, 
                 back_cam=False):
        self.image = image
        self.inp_video = inp_video
        self.out_video = out_video
        self.device = device
        self.display = display
        self.webcam = webcam
        self.front_cam = front_cam
        self.back_cam = back_cam

        self.yolo_model = YOLO(yolo_model_path).to(self.device)

        # ByteTrack tracking cho phát hiện rời khỏi chỗ ngồi
        self.person_original_seat = {}     # track_id -> seat_id ban đầu
        self.people_out_of_seat = {}       # track_id -> bbox cho những người rời khỏi chỗ
        self.log_behaviors = defaultdict(list)   # Logs cho rời khỏi chỗ ngồi
        
        
        
    def __call__(self):
        if self.webcam:
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = self._process(frame)

                if self.display:
                    cv2.imshow('Result', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()

        elif self.inp_video:
            cap = cv2.VideoCapture(self.inp_video)
            w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                    cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
            writer = cv2.VideoWriter(self.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            cnt = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cnt += 1
                if cnt % 2 == 0:
                    continue

                frame = self._process(frame)
                writer.write(frame)

                if self.display:
                    cv2.imshow('Result', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            cap.release()
            writer.release()
            if self.display:
                cv2.destroyAllWindows() 
            print(f'Output saved to {self.out_video}')
        
        elif self.image:
            frame = cv2.imread(self.image)
            frame = self._process(frame)
            
            if self.display:
                cv2.imshow('Result', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def _process(self, frame):
        """Core processing function - chỉ xử lý phát hiện rời khỏi chỗ ngồi"""
        self.draw_cell_computer(frame)
        frame = self.detect_out_of_seat(frame)
        return frame
                

    def detect_out_of_seat(self, frame):
        """
        Phát hiện người rời khỏi chỗ ngồi sử dụng ByteTrack
        """
        # Sử dụng YOLO với ByteTrack
        results = self.yolo_model.track(frame, classes=[0], device=self.device, conf=0.3, iou=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        if not results or not hasattr(results[0], 'boxes') or results[0].boxes is None:
            return frame
            
        boxes = results[0].boxes
        if not hasattr(boxes, 'id') or boxes.id is None:
            return frame
            
        # Lấy dữ liệu tracking
        bboxes = boxes.xyxy.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)
        
        # Reset people out of seat cho frame này
        current_out_of_seat = {}
        
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            if confidences[i] < 0.3:
                continue
                
            track_id = track_ids[i]
            person_box = (x1, y1, x2, y2)
            person_assigned = False
            assigned_seat = None
            
            # Kiểm tra gán ghế
            if self.front_cam:
                for seat_id, seat_coords in FRONT_ROIS.items():
                    if calculate_iou_rotated(seat_coords, person_box) > IOU_THRESHOLD:
                        person_assigned = True
                        assigned_seat = seat_id
                        break
                        
            if self.back_cam and not person_assigned:
                for seat_id, seat_coords in BACK_ROIS.items():
                    if calculate_iou_rotated(seat_coords, person_box) > IOU_THRESHOLD:
                        person_assigned = True
                        assigned_seat = seat_id
                        break
            
            # Gán ghế gốc cho track_id mới (frame đầu tiên xuất hiện)
            if track_id not in self.person_original_seat and person_assigned:
                self.person_original_seat[track_id] = assigned_seat
            
            # Kiểm tra rời khỏi chỗ ngồi gốc
            is_out_of_seat = False
            if track_id in self.person_original_seat:
                original_seat = self.person_original_seat[track_id]
                if not person_assigned or assigned_seat != original_seat:
                    is_out_of_seat = True
                    current_out_of_seat[track_id] = person_box
                    
                    # Log behavior cho người rời khỏi chỗ
                    out_of_seat_id = f"OUT_OF_SEAT_ID_{track_id}"
                    if (len(self.log_behaviors[out_of_seat_id]) == 0 or 
                        self.log_behaviors[out_of_seat_id][-1] != "ROI_KHOI_CHO"):
                        self.log_behaviors[out_of_seat_id].append("ROI_KHOI_CHO")
            elif not person_assigned:
                # Người mới xuất hiện không có ghế
                is_out_of_seat = True
                current_out_of_seat[track_id] = person_box
                
                out_of_seat_id = f"OUT_OF_SEAT_ID_{track_id}"
                if (len(self.log_behaviors[out_of_seat_id]) == 0 or 
                    self.log_behaviors[out_of_seat_id][-1] != "ROI_KHOI_CHO"):
                    self.log_behaviors[out_of_seat_id].append("ROI_KHOI_CHO")
            
            # Vẽ bbox
            if is_out_of_seat:
                # Bbox đỏ cho người rời khỏi chỗ
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, 'ROI KHOI CHO', (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f'ID:{track_id}', (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                # Bbox bình thường cho người trong ghế
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 150), 1)
                cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Cập nhật people_out_of_seat
        self.people_out_of_seat = current_out_of_seat
        
        return frame
    
    def draw_cell_computer(self, frame):
        """Vẽ các khu vực ghế ngồi"""
        if self.front_cam: 
            for seat_id, seat_coords in FRONT_ROIS.items():
                (x1, y1, x2, y2, x3, y3) = seat_coords
                x4, y4 = x1 + x3 - x2, y1 + y3 - y2
                pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(255,0,0), thickness=1)
                cv2.putText(frame, f'{seat_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        if self.back_cam:
            for seat_id, seat_coords in BACK_ROIS.items():
                (x1, y1, x2, y2, x3, y3) = seat_coords
                x4, y4 = x1 + x3 - x2, y1 + y3 - y2
                pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(255,0,0), thickness=1)
                cv2.putText(frame, f'{seat_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


inp_video = './demo/frontal_test2_13_9.mp4'


parser = ArgumentParser()
parser.add_argument('--image', type=str, default=None)
parser.add_argument('--inp_video', type=str, default=inp_video)
parser.add_argument('--out_video', type=str, default='./demo/vid_out_out_of_seat.mp4')
parser.add_argument('--yolo_model_path', type=str, default='./model/yolov8n.pt')

parser.add_argument('--device', type=str, default='cpu')    # cuda for gpu 
parser.add_argument('--display', action='store_true', default=True)
parser.add_argument('--webcam', action='store_true', default=False)
parser.add_argument('--front_cam', action='store_true', default=True)
parser.add_argument('--back_cam', action='store_true', default=False)

args = parser.parse_args()

t1 = time.time()
exam = Exam_Monitoring(
                    image=args.image, 
                    inp_video=args.inp_video, 
                    out_video=args.out_video, 
                    yolo_model_path=args.yolo_model_path, 
                    device=args.device, 
                    display=args.display, 
                    webcam=args.webcam, 
                    front_cam=args.front_cam, 
                    back_cam=args.back_cam
                    )
exam()
print("Running time:", time.time() - t1)
print("Out of seat behaviors:", exam.log_behaviors)

# python Exam_Monitoring_OutOfSeat.py --inp_video='./demo/frontal_test2_13_9.mp4' --front_cam --display --device=cpu