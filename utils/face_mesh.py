import math
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from utils.process_coords import CAMERA_ANGLE_CORRECTION
           
BASE_YAW_THRESHOLD = -30            
CONFIDENCE_FRAMES = 2              
MIN_CONFIDENCE_SCORE = 0.3


def analyze_face_orientation(face_landmarks):
    """Phân tích hướng khuôn mặt bằng nhiều phương pháp"""
    try:
        # Phương pháp 1: So sánh khoảng cách mắt trái-phải với mũi
        left_eye = face_landmarks.landmark[33]   # Left eye inner corner
        right_eye = face_landmarks.landmark[263] # Right eye inner corner
        nose_tip = face_landmarks.landmark[1]    # Nose tip
        nose_bridge = face_landmarks.landmark[168] # Nose bridge
        
        eye_center_x = (left_eye.x + right_eye.x) / 2
        nose_x = nose_tip.x
        
        eye_width = abs(right_eye.x - left_eye.x)
        if eye_width > 0:
            nose_offset_normalized = (nose_x - eye_center_x) / eye_width
        else:
            nose_offset_normalized = 0
        
        # Phương pháp 2: So sánh độ rộng mắt trái và phải
        left_eye_width = abs(face_landmarks.landmark[33].x - face_landmarks.landmark[133].x)
        right_eye_width = abs(face_landmarks.landmark[362].x - face_landmarks.landmark[263].x)
        
        if left_eye_width + right_eye_width > 0:
            eye_width_ratio = (right_eye_width - left_eye_width) / (left_eye_width + right_eye_width)
        else:
            eye_width_ratio = 0
        
        # Phương pháp 3: Góc nghiêng của đường mắt
        eye_angle = math.degrees(math.atan2(
            right_eye.y - left_eye.y,
            right_eye.x - left_eye.x
        ))
        
        return {
            'nose_offset': nose_offset_normalized,
            'eye_ratio': eye_width_ratio,
            'eye_angle': eye_angle,
            'confidence': min(left_eye_width + right_eye_width, 1.0)
        }
    except:
        return None
    

def determine_turn_direction(yaw, pitch, roll, face_orientation, seat_id):
    """Xác định hướng quay với độ chính xác cao theo từng vị trí"""
    threshold, sensitivity, yaw_offset = get_position_specific_threshold(seat_id) # Cấu hình cho từng vị trí cụ thể
    # Hiệu chỉnh yaw theo vị trí camera
    corrected_yaw = yaw - yaw_offset if yaw is not None else None
    direction = "normal"
    confidence_score = 0.0
    
    if corrected_yaw is None:
        return direction, confidence_score
    effective_yaw = corrected_yaw * sensitivity
    
    # Xác định hướng quay cơ bản
    if effective_yaw > threshold:
        direction = "quay_trai"
        confidence_score = min(abs(effective_yaw) / 45.0, 1.0)
    elif effective_yaw < -threshold:
        direction = "quay_phai" 
        confidence_score = min(abs(effective_yaw) / 45.0, 1.0)
    
    # Xác thực bằng face orientation
    if face_orientation and direction != "normal":
        nose_offset = face_orientation['nose_offset']
        eye_ratio = face_orientation['eye_ratio']
        face_confidence = face_orientation['confidence']
        
        if direction == "quay_trai": # Khi quay trái, mũi nên lệch trái, mắt phải nên nhỏ hơn
            if nose_offset > 0.1 and eye_ratio < -0.1:
                confidence_score += 0.2 * face_confidence
            elif nose_offset < -0.05 or eye_ratio > 0.05:
                confidence_score -= 0.3  # Penalty cho không nhất quán
        elif direction == "quay_phai": # Khi quay phải, mũi nên lệch phải, mắt trái nên nhỏ hơn
            if nose_offset < -0.1 and eye_ratio > 0.1:
                confidence_score += 0.2 * face_confidence
            elif nose_offset > 0.05 or eye_ratio < -0.05:
                confidence_score -= 0.15  # Penalty cho không nhất quán
    
    # Áp dụng ngưỡng confidence
    if confidence_score < MIN_CONFIDENCE_SCORE:
        direction = "normal"
        confidence_score = 0.0
    return direction, confidence_score

def get_head_pose(image_shape, face_landmarks):
    """Phân tích tư thế đầu với độ ổn định cao"""
    img_h, img_w, _ = image_shape
    
    if img_h < 50 or img_w < 50:  # Khuôn mặt quá nhỏ
        return None, None, None
    
    # Sử dụng nhiều điểm mốc hơn để tăng độ chính xác
    try:
        # 6 điểm chính + 4 điểm phụ để tăng độ ổn định
        face_3d_model_points = np.array([
            [0.0, 0.0, 0.0],           # Nose tip
            [0.0, -330.0, -65.0],     # Chin
            [-225.0, 170.0, -135.0],  # Left eye left corner
            [225.0, 170.0, -135.0],   # Right eye right corner
            [-150.0, -150.0, -125.0], # Left mouth corner
            [150.0, -150.0, -125.0],  # Right mouth corner
            [-75.0, 0.0, -100.0],     # Left nose wing
            [75.0, 0.0, -100.0],      # Right nose wing
        ], dtype=np.float64)
        
        landmarks_2d = []
        landmark_indices = [1, 152, 263, 33, 287, 57, 235, 454]  # Thêm 2 điểm cánh mũi
        
        for idx in landmark_indices:
            x = face_landmarks.landmark[idx].x * img_w
            y = face_landmarks.landmark[idx].y * img_h
            if 0 <= x <= img_w and 0 <= y <= img_h: # Kiểm tra điểm có nằm trong ảnh không
                landmarks_2d.append([x, y])
            else:
                return None, None, None
                
        if len(landmarks_2d) != len(landmark_indices):
            return None, None, None
            
        face_2d_image_points = np.array(landmarks_2d, dtype=np.float64)
        
        focal_length = max(img_w, img_h) * 1.2 
        cam_matrix = np.array([
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # Sử dụng RANSAC để tăng độ robust
        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d_model_points, 
            face_2d_image_points, 
            cam_matrix, 
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None, None
        
        # Chuyển đổi rotation vector thành rotation matrix
        rmat, _ = cv2.Rodrigues(rot_vec)
        
        # Tính toán các góc Euler
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        yaw = angles[1]
        pitch = angles[0]  
        roll = angles[2]
        
        # Lọc các góc bất thường
        if abs(yaw) > 90 or abs(pitch) > 60 or abs(roll) > 45:
            return None, None, None
        return yaw, pitch, roll
        
    except Exception:
        return None, None, None
        
def get_position_specific_threshold(seat_id):
    """Lấy ngưỡng phù hợp với từng vị trí ghế"""
    if seat_id in CAMERA_ANGLE_CORRECTION:
        config = CAMERA_ANGLE_CORRECTION[seat_id]
        return config["min_threshold"], config["sensitivity"], config["yaw_offset"]
    else:
        return BASE_YAW_THRESHOLD, 1.0, 0
    

def smooth_behavior_detection(behavior_history, seat_id, current_behavior, confidence):
    """Làm mượt việc phát hiện hành vi với trọng số confidence"""
    if seat_id not in behavior_history:
        behavior_history[seat_id] = []
    behavior_history[seat_id].append((current_behavior, confidence))
    
    # Chỉ giữ lại CONFIDENCE_FRAMES frame gần nhất
    if len(behavior_history[seat_id]) > CONFIDENCE_FRAMES:
        behavior_history[seat_id] = behavior_history[seat_id][-CONFIDENCE_FRAMES:]
    
    # Tính toán hành vi chủ đạo với trọng số confidence
    if len(behavior_history[seat_id]) >= CONFIDENCE_FRAMES:
        behavior_scores = {"normal": 0, "quay_trai": 0, "quay_phai": 0}
        
        for behavior, conf in behavior_history[seat_id]:
            behavior_scores[behavior] += conf
        # Trả về hành vi có điểm số cao nhất
        dominant_behavior = max(behavior_scores, key=behavior_scores.get)
        max_score = behavior_scores[dominant_behavior]
        
        if max_score > MIN_CONFIDENCE_SCORE * CONFIDENCE_FRAMES * 0.6:
            return dominant_behavior
    
    return "normal"
