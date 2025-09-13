import math
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from utils.process_coords import CAMERA_ANGLE_CORRECTION


HEIGHT_PERSON = {
    "H300": (1.5, 100),     # (ratio, min_height)
    "H500": (1.8, 180),
    "H700": (1.6, 150),
}

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def calculate_angle(x, y, z):
    """
        Input: Coordinates (x, y) of 3 points
    """
    x = np.array(x) # First
    y = np.array(y) # Mid
    z = np.array(z) # End

    radians = np.arctan2(z[1]-y[1], z[0]-y[0]) - np.arctan2(x[1]-y[1], x[0]-y[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
    return angle

def process_stand_seat(landmarks, image):
    """
        Input: landmarks of only one person (33 points)
    """
    # print(landmarks)
    mp_pose = solutions.pose    # mp_pose.PoseLandmark = [0,1,...32]
    w, h  = image.shape[1], image.shape[0]
    LEFT_SHOULDER = [landmarks[11].x, landmarks[11].y]
    LEFT_HIP = [landmarks[23].x, landmarks[23].y]
    LEFT_KNEE = [landmarks[25].x, landmarks[25].y]
    #   print("left", landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility)
    if landmarks[23].visibility < 0.5 or landmarks[25].visibility < 0.5:
        angle_left = 0
    else:
        angle_left = calculate_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)

    RIGHT_SHOULDER = [landmarks[12].x, landmarks[12].y]
    RIGHT_HIP = [landmarks[24].x, landmarks[24].y]
    RIGHT_KNEE = [landmarks[26].x, landmarks[26].y]
    #   print("right", landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility)
    if landmarks[24].visibility < 0.5 or landmarks[26].visibility < 0.5:
        angle_right = 0
    else:
        angle_right = calculate_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)

    if angle_left > 160 or angle_right > 160:
        status = "STANDING",
        angle = (angle_left + angle_right) / 2 if angle_left > 0 and angle_right > 0 else max(angle_left, angle_right)
        shoulder = ((LEFT_SHOULDER[0] + RIGHT_SHOULDER[0])/2 * w, (LEFT_SHOULDER[1] + RIGHT_SHOULDER[1])/2 * h)
        # elif angle_left == 0 and angle_right == 0: 
        #   status = ""
        # cv2.putText(image, f"{status[0]}",
        #                     tuple(np.multiply(LEFT_SHOULDER, [w, h]).astype(int)),  
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
        #                         )
        # _{angle_left:.1f}_{angle_right:.1f}
        return angle, shoulder
    else:
        return None, None


def standing_base_height(x1, y1, x2, y2):
    # x1, y1, x2, y2 = person_box
    height, width = y2 - y1, x2 - x1
    y_center = (y1 + y2) / 2
    # Xử lý theo chiều cao và toạ độ
    if y_center < 300:
        if height > HEIGHT_PERSON["H300"][1] and height / width > HEIGHT_PERSON["H300"][0]:
            return height/width/HEIGHT_PERSON["H300"][0]*10 - 1 
    elif y_center < 500:
        if height > HEIGHT_PERSON["H500"][1] and height / width > HEIGHT_PERSON["H500"][0]:
            return height/width/HEIGHT_PERSON["H500"][0]*10 - 1
    else:
        if height > HEIGHT_PERSON["H700"][1] and height / width > HEIGHT_PERSON["H700"][0]:
            return height/width/HEIGHT_PERSON["H700"][0]*10 - 1
    return 0.0


def pad_to_square(img, pad_value=0):
    h, w = img.shape[:2]
    if h == w:
        return img, 0, 0
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=[pad_value]*3)
    return padded, left, top