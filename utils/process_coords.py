import cv2
import numpy as np


# Format: (x1, y1, x2, y2, x3, y3)  (hình bình hành)  
#        D           C(x3, y3)
#        A(x1, y1)   B(x2, y2)
#          
FRONT_ROIS = {
    "A1": (390, 570, 600, 500, 470, 330), 
    "A2": (600, 500, 800, 430, 650, 290), 
    "A3": (800, 430, 950, 370, 790, 260), 
    "A4": (950, 370, 1040, 340, 880, 240),
    "A5": (1070, 360, 1130, 330, 980, 230),
    "A6": (1130, 330, 1180, 310, 1050, 220),

    "B1": (300, 370, 470, 330, 400, 220), 
    "B2": (470, 330, 640, 290, 550, 200), 
    "B3": (650, 290, 770, 260, 690, 180),
    "B4": (770, 260, 860, 240, 760, 170),
    "B5": (890, 250, 990, 230, 880, 170),
    # "B6": (990, 230, 1040, 220, 940, 160),

    "C1": (240, 250, 380, 220, 330, 140),
    "C2": (380, 220, 500, 220, 450, 140),
    "C3": (520, 220, 630, 190, 570, 130),
    "C4": (630, 190, 740, 180, 640, 120),
    "C5": (820, 200, 890, 180, 810, 120),

    "D5": (700, 150, 790, 140, 720, 100),

    "E5": (600, 110, 690, 100, 630, 30),

}

BACK_ROIS = {
    "A4": (530, 140, 590, 100, 520, 50),
    "A5": (380, 160, 430, 140, 370, 90),
    "A6": (280, 200, 380, 160, 310, 120),
    "A7": (170, 250, 280, 200, 220, 150),
    "A8": (70, 290, 170, 250, 110, 180),

    "B4": (660, 160, 730, 120, 620, 70),
    "B5": (470, 210, 580, 150, 470, 110),
    "B6": (360, 250, 470, 210, 390, 160),
    "B7": (220, 310, 360, 250, 310, 200),
    "B8": (100, 370, 220, 310, 160, 250),

    "C4": (810, 200, 860, 150, 750, 100),
    "C5": (670, 260, 770, 200, 610, 140),
    "C6": (530, 330, 670, 260, 520, 190),
    "C7": (370, 420, 530, 330, 400, 230),
    "C8": (230, 500, 370, 420, 250, 310),

    "D1": (1100, 150, 1130, 110, 1030, 70),
    "D2": (1080, 170, 1100, 150, 990, 110),
    "D3": (1050, 210, 1080, 180, 980, 100),
    "D4": (1000, 260, 1050, 210, 920, 130),
    "D5": (890, 370, 990, 280, 790, 210),
    "D6": (780, 470, 890, 370, 670, 290),
    "D7": (610, 600, 750, 500, 690, 260),
    "D8": (470, 730, 610, 600, 420, 420),

    "E1": (1230, 160, 1240, 140, 1160, 120),
    "E2": (1220, 180, 1230, 160, 1150, 140),
    "E3": (1200, 220, 1220, 180, 1140, 160),
    "E4": (1170, 300, 1200, 220, 1120, 190),
    "E5": (1150, 440, 1190, 380, 1020, 320),
    "E6": (1090, 560, 1150, 440, 980, 360),
    "E7": (980, 750, 1190, 560, 890, 470),
    "E8": (970, 760, 1010, 700, 670, 690),
}

CAMERA_ANGLE_CORRECTION = {
    # Cho bài toán phát hiện hướng quay theo vị trí ghế
    "A1": {"yaw_offset": -40, "sensitivity": 1.2, "min_threshold": 18},
    "A2": {"yaw_offset": -40, "sensitivity": 1.1, "min_threshold": 16}, 
    "A3": {"yaw_offset": -40, "sensitivity": 1.1, "min_threshold": 16},
    "A4": {"yaw_offset": -40, "sensitivity": 1.0, "min_threshold": 15},
    "A5": {"yaw_offset": -44, "sensitivity": 1.0, "min_threshold": 15},
    "A6": {"yaw_offset": -48, "sensitivity": 1.0, "min_threshold": 14},

    "B1": {"yaw_offset": -30, "sensitivity": 1.3, "min_threshold": 20},
    "B2": {"yaw_offset": -30, "sensitivity": 1.2, "min_threshold": 18},
    "B3": {"yaw_offset": -30, "sensitivity": 1.0, "min_threshold": 15},
    "B4": {"yaw_offset": -30, "sensitivity": 1.0, "min_threshold": 15},
    "B5": {"yaw_offset": -44, "sensitivity": 0.9, "min_threshold": 18}, #  false positive
    "B6": {"yaw_offset": -48, "sensitivity": 0.9, "min_threshold": 18},


    "C1": {"yaw_offset": -47, "sensitivity": 1.1, "min_threshold": 16},
    "C2": {"yaw_offset": -47, "sensitivity": 1.2, "min_threshold": 14}, 
    "C3": {"yaw_offset": -47, "sensitivity": 1.1, "min_threshold": 15},
    "C4": {"yaw_offset": -47, "sensitivity": 1.0, "min_threshold": 15},
    "C5": {"yaw_offset": -37, "sensitivity": 0.8, "min_threshold": 20}, 

    "D1": {"yaw_offset": -37, "sensitivity": 1.0, "min_threshold": 15},

    "D6": {"yaw_offset": -37, "sensitivity": 0.9, "min_threshold": 17},
}

def calculate_iou_rotated(boxA, boxB):
    """
        boxA: hình bình hành A (x1, y1, x2, y2, x3, y3)
        => (x4, y4) = (x1 + x3 - x2, y1 + y3 - y2)
        boxB: hình chữ nhật (x1, y1, x2, y2)
    """
    x1,y1,x2,y2,x3,y3 = map(float, boxA[:6])
    x4, y4 = x1 + x3 - x2, y1 + y3 - y2
    polyA = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], dtype=np.float32)

    x_min,y_min,x_max,y_max = map(float, boxB[:4])
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    polyB = np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]], dtype=np.float32)

    hullA = cv2.convexHull(polyA)
    hullB = cv2.convexHull(polyB)

    inter = cv2.intersectConvexConvex(hullA, hullB)
    # OpenCV versions return different shapes; handle both
    inter_pts = inter[1] if isinstance(inter, tuple) and len(inter) > 1 else inter
    if inter_pts is None or len(inter_pts) == 0:
        return 0.0

    interArea = abs(cv2.contourArea(inter_pts))
    areaA = abs(cv2.contourArea(hullA))
    areaB = abs(cv2.contourArea(hullB))
    # print("Areas:", areaA, areaB, interArea)
    union = areaA + areaB - interArea
    if union <= 1e-8:
        return 0.0
    return float(interArea / union)
    
# print(calculate_iou_rotated((0, 0, 1, 0, 2, 2), (1, 0, 2, 1)))

def calculate_iou(boxA, boxB, format="xyxy"):    
    if format == "xywh":
        boxA = (boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3])
        boxB = (boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3])

    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    interArea = inter_w * inter_h
        
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if boxAArea + boxBArea - interArea == 0:
        return 0.0
        
    return interArea / float(boxAArea + boxBArea - interArea)
