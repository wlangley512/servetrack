import cv2
import sys
import numpy as np
from ultralytics import YOLO
import math
import time
from tqdm import tqdm
import os 

# Constants #
#towardaway = sys.argv[5]
video = sys.argv[1]
min_conf=float(sys.argv[2])
max_deviation = int(sys.argv[3])
model_name = sys.argv[4]
cap = cv2.VideoCapture(video)
label = None
line_color = (0, 0, 0)
total_mag = 0
mag_test = None
# Constants end #

model=YOLO(model_name)
classNames = ['ball']


center_points = []

# Video output creation 
filecount = 0
output_path = 'output/'
output_file = output_path + 'track%d.mp4' % filecount
file_exists = os.path.exists(output_path)
if not file_exists:
    os.makedirs(output_path)
while os.path.isfile(output_file):
    filecount += 1
    output_file = output_path + 'track%d.mp4' % filecount


# video output constants
fps = cap.get(5)
width = int(cap.get(3))
height = int(cap.get(4))
dim = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
tframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
final = cv2.VideoWriter(output_file,fourcc, fps, dim)
#####

# Progress bar
progress=tqdm(total=tframes)

# Function to filter false positives
def filter_outliers(center_points, max_deviation):
    filtered_points = [center_points[0]]

    for i in range(1, len(center_points)):
        prev_x, prev_y = filtered_points[-1]
        curr_x, curr_y = center_points[i]
        
        # Euclidean distance  
        distance = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)

        if distance <= max_deviation:
           filtered_points.append(center_points[i])

    
    return filtered_points

# Function to calculate angle between points
def calculate_angle(segment1, segment2): 
    vector1 = np.array([segment1[1][0] - segment1[0][0], segment1[1][1] - segment1[0][1]])
    vector2 = np.array([segment2[1][0] - segment2[0][0], segment2[1][1] - segment2[0][1]]) 
    angle_radians = np.arctan2(np.linalg.det([vector1, vector2]), np.dot(vector1, vector2))
    angle_degrees = np.degrees(angle_radians)
    return np.abs(math.ceil(angle_degrees))

# Function to track line of ball 
def line_render_away(points, img, line_color):
    balance = 30 / fps
    counter = 0
    consecutive_negative_frames = 0
    label = None
    tossed =  True
    falling = False
    served = False
    passed = False

    if len(points) < 3:
        return

    for i in range(3, len(points) - 1):
        counter+=1
        segment1 = (points[i-2], points[i-1])
        segment2 = (points[i-1], points[i])

        if points[i - 1] is None or points[i] is None:
            continue
        diff_x = points[i-1][0] - points[i][0]
        diff_y = points[i-2][1] - points[i][1]
        
        bal_diff_y = diff_y / balance
        bal_diff_x = math.ceil(diff_x / balance)
        angle = calculate_angle(segment1, segment2)
        
        if bal_diff_y < 0:
            consecutive_negative_frames += 1
        else: 
            consecutive_negative_frames = 0
        
        # Makes sure ball is falling before checking for change in angle 
        if consecutive_negative_frames >= 3 and counter > 15:
            falling = True

  
        #if this works it sucks
        #update: jesus christ
        if tossed:
            if falling and diff_y >= 0: # Start of Serve Red Line
                cv2.line(img, points[i-1], points[i], (0, 0, 255), 2)
                tossed = False
                falling = False
                served = True
            else:# Default Toss Green Line
                cv2.line(img, points[i-1], points[i], (0, 255, 0), 2)
                label = "Toss"
        elif served:
            if falling and diff_y >= 0: # Start of Pass Yellow Line
                cv2.line(img, points[i-1], points[i], (0, 255, 255), 2)
                served = False
                passed = True
            else:# Default Serve Red Line
                label = "Serve"
                cv2.line(img, points[i-1], points[i], (0, 0, 255), 2)
        elif passed: 
            max_deviation = 20
            label = "Pass"
            cv2.line(img, points[i-1], points[i], (0, 255, 255), 2)
    
    return label


while True:
    ret, img = cap.read() 
    cv2.putText(img, video, (300, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if not ret:
        break
    
    progress.update(1)
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            # Class Name
            cls = int(box.cls[0]) 
            
            if conf > min_conf:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                # Center points for circle
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)
                current_center_point = (cx, cy)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                
                # Update previous frame variables
                prev_frame_time = time.time()
                prev_center_point = current_center_point

                # Create mid points and bounding box
                center_points.append((cx, cy)) 
                cv2.rectangle (img, (x1, y1), (x2, y2), (255, 255, 0), 1)
                 
                #for pt in center_points:
                #     cv2.circle(img, pt, 2, (255, 0, 0), -1)
                
                # Outliers now filtered
                center_points = filter_outliers(center_points, max_deviation)

                currentClass = classNames[cls]
                if currentClass == "ball":
                    cv2.putText(img, f'{mag_test}', (max(30, x1), max(30, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
         
        #if towardaway == 1: # serving away
        mag_test = line_render_away(center_points, img, line_color)
        #elif towardaway = 0: #serving toward
        #    total_mag = 
    
    final.write(img)
    cv2.waitKey(1)
    
final.release()
progress.close()
