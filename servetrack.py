#github
#create github
#create serve detection

import cv2
import sys
import numpy as np
from ultralytics import YOLO
import math
import time
from tqdm import tqdm
import os 
import logging

# Constants #
video = sys.argv[1]
max_deviation = int(sys.argv[3])
min_conf=float(sys.argv[2])
cap = cv2.VideoCapture(video)
line_color = (0, 0, 0)
total_mag = 0
mag_test = None
prev_center_x = 0
# Constants end #

model=YOLO('Yolo-Weights/best100.pt')
classNames = ['ball']


center_points = []


filecount = 0
output_path = 'output/angletest/track%d.mp4' % filecount
while os.path.isfile(output_path):
    filecount += 1
    output_path = 'output/angletest/track%d.mp4' % filecount

####LOG MAKER for debugging purposes
logcount = 0
log_file = 'output/logs/angletest/log%d.txt' % logcount
while os.path.isfile(log_file):
    logcount += 1
    log_file = 'output/logs/angletest/log%d.txt' % logcount
os.makedirs(os.path.dirname(log_file), exist_ok=True)
lf = open(log_file, 'w+')
sys.stdout = log_file
####

lf.write(video)
fps = cap.get(5)
width = int(cap.get(3))
height = int(cap.get(4))
dim = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
tframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#output location
final = cv2.VideoWriter(output_path,fourcc, fps, dim)
progress=tqdm(total=tframes)

count = 0

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

def calculate_angle(segment1, segment2): 
    vector1 = np.array([segment1[1][0] - segment1[0][0], segment1[1][1] - segment1[0][1]])
    vector2 = np.array([segment2[1][0] - segment2[0][0], segment2[1][1] - segment2[0][1]]) 
    #dot_product = np.dot(vector1, vector2)
    #magnitude1 = np.linalg.norm(vector1)
    #magnitude2 = np.linalg.norm(vector2)
    #cosine_angle = dot_product / (magnitude1 * magnitude2)
    #angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) 
    #angle_degrees = np.degrees(angle_radians)
    angle_radians = np.arctan2(np.linalg.det([vector1, vector2]), np.dot(vector1, vector2))
    angle_degrees = np.degrees(angle_radians)
    return np.abs(angle_degrees)

def line_render(points, img, line_color):
    balance = 30 / fps
    angle = 0
    counter = 0
    bagaga = 0
    diff_y = 0
    diff_x = 0
    bal_diff_y = 0
    bal_diff_x = 0
    poop_diff = None
    tossed =  True
    served = False
    passed = False
    #color = (0, 255, 255)
    if len(points) < 3:
        return
    for i in range(3, len(points) - 1):
        #print("X Point: ", points[i][0], file=lf)
        #total_mag = 0
        segment1 = (points[i-2], points[i-1])
        segment2 = (points[i-1], points[i])

        if points[i - 1] is None or points[i] is None:
            continue
        diff_x = points[i-3][0] - points[i][0]
        diff_y = points[i-3][1] - points[i][1]
        
        bal_diff_y = diff_y / balance
        bal_diff_x = diff_x / balance 
        angle = calculate_angle(segment1, segment2)
        
        print("angle: " ,  angle, file=lf)
       
        #prevent false positive at start if player lower balls before toss
        counter+=1

        #if this works it sucks
        #update: jesus christ
        if tossed:
            cv2.line(img, points[i-1], points[i], (0, 255, 0), 2)
            poop_diff = "Toss"
            tossed = True
            if (bal_diff_y <= -10 and counter > 15 and np.abs(bal_diff_x) > 3):
                if angle > 10 and angle < 70:
                    tossed = False
                    served = True
        elif served:
            #if np.abs(diff_x) > 3 and np.abs(diff_y) > 3:
            poop_diff = "Serv"
            cv2.line(img, points[i-1], points[i], (0, 0, 255), 2)
            #else: 
            #    cv2.line(img, points[i-1], points[i], (0, 255, 0), 2)

    return round(angle, 2), bal_diff_y, bal_diff_x, counter 
while True:
    ret, img = cap.read()
    
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
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

                # Create mid points and bounding box
                center_points.append((cx, cy)) 
                cv2.rectangle (img, (x1, y1), (x2, y2), (255, 255, 0), 1)
                 
                #for pt in center_points:
                #     cv2.circle(img, pt, 2, (255, 0, 0), -1)
                
                # Outliers now filtered
                center_points = filter_outliers(center_points, max_deviation)

                #mag_test = change_line_color(center_points, img, line_color)
                #print("line color after function call: ", line_color) 
                currentClass = classNames[cls]
                #if currentClass == "ball":
                    #if line_color == (0, 255, 0):
                    #    cv2.putText(img, f'{currentClass} {conf}', (max(0, x1), max(30, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    #elif line_color == (0, 0, 255):
                    #    cv2.putText(img, f'le epic serve detected', (max(0, x1), max(30, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if currentClass == "ball":
                    cv2.putText(img, f'{total_mag}', (max(0, x1), max(30, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
        total_mag = line_render(center_points, img, line_color)
        #Draw line
        #if len(center_points)>=2:
        #    for i in range (1, len(center_points)):
        #        cv2.line(img, center_points[i - 1], center_points[i], line_color, 2)
    
    
    final.write(img)
    #cv2.imshow("Image", img)
    cv2.waitKey(1)
    

final.release()
progress.close()
lf.close()