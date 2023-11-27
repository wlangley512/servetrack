import sys
import numpy as np
from ultralytics import YOLO
import math
import cv2
import time
from tqdm import tqdm
import os
from matplotlib import pyplot as plt

speed_flag = None
elapsed_time = 0
class Stopwatch:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        elapsed_time = time.time() - self.start_time
        self.start_time = None
        return elapsed_time

stopwatch = Stopwatch()

useful_data = None 
flag1 = True
flag2 = True
mph = .681818
# Constants #
#towardaway = sys.argv[5]
video = sys.argv[1]
min_conf=float(sys.argv[2])
max_deviation = int(sys.argv[3])
model_name = sys.argv[4]
cap = cv2.VideoCapture(video)
point_color = 'g'
output_path_video = 'output/videos/'
output_path_graph = 'output/graphs/'
# Constants end #

model=YOLO(model_name)
classNames = ['ball']


center_points = []

# Video and graph output creation 
videocount = 0
output_file_video = output_path_video + 'track%d.mp4' % videocount
video_file_exists = os.path.exists(output_path_video)
if not video_file_exists:
    os.makedirs(output_path_video)
while os.path.isfile(output_file_video):
    videocount += 1
    output_file_video = output_path_video + 'track%d.mp4' % videocount

graphcount = 0
output_file_graph = output_path_graph + 'graph%d.png' % graphcount
graph_file_exists = os.path.exists(output_path_graph)
if not graph_file_exists:
    os.makedirs(output_path_graph)
while os.path.isfile(output_file_graph):
    graphcount += 1
    output_file_graph = output_path_graph + 'graph%d.png' % graphcount
######################################


# Video output constants
fps = cap.get(5)
width = int(cap.get(3))
height = int(cap.get(4))
dim = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
tframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
final = cv2.VideoWriter(output_file_video,fourcc, fps, dim)
######################################
#####
first_frame = True
font = cv2.FONT_HERSHEY_SIMPLEX 
bl = []
br = []
tl = []
tr = []
count = 0
dotcount = 0
# Progress bar
progress=tqdm(total=tframes)

# plotting stuff
fig = plt.figure()
ax = plt.subplot(projection='3d')
ax1 = plt.subplot(projection='3d')

def mouse_event(event, x, y, flags, params):
    global count, first_frame, bl, br, tl, tr
    if event == cv2.EVENT_LBUTTONDOWN:
        #cv2.imshow("img", first_frame)
        print(x, "test", y)
        #cv2.putText(img, str(x) + ',' +
        #    str(y),
        #    (x,y), font, 1, 
            #(255, 255, 0), 2) 
        count+=1
        if count == 1:
            bl = [x, y]
        if count == 2:
            br = [x, y]
        if count == 3:
            tl = [x, y]
            print(first_frame)
        if count == 4:
            tr = [x, y]
            first_frame = False
            #cv2.imshow("yo", img)
            print(first_frame)


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

#def line_color(falling, tossed, served, passed)

# Function to calculate angle between points
def calculate_angle(segment1, segment2): 
    vector1 = np.array([segment1[1][0] - segment1[0][0], segment1[1][1] - segment1[0][1]])
    vector2 = np.array([segment2[1][0] - segment2[0][0], segment2[1][1] - segment2[0][1]]) 
    angle_radians = np.arctan2(np.linalg.det([vector1, vector2]), np.dot(vector1, vector2))
    angle_degrees = np.degrees(angle_radians)
    return np.abs(math.ceil(angle_degrees))

# Function to track line of ball 
def line_render_away(points, img):
    global point_color
    global elapsed_time
    global stopwatch
    global speed_flag 
    balance = 30 / fps
    global flag1
    global flag2 
    tossed =  True
    falling = False
    served = False
    passed = False
    counter = 0
    consecutive_negative_frames = 0
    label = "Toss"
    
    if speed_flag and flag1:
        stopwatch.start()
        print("start") 
        flag1 = False
    for i in range(3, len(points)):
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
                speed_flag = True
                cv2.line(img, points[i-1], points[i], (0, 0, 255), 2)
                point_color = 'r'
                tossed = False
                falling = False
                served = True
                #stopwatch.start()
            else:# Default Toss Green Line
                cv2.line(img, points[i-1], points[i], (0, 255, 0), 2)
                point_color = 'g'
                label = "Toss"
        elif served:
            if falling and diff_y >= 0: # Start of Pass Yellow Line
                speed_flag = False
                cv2.line(img, points[i-1], points[i], (0, 255, 255), 2)
                point_color = 'y'
                served = False 
                #elapsed_time = stopwatch.stop()
                passed = True
            else:# Default Serve Red Line
                label = "Serve"
                cv2.line(img, points[i-1], points[i], (0, 0, 255), 2)
                point_color = 'r'
        elif passed:
            max_deviation = 20
            label = "Pass"
            cv2.line(img, points[i-1], points[i], (0, 255, 255), 2)
            point_color = 'y'
    
    if speed_flag == False and flag2:
        elapsed_time = stopwatch.stop()
        print("stop")
        flag2 = False
    
    return label, speed_flag

count = 0
while True:
    ret, img = cap.read() 
    
    cv2.putText(img, video, (width-300, height-200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, "Max deviation: " + f"{max_deviation}", (width-300, height-220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, "Confidence: " + f"{min_conf}", (width-300, height-240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if not ret:
        break
    
    progress.update(1)
    results = model(img, stream=True, verbose=False)
    if first_frame:
        cv2.imshow("img", img)
        cv2.setMouseCallback('img', mouse_event)
    else:
        cv2.destroyAllWindows()
        for r in results:
            boxes = r.boxes
            useful_data = line_render_away(center_points, img)
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
                    

                    # Create mid points and bounding box
                    center_points.append((cx, cy)) 
                    cv2.rectangle (img, (x1, y1), (x2, y2), (255, 255, 0), 1)
                     
                    #for pt in center_points:
                            #     cv2.circle(img, pt, 2, (255, 0, 0), -1)
                    
                    # Outliers now filtered
                    center_points = filter_outliers(center_points, max_deviation)
                    currentClass = classNames[cls]
                    if currentClass == "ball":
                        cv2.putText(img, f'{useful_data}', (max(30, x1), max(30, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    
                    ax.view_init(-90, -90)
                    
                    # Court
                    x, y, z = zip(*[point + [0] for point in [bl, br, tl, tr]])
                    x, y, z = np.array(x), np.array(y), np.array(y)
                    # Plotting the points and connecting lines in 3D
                    ax.plot(x.tolist() + [x[0]], y.tolist() + [y[0]], 0, c='r', zorder=1)
                    
                    # Ball 
                    # pretty gross dot size code 
                    ax.plot(cx, cy,0,zorder = 2,  marker = 'o', linewidth = 1, c = point_color, markersize=min(7,5-dotcount)) 
                    dotcount+=0.03 
            #if towardaway == 1: # serving away
            #elif towardaway = 0: #serving toward
            #    total_mag = 
    
    cv2.putText(img, "Label: " + f"{useful_data}", (width-300, height-280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, "Time: " + f"{elapsed_time:.2f}", (width-300, height-260), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.waitKey(0)
    count +=1 
    final.write(img)

print(elapsed_time)
ax.set_zlim(height, 0)
plt.xlim(0, width)
plt.ylim(0, height)

plt.gcf().set_size_inches(12, 12)
plt.savefig(output_file_graph, dpi=150, bbox_inches='tight')
final.release()
progress.close()
plt.show()
