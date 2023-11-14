
# Servetrack: Volleyball Serve Visualization

Python script that analyzes any clear video of a volleyball serve. We can track the trajectory of the ball and get lots of interesting data! 
Intended for use with scout camera view.h

Insert Gifs Here
## Installation
1. Clone the repository
```
git clone
```
2. Install required  


## How to use

1. In your terminal navigate to the directory where you have cloned the repository.
2. The script can be executed with the following command:
```
python yolotest.py --input_video_path --confidence(.0-1.0) --max_deviation_between_points
```
3. Adjust the confidence and max_deviation as needed.
4. Output videos will be located in the ```output/``` directory

## Shortcomings
1. The dataset used by Servetrack requires a moderately clear picture to detect the ball on successive frames. For example, detection quality might degrade if the camera angle is low and bright flourescent ceiling lights are visible. (Insert example gif of me.mp4)
2. If a false positive is detected before the actual ball, this can lead to the track line not being drawn at all. This can be worked around by using the less trained dataset or by tweaking the confidence values.