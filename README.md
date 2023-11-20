
# Servetrack: Volleyball Serve Visualization
<img src="https://github.com/wlangley512/servetrack/blob/main/assets/track3.gif">
Python script that analyzes any clear video of a volleyball serve. We can track the trajectory of the ball and get lots of interesting data*! 
Intended for use with scout camera view.
*Interesting data output coming soon

Insert Gifs Here
## Installation
1. Clone the repository
```
git clone https://github.com/wlangley512/servetrack.git
```
2. Install requirments.txt
```
pip install -r requirements.txt
```

## How to use

1. In your terminal navigate to the directory where you have cloned the repository.
2. The script can be executed with the following command:
3. Adjust the confidence and max_deviation as needed. I suggest starting with .5 or .6 confidence and 150 max_deviation. If your line starts to connect to false positives either lower max_deviation or raise confidence.
4. Output videos will be located in the ```output/``` directory
```
python yolotest.py --input_video_path --confidence(.0-1.0) --max_deviation_between_points
```
<img src= "https://github.com/wlangley512/servetrack/blob/main/assets/sample_input.png">

## Shortcomings
1. The dataset used by Servetrack requires a moderately clear picture to detect the ball on successive frames. For example, detection quality might degrade if the camera angle is low and bright flourescent ceiling lights are visible. (Insert example gif of me.mp4)
2. If a false positive is detected before the actual ball, this can lead to the track line not being drawn at all. This can be worked around by using the less trained dataset or by tweaking the confidence values.
3. New, much more robust, dataset will appear in the future. However, currently do not have the means to create a better one. So good output requires some tweaking!
4. Currently only handles balls being served away from the camera, please keep this in mind when running tests.
