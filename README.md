# MediaPipe Pose

## Demo

```
source mp_env/bin/activate
python3 pose-recorded.py
python3 pose-webcam.py
```

## Output Format

<https://google.github.io/mediapipe/solutions/pose#pose_landmarks>

* pose_landmarks
  * x
  * y
  * z
  * visibility
* pose_world_landmarks
* segmentation_mask

All joints:

```
['LEFT_ANKLE', 'LEFT_EAR', 'LEFT_ELBOW', 'LEFT_EYE', 'LEFT_EYE_INNER', 'LEFT_EYE_OUTER', 'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_HIP', 'LEFT_INDEX', 'LEFT_KNEE', 'LEFT_PINKY', 'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_WRIST', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'NOSE', 'RIGHT_ANKLE', 'RIGHT_EAR', 'RIGHT_ELBOW', 'RIGHT_EYE', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER', 'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_HIP', 'RIGHT_INDEX', 'RIGHT_KNEE', 'RIGHT_PINKY', 'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_WRIST']
```