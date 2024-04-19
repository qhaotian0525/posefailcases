import mediapipe as mp
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


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

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
model_path = "./pose_landmarker_full.task"
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_poses = 5,
    min_pose_detection_confidence = 0.35,
    min_pose_presence_confidence = 0.35)

detector = PoseLandmarker.create_from_options(options)
imggg = cv2.imread("./d.png")
imggg = cv2.cvtColor(imggg, cv2.COLOR_BGR2RGB)
#mp_image = mp.Image.create_from_file('./a.png')
mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = imggg)
pose_landmarker_result = detector.detect(mp_image)
print(mp_image.numpy_view().shape)
annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
bgr_ = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
cv2.imshow("F", bgr_)
cv2.waitKey(0)

cv2.destroyAllWindows()