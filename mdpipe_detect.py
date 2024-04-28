import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  MARGIN = 10  # pixels
  ROW_SIZE = 10  # pixels
  FONT_SIZE = 1
  FONT_THICKNESS = 1
  TEXT_COLOR = (255, 0, 0)  # red
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image






def model_init(img_indicator, model_path) -> ObjectDetector:
  if img_indicator == True:
    mode = VisionRunningMode.IMAGE
  else:
    mode = VisionRunningMode.IMAGE
  options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path= model_path),
    max_results=5,
    running_mode=mode,
    category_allowlist = ["person"],
    score_threshold = 0.5
    )
  detector = ObjectDetector.create_from_options(options)
  return detector


def model_predict(detector, file_path,file_name, img_indicator, show_img):
  if img_indicator == True:
    img = cv2.imread(file_path + file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = img)
    detection_result = detector.detect(mp_image)
    image_copy = np.copy(mp_image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    if show_img == True:
      cv2.imshow("F", rgb_annotated_image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    cv2.imwrite("PROCESSED_"+file_name, rgb_annotated_image)

    return 1
  
  else:
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(file_path + file_name)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 

    out = cv2.VideoWriter(
    file_path + 'output-' + file_name ,
    fourcc,
    frame_rate,
    frame_size)

    while True:
      ret, f = cap.read()
      if ret == False:
        break

      f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
      
      mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = f)
      image_copy = np.copy(mp_image.numpy_view())
      detection_result = detector.detect(mp_image)
      annotated_image = visualize(image_copy, detection_result)
      rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
      out.write(rgb_annotated_image.astype('uint8'))

    cap.release()
    out.release()

    return 1



    







'''model_path = './efficientdet_lite0.tflite'



options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path= model_path),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE,
    category_allowlist = ["person"],
    score_threshold = 0.5
    )
detector = ObjectDetector.create_from_options(options)

imggg = cv2.imread("./e.png")
imggg = cv2.cvtColor(imggg, cv2.COLOR_BGR2RGB)
#mp_image = mp.Image.create_from_file('./a.png')
mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = imggg)

detection_result = detector.detect(mp_image)
image_copy = np.copy(mp_image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2.imshow("F", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(detection_result.detections[0].bounding_box)'''

if __name__ == "__main__":
  model_path = './efficientdet_lite0.tflite'
  video_path = "./"
  video_name = "fps_test.mp4"

  img_indicator = False
  detector = model_init(img_indicator, model_path)
  model_predict(detector, video_path, video_name, img_indicator, False)
  

