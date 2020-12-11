import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# For webcam input:
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
while cap.isOpened():
  success, image = cap.read()
  if not success:
    break

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = pose.process(image)

  # Draw the pose annotation on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  mp_drawing.draw_landmarks(
      image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
  image = cv2.circle(image, (150, 22), radius=0, color=(255, 0, 255), thickness=100)

  #print('nose landmark:', results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
  #print(results.pose_landmarks.PoseLandmark)
  #print(mp_pose.PoseLandmark.landmark[mp_pose.PoseLandmark.NOSE])
  #print(mp_pose.PoseLandmark.NOSE.value)

  #print(mp_pose.PoseLandmark.LEFT_EAR.value)
  #print(mp_pose.PoseLandmark.RIGHT_HIP.value)
  #print(mp_pose.PoseLandmark.LEFT_EYE)
  #print(results.pose_landmarks)
  #print(results.pose_landmarks)
  #print(mp_pose.POSE_CONNECTIONS)


  cv2.imshow('MediaPipe Pose', image)
  
  print(
  f'Nose coordinates: ('
  f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x}, '
  f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y})'
  )

  # if mp_pose.PoseLandmark = 11:
  #   print("yes")

  if cv2.waitKey(5) & 0xFF == 27:
    break


pose.close()
cap.release()

