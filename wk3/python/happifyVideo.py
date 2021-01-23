import cv2,dlib,time,math
import numpy as np
import mls as mls
import faceBlendCommon as fbc
from dataPath import DATA_PATH

mls.GRID = 30

# Function to add boundary points of the image to the given set of points
def addBoundaryPoints(cols, rows, points):
  # include the points on the boundaries
  points = np.append(points,[[0, 0]],axis=0)
  points = np.append(points,[[0, cols-1]],axis=0)
  points = np.append(points,[[rows-1, 0]],axis=0)
  points = np.append(points,[[rows-1, cols-1]],axis=0)
  points = np.append(points,[[0, cols/2]],axis=0)
  points = np.append(points,[[rows/2, 0]],axis=0)
  points = np.append(points,[[rows-1, cols/2]],axis=0)
  points = np.append(points,[[rows/2, cols-1]],axis=0)
  return points

# Variables for resizing to a standard height
RESIZE_HEIGHT = 480
FACE_DOWNSAMPLE_RATIO = 1.5
SKIP_FRAMES = 2

# Varibales for Dlib
modelPath = DATA_PATH + "models/shape_predictor_68_face_landmarks.dat"
faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(modelPath)

# Amount of bulge to be given for fatify
offset1 = 1.5
offset2 = 1.6

# Points that should not move
anchorPoints = [8, 30]

# Points that will be deformed
# For lips
deformedPoints1 = [48, 57, 54]
# For eyes
deformedPoints2 = [21, 22, 36, 45]

# Setup the video stream
cap = cv2.VideoCapture(DATA_PATH + "videos/sample-video.mp4")

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Variables for Stabilization
isFirstFrame = False
sigma = 100
count = 0
while(1):
  t = time.time()

  # Read an image and get the landmark points
  ret, src = cap.read()
  height, width = src.shape[:2]
  IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
  src = cv2.resize(src,None,
                     fx=1.0/IMAGE_RESIZE,
                     fy=1.0/IMAGE_RESIZE,
                     interpolation = cv2.INTER_LINEAR)

  # find landmarks after skipping SKIP_FRAMES number of frames
  if (count % SKIP_FRAMES == 0):
    landmarks = fbc.getLandmarks(faceDetector, landmarkDetector, cv2.cvtColor(src, cv2.COLOR_BGR2RGB), FACE_DOWNSAMPLE_RATIO)

  if len(landmarks) != 68:
    print("points no detected")
    continue

  ################ Optical Flow and Stabilization Code #####################
  srcGray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

  if(isFirstFrame == False):
    isFirstFrame = True
    landmarksPrev = np.array(landmarks, np.float32)
    srcGrayPrev = np.copy(srcGray)

  lk_params = dict( winSize  = (101,101),maxLevel = 5,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
  landmarksNext, st , err = cv2.calcOpticalFlowPyrLK(srcGrayPrev, srcGray, landmarksPrev, np.array(landmarks,np.float32),**lk_params)

  # Final landmark points are a weighted average of detected landmarks and tracked landmarks
  for k in range(0,len(landmarks)):
    d = cv2.norm(np.array(landmarks[k]) - landmarksNext[k])
    alpha = math.exp(-d*d/sigma)
    landmarks[k] = (1 - alpha) * np.array(landmarks[k]) + alpha * landmarksNext[k]
    landmarks[k] = fbc.constrainPoint(landmarks[k], src.shape[1], src.shape[0])

  # Update varibales for next pass
  landmarksPrev = np.array(landmarks, np.float32)
  srcGrayPrev = srcGray
  ################ End of Optical Flow and Stabilization Code ###############


  # Set the center to tip of chin
  center1x, center1y = landmarks[8][0], landmarks[8][1]
  # Set the center to point on nose
  center2x, center2y = landmarks[28][0], landmarks[28][1]

  # Variables for storing the original and deformed points
  srcPoints = []
  dstPoints=[]

  # Adding the original and deformed points using the landmark points
  for idx in anchorPoints:
    srcPoints.append([landmarks[idx][0], landmarks[idx][1]])
    dstPoints.append([landmarks[idx][0], landmarks[idx][1]])

  for idx in deformedPoints1:
    srcPoints.append([landmarks[idx][0], landmarks[idx][1]])
    dstPoints.append([offset1*(landmarks[idx][0] - center1x) + center1x, offset1*(landmarks[idx][1] - center1y) + center1y])

  for idx in deformedPoints2:
    srcPoints.append([landmarks[idx][0], landmarks[idx][1]])
    dstPoints.append([offset2*(landmarks[idx][0] - center2x) + center2x, offset2*(landmarks[idx][1] - center2y) + center2y])

  # Converting them to numpy arrays
  srcPoints = np.array(srcPoints)
  dstPoints = np.array(dstPoints)

  # Adding the boundary points to keep the image stable globally
  srcPoints = addBoundaryPoints(src.shape[0],src.shape[1],srcPoints)
  dstPoints = addBoundaryPoints(src.shape[0],src.shape[1],dstPoints)

  print("Points gathered {}".format(time.time() - t))

  # Performing moving least squares deformation on the image using the points gathered above
  dst = mls.MLSWarpImage(src, srcPoints, dstPoints)

  print("Warping done {}".format(time.time() - t))

  cv2.imshow('distorted', dst)
  if cv2.waitKey(1) & 0xFF == 27:
    break
  count += 1

cap.release()
