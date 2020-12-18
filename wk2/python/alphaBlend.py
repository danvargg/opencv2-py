import cv2
import numpy as np
from dataPath import DATA_PATH

# Read the foreground image with alpha channel
foreGroundImage = cv2.imread(DATA_PATH + "images/foreGroundAssetLarge.png", -1)

# Split png foreground image
b,g,r,a = cv2.split(foreGroundImage)

# Save the foregroung RGB content into a single object
foreground = cv2.merge((b,g,r))

# Save the alpha information into a single Mat
alpha = cv2.merge((a,a,a))

# Read background image
background = cv2.imread(DATA_PATH + "images/backGroundLarge.jpg")
# Convert uint8 to float
foreground = foreground.astype(float)
background = background.astype(float)
alpha = alpha.astype(float)/255

# Perform alpha blending
foreground = cv2.multiply(alpha, foreground)
background = cv2.multiply(1.0 - alpha, background)
outImage = cv2.add(foreground, background)

# Save output
cv2.imwrite("results/alphaBlend.png", outImage)

# Display output
cv2.imshow("outImg", outImage/255)
cv2.waitKey(0)