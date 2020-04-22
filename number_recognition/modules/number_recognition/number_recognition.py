from PIL import Image
import pytesseract
import cv2
import os
from enum import Enum

class PreprocessEnum(Enum):
  THRESH="thresh"
  BLUR="blur"

class NumberRecognition:
  def __init__(self, image):
    self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  def recognize_text(self, preprocess=PreprocessEnum.THRESH):
    image = self.image
    # check to see if we should apply thresholding to preprocess the
    # image
    if preprocess == PreprocessEnum.THRESH:
      image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # make a check to see if median blurring should be done to remove
    # noise
    elif preprocess == PreprocessEnum.BLUR:
      image = cv2.medianBlur(image, 3)

  return pytesseract.image_to_string(Image.fromarray(image))