import json
import cv2

import numpy as np

def getJSON(filePathandName):
    with open(filePathandName,'r') as f:
        return json.load(f)
    

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
