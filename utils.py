import numpy as np
import cv2
from typing import List
from enum import Enum


class CV2TextColors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)


class WaveState(Enum):
    WAVE_INIT = 0
    WAVE_STARTED = 1
    WAVE_OUTWARD = 2
    WAVE_INWARDS = 4
    WAVE_COMPLETE = 5


class WaveObservation(Enum):
    WAVE_IN = 0
    WAVE_OUT = 2


def calc_angle(j1: List, j2: List, j3: List):
    j1 = np.array(j1)
    j2 = np.array(j2)
    j3 = np.array(j3)

    angle_rads = np.arctan2(j3[1] - j2[1], j3[0] - j2[0]) - np.arctan2(
        j1[1] - j2[1], j1[0] - j2[0]
    )

    angle_deg = np.degrees(angle_rads)

    if angle_deg > 180.0:
        angle_deg -= 360.0

    return angle_deg


def draw_text_on_img(text, origin, scale, image, color=CV2TextColors.YELLOW):
    cv2.putText(
        img=image,
        text=text,
        org=origin,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=scale,
        color=color.value,
        thickness=2,
    )
    return image
