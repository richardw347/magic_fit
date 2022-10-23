import cv2
import mediapipe as mp
import numpy as np
from landmark_analyser import LandmarkAnalyser
from utils import draw_text_on_img

VIDEO_FILE = "video/A.mp4"
# VIDEO_FILE = "video/B.mp4"
VISUALISE = True
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

analyser = LandmarkAnalyser(
    wave_angle_thresh=45.0,
    wave_min=15.0,
    wave_max=75.0,
)
cap = cv2.VideoCapture(VIDEO_FILE)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():

        # read a frame from the video
        success, image = cap.read()
        if not success:
            break

        # conver the color format and compute the blaze pose landmarks
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        landmarks = results.pose_landmarks.landmark

        # analyse the landmarks and extract the wave count, percentage and a number of other
        # useful properties
        (
            shoulder,
            elbow,
            shoulder_angle,
            elbow_angle,
            wave_obs,
            wave_state,
            wave_count,
            wave_percentage,
            wave_performance,
        ) = analyser(landmarks, image.shape[1], image.shape[0])

        print(wave_count, wave_percentage)

        if VISUALISE:

            # draw the landmarks on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            # draw some useful text on the image
            image = draw_text_on_img(
                "{:.2f}".format(shoulder_angle),
                (int(shoulder[0]), int(shoulder[1])),
                0.8,
                image,
            )
            image = draw_text_on_img(
                "{:.2f}".format(elbow_angle), (int(elbow[0]), int(elbow[1])), 0.8, image
            )
            image = draw_text_on_img(wave_obs.name, (50, 100), 1.5, image)
            image = draw_text_on_img(wave_state.name, (50, 200), 1.5, image)
            image = draw_text_on_img(f"WAVE COUNT: {wave_count}", (50, 300), 1.5, image)
            image = draw_text_on_img(
                f"WAVE PERC: {wave_percentage:.2f}%", (50, 400), 1.5, image
            )
            image = draw_text_on_img(
                f"WAVE PERF: {wave_performance:.2f}%", (50, 500), 1.5, image
            )

            cv2.imshow("MediaPipe Pose", image)

            if cv2.waitKey(5) == ord("q"):
                break


cv2.destroyWindow("MediaPipe Pose")
cap.release()
