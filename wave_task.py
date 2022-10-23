import cv2
import mediapipe as mp
import numpy as np
from landmark_analyser import LandmarkAnalyser
from utils import draw_text_on_img, WaveState, CV2TextColors

VIDEO_FILE = "video/A.mp4"
# VIDEO_FILE = "video/B.mp4"
VISUALISE = True
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

analyser = LandmarkAnalyser(
    wave_angle_thresh=45.0,
    wave_min=15.0,
    wave_max=65.0,
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

        print(f"Wave count: {wave_count}, wave percentage: {wave_percentage}%")

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

            text_size = 1.2
            x_orig = 50
            y_orig = 100
            y_mod = 50
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
            image = draw_text_on_img(
                f"Observation: {wave_obs.name}", (x_orig, y_orig), text_size, image
            )
            image = draw_text_on_img(
                f"State: {wave_state.name}",
                (x_orig, y_orig + (y_mod * 1)),
                text_size,
                image,
            )
            image = draw_text_on_img(
                f"Count: {wave_count}", (x_orig, y_orig + (y_mod * 2)), text_size, image
            )
            image = draw_text_on_img(
                f"Percentage: {wave_percentage:.2f}%",
                (x_orig, y_orig + (y_mod * 3)),
                text_size,
                image,
            )
            image = draw_text_on_img(
                f"Performance: {wave_performance:.2f}%",
                (x_orig, y_orig + (y_mod * 4)),
                text_size,
                image,
            )

            # display performance assessment when in the right states
            if (
                WaveState.WAVE_OUTWARD.value
                < wave_state.value
                <= WaveState.WAVE_COMPLETE.value
            ):
                if wave_performance > 90:
                    image = draw_text_on_img(
                        "GOOD",
                        (x_orig, y_orig + (y_mod * 5)),
                        text_size,
                        image,
                        color=CV2TextColors.GREEN,
                    )
                else:
                    image = draw_text_on_img(
                        "GO FURTHER",
                        (x_orig, y_orig + (y_mod * 5)),
                        text_size,
                        image,
                        color=CV2TextColors.RED,
                    )
            cv2.imshow("MediaPipe Pose", image)

            if cv2.waitKey(5) == ord("q"):
                break


cv2.destroyWindow("MediaPipe Pose")
cap.release()
