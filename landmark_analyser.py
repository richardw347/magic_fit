from typing import Tuple, List
from utils import WaveState, WaveObservation, calc_angle
import mediapipe as mp

mp_pose = mp.solutions.pose


class LandmarkAnalyser:
    def __init__(
        self,
        wave_angle_thresh: float = 45.0,
        wave_min: float = 15.0,
        wave_max: float = 75.0,
    ) -> None:
        self.wave_angle_thresh = wave_angle_thresh
        self.wave_count = 0
        self.half_rep = 0
        self.wave_state = WaveState.WAVE_INIT
        self.min_angle = wave_min
        self.max_angle = wave_max
        self.prev_angle = 0
        self.max_performance = 0

    def reset_wave_count(self):
        self.wave_count = 0

    def __call__(
        self, landmarks, width: int, height: int
    ) -> Tuple[List, List, float, float, WaveObservation, WaveState, int, float, float]:

        # Here we extract the x,y coordindates for the relevant landmark features and
        # unnormalise the coordinates to so back to image coordinates
        # mp_pose normalises the values to between 0 and 1.
        hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height,
        ]
        shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height,
        ]
        elbow = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height,
        ]
        wrist = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * height,
        ]

        # I don't use the shoulder angle later but I've kept the calculation and visualisation
        # just for reference
        shoulder_angle = abs(calc_angle(hip, shoulder, elbow))
        elbow_angle = abs(calc_angle(shoulder, elbow, wrist))

        # Here the wave direction is calculated by subtracting the current elbow angle from the
        # previous observation, positive number means the arm is moving OUTWARDS negative means
        # INWARDS
        wave_direction = 0
        if self.prev_angle != 0:
            wave_direction = int(elbow_angle - self.prev_angle)
        self.prev_angle = elbow_angle

        # Classify the wave as WAVE_IN or WAVE_OUT and
        # handle the state logic
        wave_obs = self._classify_wave_angle(elbow_angle)
        self._handle_wave_logic(wave_obs, wave_direction)

        # Here's an example of performance monitoring
        # we calculate how close the user got to the ideal target
        # of 75 degrees and convert that to a percentage score
        max_performance = 0
        wave_perc = 0
        if self.wave_state in [WaveState.WAVE_INIT, WaveState.WAVE_STARTED]:
            max_performance = 0
            wave_perc = 0
        else:
            wave_perc = self._calc_wave_percentage(elbow_angle, wave_direction)
            self.max_performance = max(
                self._calc_performance(elbow_angle), self.max_performance
            )
            max_performance = self.max_performance

        return (
            shoulder,
            elbow,
            shoulder_angle,
            elbow_angle,
            wave_obs,
            self.wave_state,
            self.wave_count,
            wave_perc,
            max_performance,
        )

    def _classify_wave_angle(self, elbow_angle):
        wave_observation = None
        if elbow_angle < self.wave_angle_thresh:
            wave_observation = WaveObservation.WAVE_IN
        else:
            wave_observation = WaveObservation.WAVE_OUT

        return wave_observation

    def _calc_performance(self, wave_angle: float):
        return (wave_angle - self.min_angle) / (self.max_angle - self.min_angle) * 100.0

    def _calc_wave_percentage(self, wave_angle: float, move_direction: float) -> float:
        perc = 0
        if move_direction > 0:
            perc = (
                (wave_angle - self.min_angle) / (self.max_angle - self.min_angle) * 50.0
            )
        elif move_direction <= 0:
            perc = (
                (self.max_angle - wave_angle) / (self.max_angle - self.min_angle) * 50.0
            )
            perc += 50.0
        return perc

    def _handle_wave_logic(
        self, wave_observation: WaveObservation, wave_direction: int
    ):
        if self.wave_state == WaveState.WAVE_INIT:
            if wave_observation == WaveObservation.WAVE_IN:
                self.wave_state = WaveState.WAVE_STARTED

        elif self.wave_state == WaveState.WAVE_STARTED:
            if wave_direction > 0:
                self.wave_state = WaveState.WAVE_OUTWARD

        elif self.wave_state == WaveState.WAVE_OUTWARD:
            if wave_direction < 0:
                self.wave_state = WaveState.WAVE_INWARDS

        elif self.wave_state == WaveState.WAVE_INWARDS:
            if wave_observation == WaveObservation.WAVE_IN and wave_direction == 0:
                self.wave_state = WaveState.WAVE_COMPLETE
        elif self.wave_state == WaveState.WAVE_COMPLETE:
            self.wave_count += 1
            self.max_performance = 0
            self.wave_state = WaveState.WAVE_INIT
