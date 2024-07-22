from flask import (
    Flask,
    Response,
    render_template,
    redirect,
    request,
    send_from_directory,
    url_for,
)
from flask_socketio import SocketIO, emit
import cv2 as cv
import numpy as np
import time
import mediapipe as mp
import os
from Utils import get_face_area
from Eye_Dector_Module import EyeDetector as EyeDet
from Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from Attention_Scorer_Module import AttentionScorer as AttScorer
import logging
import threading


app = Flask(__name__)
socketio = SocketIO(app)

# Camera matrix obtained from the camera calibration script, using a 9x6 chessboard
camera_matrix = np.array(
    [[899.12150372, 0.0, 644.26261492], [0.0, 899.45280671, 372.28009436], [0, 0, 1]],
    dtype="double",
)

# Distortion coefficients obtained from the camera calibration script, using a 9x6 chessboard
dist_coeffs = np.array(
    [[-0.03792548, 0.09233237, 0.00419088, 0.00317323, -0.15804257]], dtype="double"
)


def _get_landmarks(lms):
    surface = 0
    for lms0 in lms:
        landmarks = [np.array([point.x, point.y, point.z]) for point in lms0.landmark]
        landmarks = np.array(landmarks)
        landmarks[landmarks[:, 0] < 0.0, 0] = 0.0
        landmarks[landmarks[:, 0] > 1.0, 0] = 1.0
        landmarks[landmarks[:, 1] < 0.0, 1] = 0.0
        landmarks[landmarks[:, 1] > 1.0, 1] = 1.0
        dx = landmarks[:, 0].max() - landmarks[:, 0].min()
        dy = landmarks[:, 1].max() - landmarks[:, 1].min()
        new_surface = dx * dy
        if new_surface > surface:
            biggest_face = landmarks
    return biggest_face


class FreshestFrame(threading.Thread):
    def __init__(self, capture, name="FreshestFrame"):
        super().__init__(name=name)
        self.capture = capture
        assert self.capture.isOpened(), "Capture device is not opened."
        self.cond = threading.Condition()
        self.running = False
        self.frame = None
        self.latestnum = 0
        self.callback = None
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            rv, img = self.capture.read()
            if not rv:
                time.sleep(0.1)
                continue
            counter += 1

            with self.cond:
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum + 1
                if seqnumber < 1:
                    seqnumber = 1

                rv = self.cond.wait_for(
                    lambda: self.latestnum >= seqnumber, timeout=timeout
                )
                if not rv:
                    return self.latestnum, self.frame

            return self.latestnum, self.frame


@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


def generate_frames():
    rtsp_url = "rtsp://admin:KADYNE@192.168.29.236/H.264"
    rtsp_url = 1
    cap = cv.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    fresh = FreshestFrame(cap)

    show_eye_proc = True
    show_axis = True
    verbose = False
    smooth_factor = 0.5
    ear_thresh = 0.15
    ear_time_thresh = 2
    gaze_thresh = 0.015
    gaze_time_thresh = 2
    pitch_thresh = 20  # updated threshold 30
    yaw_thresh = 20  # updated threshold 30
    roll_thresh = 20  # updated threshold 30
    pose_time_thresh = 2.5

    if not cv.useOptimized():
        try:
            cv.setUseOptimized(True)  # set OpenCV optimization to True
        except:
            print(
                "OpenCV optimization could not be set to True, the script may be slower than expected"
            )

    detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    Eye_det = EyeDet(show_processing=show_eye_proc)
    Head_pose = HeadPoseEst(show_axis=show_axis)
    t0 = time.perf_counter()
    Scorer = AttScorer(
        t_now=t0,
        ear_thresh=ear_thresh,
        gaze_time_thresh=gaze_time_thresh,
        roll_thresh=roll_thresh,
        pitch_thresh=pitch_thresh,
        yaw_thresh=yaw_thresh,
        ear_time_thresh=ear_time_thresh,
        gaze_thresh=gaze_thresh,
        pose_time_thresh=pose_time_thresh,
        verbose=verbose,
    )

    i = 0
    time.sleep(0.01)
    while True:
        t_now = time.perf_counter()
        fps = i / (t_now - t0)
        if fps == 0:
            fps = 10

        _, frame = fresh.read(seqnumber=i + 1)

        if frame is None:
            print("Can't receive frame from camera/stream end")
            break

        e1 = cv.getTickCount()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_size = frame.shape[1], frame.shape[0]
        gray = np.expand_dims(cv.bilateralFilter(gray, 5, 10, 10), axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)
        lms = detector.process(gray).multi_face_landmarks

        if lms:
            landmarks = _get_landmarks(lms)
            Eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size
            )
            ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)
            tired, perclos_score = Scorer.get_PERCLOS(t_now, fps, ear)
            gaze = Eye_det.get_Gaze_Score(
                frame=gray, landmarks=landmarks, frame_size=frame_size
            )
            frame_det, roll, pitch, yaw = Head_pose.get_pose(
                frame=frame, landmarks=landmarks, frame_size=frame_size
            )
            asleep, looking_away, distracted = Scorer.eval_scores(
                t_now=t_now,
                ear_score=ear,
                gaze_score=gaze,
                head_roll=roll,
                head_pitch=pitch,
                head_yaw=yaw,
            )
            if frame_det is not None:
                frame = frame_det

            if tired or asleep:
                cv.rectangle(
                    frame,
                    (270, frame.shape[0] - 50),
                    (520, frame.shape[0] - 20),
                    (0, 0, 255),
                    -1,
                )
                cv.putText(
                    frame,
                    "           ASLEEP",
                    (280, frame.shape[0] - 30),
                    cv.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )

            if distracted:
                cv.rectangle(
                    frame,
                    (270, frame.shape[0] - 50),
                    (520, frame.shape[0] - 20),
                    (0, 0, 255),
                    -1,
                )
                cv.putText(
                    frame,
                    "       NON-ATTENTIVE",
                    (280, frame.shape[0] - 30),
                    cv.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )

            if not tired and not distracted and not asleep:
                cv.rectangle(
                    frame,
                    (10, frame.shape[0] - 50),
                    (260, frame.shape[0] - 20),
                    (0, 100, 0),
                    -1,
                )
                cv.putText(
                    frame,
                    "          ATTENTIVE",
                    (20, frame.shape[0] - 30),
                    cv.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )

        e2 = cv.getTickCount()
        proc_time_frame_ms = ((e2 - e1) / cv.getTickFrequency()) * 1000

        ret, buffer = cv.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        i += 1

    fresh.release()
    cv.destroyAllWindows()


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


@app.route("/stop")
def stop():
    logging.info("Stopping the application")
    ip_address = request.remote_addr
    logging.info(f"Request received from IP address: {ip_address}")
    print(f"Request received from IP address: {ip_address}")
    stop_server()
    return "Server shutting down..."


def stop_server():
    os._exit(0)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Flask app on port {port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
