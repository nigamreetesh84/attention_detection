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

import cv2
import numpy as np
from numpy import linalg as LA
from Utils import resize


EYES_LMS_NUMS = [33, 133, 160, 144, 158, 153, 362, 263, 385, 380, 387, 373]
LEFT_IRIS_NUM = 468
RIGHT_IRIS_NUM = 473


class EyeDetector:

    def __init__(self, show_processing: bool = False):

        self.show_processing = show_processing

    @staticmethod
    def _calc_EAR_eye(eye_pts):
        
        ear_eye = (
            LA.norm(eye_pts[2] - eye_pts[3]) + LA.norm(eye_pts[4] - eye_pts[5])
        ) / (2 * LA.norm(eye_pts[0] - eye_pts[1]))
        """
        EAR is computed as the mean of two measures of eye opening (see mediapipe face keypoints for the eye)
        divided by the eye lenght
        """
        return ear_eye

    def show_eye_keypoints(self, color_frame, landmarks, frame_size):
        

        cv2.circle(
            color_frame,
            (landmarks[LEFT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
            3,
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.circle(
            color_frame,
            (landmarks[RIGHT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
            3,
            (255, 255, 255),
            cv2.FILLED,
        )

        for n in EYES_LMS_NUMS:
            x = int(landmarks[n, 0] * frame_size[0])
            y = int(landmarks[n, 1] * frame_size[1])
            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
        return

    def get_EAR(self, frame, landmarks):
        

        # numpy array for storing the keypoints positions of the left and right eyes
        eye_pts_l = np.zeros(shape=(6, 2))
        eye_pts_r = eye_pts_l.copy()

        # get the face mesh keypoints
        for i in range(len(EYES_LMS_NUMS) // 2):
            # array of x,y coordinates for the left eye reference point
            eye_pts_l[i] = landmarks[EYES_LMS_NUMS[i], :2]
            # array of x,y coordinates for the right eye reference point
            eye_pts_r[i] = landmarks[EYES_LMS_NUMS[i + 6], :2]

        ear_left = self._calc_EAR_eye(eye_pts_l)  # computing the left eye EAR score
        ear_right = self._calc_EAR_eye(eye_pts_r)  # computing the right eye EAR score

        # computing the average EAR score
        ear_avg = (ear_left + ear_right) / 2

        return ear_avg

    @staticmethod
    def _calc_1eye_score(landmarks, eye_lms_nums, eye_iris_num, frame_size, frame):
        """Gets each eye score and its picture."""
        iris = landmarks[eye_iris_num, :2]

        eye_x_min = landmarks[eye_lms_nums, 0].min()
        eye_y_min = landmarks[eye_lms_nums, 1].min()
        eye_x_max = landmarks[eye_lms_nums, 0].max()
        eye_y_max = landmarks[eye_lms_nums, 1].max()

        eye_center = np.array(
            ((eye_x_min + eye_x_max) / 2, (eye_y_min + eye_y_max) / 2)
        )

        eye_gaze_score = LA.norm(iris - eye_center) / eye_center[0]

        eye_x_min_frame = int(eye_x_min * frame_size[0])
        eye_y_min_frame = int(eye_y_min * frame_size[1])
        eye_x_max_frame = int(eye_x_max * frame_size[0])
        eye_y_max_frame = int(eye_y_max * frame_size[1])

        eye = frame[eye_y_min_frame:eye_y_max_frame, eye_x_min_frame:eye_x_max_frame]

        return eye_gaze_score, eye

    def get_Gaze_Score(self, frame, landmarks, frame_size):


        left_gaze_score, left_eye = self._calc_1eye_score(
            landmarks, EYES_LMS_NUMS[:6], LEFT_IRIS_NUM, frame_size, frame
        )
        right_gaze_score, right_eye = self._calc_1eye_score(
            landmarks, EYES_LMS_NUMS[6:], RIGHT_IRIS_NUM, frame_size, frame
        )

        # if show_processing is True, shows the eyes ROI, eye center, pupil center and line distance

        # computes the average gaze score for the 2 eyes
        avg_gaze_score = (left_gaze_score + right_gaze_score) / 2

        if self.show_processing and (left_eye is not None) and (right_eye is not None):
            left_eye = resize(left_eye, 1000)
            right_eye = resize(right_eye, 1000)
            cv2.imshow("left eye", left_eye)
            cv2.imshow("right eye", right_eye)

        return avg_gaze_score


import numpy as np


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Debugger(metaclass=Singleton):
    
    def set_debug(self, debug):
        
        self.debug = debug

    def toggle(self):
        
        self.debug = not self.debug

    def get_debug(self):
        """
        This method is used to get the value of the "debug" attribute.

        Returns
        -------

        """
        return self.debug


DEBUG = Debugger()
DEBUG.set_debug(False)


class PCF:
    def __init__(
        self,
        near=1,
        far=10000,
        frame_height=1920,
        frame_width=1080,
        fy=1074.520446598223,
    ):

        
        self.near = near
        self.far = far
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.fy = fy

        fov_y = 2 * np.arctan(frame_height / (2 * fy))
        
        # kDegreesToRadians = np.pi / 180.0 # never used
        height_at_near = 2 * near * np.tan(0.5 * fov_y)
        width_at_near = frame_width * height_at_near / frame_height
        # print(height_at_near)
        
        self.fov_y = fov_y
        self.left = -0.5 * width_at_near
        self.right = 0.5 * width_at_near
        self.bottom = -0.5 * height_at_near
        self.top = 0.5 * height_at_near


canonical_metric_landmarks = np.array(
    [
        0.000000,
        -3.406404,
        5.979507,
        0.499977,
        0.652534,
        0.000000,
        -1.126865,
        7.475604,
        0.500026,
        0.547487,
        0.000000,
        -2.089024,
        6.058267,
        0.499974,
        0.602372,
        -0.463928,
        0.955357,
        6.633583,
        0.482113,
        0.471979,
        0.000000,
        -0.463170,
        7.586580,
        0.500151,
        0.527156,
        0.000000,
        0.365669,
        7.242870,
        0.499910,
        0.498253,
        0.000000,
        2.473255,
        5.788627,
        0.499523,
        0.401062,
        -4.253081,
        2.577646,
        3.279702,
        0.289712,
        0.380764,
        0.000000,
        4.019042,
        5.284764,
        0.499955,
        0.312398,
        0.000000,
        4.885979,
        5.385258,
        0.499987,
        0.269919,
        0.000000,
        8.261778,
        4.481535,
        0.500023,
        0.107050,
        0.000000,
        -3.706811,
        5.864924,
        0.500023,
        0.666234,
        0.000000,
        -3.918301,
        5.569430,
        0.500016,
        0.679224,
        0.000000,
        -3.994436,
        5.219482,
        0.500023,
        0.692348,
        0.000000,
        -4.542400,
        5.404754,
        0.499977,
        0.695278,
        0.000000,
        -4.745577,
        5.529457,
        0.499977,
        0.705934,
        0.000000,
        -5.019567,
        5.601448,
        0.499977,
        0.719385,
        0.000000,
        -5.365123,
        5.535441,
        0.499977,
        0.737019,
        0.000000,
        -6.149624,
        5.071372,
        0.499968,
        0.781371,
        0.000000,
        -1.501095,
        7.112196,
        0.499816,
        0.562981,
        -0.416106,
        -1.466449,
        6.447657,
        0.473773,
        0.573910,
        -7.087960,
        5.434801,
        0.099620,
        0.104907,
        0.254141,
        -2.628639,
        2.035898,
        3.848121,
        0.365930,
        0.409576,
        -3.198363,
        1.985815,
        3.796952,
        0.338758,
        0.413025,
        -3.775151,
        2.039402,
        3.646194,
        0.311120,
        0.409460,
        -4.465819,
        2.422950,
        3.155168,
        0.274658,
        0.389131,
        -2.164289,
        2.189867,
        3.851822,
        0.393362,
        0.403706,
        -3.208229,
        3.223926,
        4.115822,
        0.345234,
        0.344011,
        -2.673803,
        3.205337,
        4.092203,
        0.370094,
        0.346076,
        -3.745193,
        3.165286,
        3.972409,
        0.319322,
        0.347265,
        -4.161018,
        3.059069,
        3.719554,
        0.297903,
        0.353591,
        -5.062006,
        1.934418,
        2.776093,
        0.247792,
        0.410810,
        -2.266659,
        -7.425768,
        4.389812,
        0.396889,
        0.842755,
        -4.445859,
        2.663991,
        3.173422,
        0.280098,
        0.375600,
        -7.214530,
        2.263009,
        0.073150,
        0.106310,
        0.399956,
        -5.799793,
        2.349546,
        2.204059,
        0.209925,
        0.391353,
        -2.844939,
        -0.720868,
        4.433130,
        0.355808,
        0.534406,
        -0.711452,
        -3.329355,
        5.877044,
        0.471751,
        0.650404,
        -0.606033,
        -3.924562,
        5.444923,
        0.474155,
        0.680192,
        -1.431615,
        -3.500953,
        5.496189,
        0.439785,
        0.657229,
        -1.914910,
        -3.803146,
        5.028930,
        0.414617,
        0.666541,
        -1.131043,
        -3.973937,
        5.189648,
        0.450374,
        0.680861,
        -1.563548,
        -4.082763,
        4.842263,
        0.428771,
        0.682691,
        -2.650112,
        -5.003649,
        4.188483,
        0.374971,
        0.727805,
        -0.427049,
        -1.094134,
        7.360529,
        0.486717,
        0.547629,
        -0.496396,
        -0.475659,
        7.440358,
        0.485301,
        0.527395,
        -5.253307,
        3.881582,
        3.363159,
        0.257765,
        0.314490,
        -1.718698,
        0.974609,
        4.558359,
        0.401223,
        0.455172,
        -1.608635,
        -0.942516,
        5.814193,
        0.429819,
        0.548615,
        -1.651267,
        -0.610868,
        5.581319,
        0.421352,
        0.533741,
        -4.765501,
        -0.701554,
        3.534632,
        0.276896,
        0.532057,
        -0.478306,
        0.295766,
        7.101013,
        0.483370,
        0.499587,
        -3.734964,
        4.508230,
        4.550454,
        0.337212,
        0.282883,
        -4.588603,
        4.302037,
        4.048484,
        0.296392,
        0.293243,
        -6.279331,
        6.615427,
        1.425850,
        0.169295,
        0.193814,
        -1.220941,
        4.142165,
        5.106035,
        0.447580,
        0.302610,
        -2.193489,
        3.100317,
        4.000575,
        0.392390,
        0.353888,
        -3.102642,
        -4.352984,
        4.095905,
        0.354490,
        0.696784,
        -6.719682,
        -4.788645,
        -1.745401,
        0.067305,
        0.730105,
        -1.193824,
        -1.306795,
        5.737747,
        0.442739,
        0.572826,
        -0.729766,
        -1.593712,
        5.833208,
        0.457098,
        0.584792,
        -2.456206,
        -4.342621,
        4.283884,
        0.381974,
        0.694711,
        -2.204823,
        -4.304508,
        4.162499,
        0.392389,
        0.694203,
        -4.985894,
        4.802461,
        3.751977,
        0.277076,
        0.271932,
        -1.592294,
        -1.257709,
        5.456949,
        0.422552,
        0.563233,
        -2.644548,
        4.524654,
        4.921559,
        0.385919,
        0.281364,
        -2.760292,
        5.100971,
        5.015990,
        0.383103,
        0.255840,
        -3.523964,
        8.005976,
        3.729163,
        0.331431,
        0.119714,
        -5.599763,
        5.715470,
        2.724259,
        0.229924,
        0.232003,
        -3.063932,
        6.566144,
        4.529981,
        0.364501,
        0.189114,
        -5.720968,
        4.254584,
        2.830852,
        0.229622,
        0.299541,
        -6.374393,
        4.785590,
        1.591691,
        0.173287,
        0.278748,
        -0.672728,
        -3.688016,
        5.737804,
        0.472879,
        0.666198,
        -1.262560,
        -3.787691,
        5.417779,
        0.446828,
        0.668527,
        -1.732553,
        -3.952767,
        5.000579,
        0.422762,
        0.673890,
        -1.043625,
        -1.464973,
        5.662455,
        0.445308,
        0.580066,
        -2.321234,
        -4.329069,
        4.258156,
        0.388103,
        0.693961,
        -2.056846,
        -4.477671,
        4.520883,
        0.403039,
        0.706540,
        -2.153084,
        -4.276322,
        4.038093,
        0.403629,
        0.693953,
        -0.946874,
        -1.035249,
        6.512274,
        0.460042,
        0.557139,
        -1.469132,
        -4.036351,
        4.604908,
        0.431158,
        0.692366,
        -1.024340,
        -3.989851,
        4.926693,
        0.452182,
        0.692366,
        -0.533422,
        -3.993222,
        5.138202,
        0.475387,
        0.692366,
        -0.769720,
        -6.095394,
        4.985883,
        0.465828,
        0.779190,
        -0.699606,
        -5.291850,
        5.448304,
        0.472329,
        0.736226,
        -0.669687,
        -4.949770,
        5.509612,
        0.473087,
        0.717857,
        -0.630947,
        -4.695101,
        5.449371,
        0.473122,
        0.704626,
        -0.583218,
        -4.517982,
        5.339869,
        0.473033,
        0.695278,
        -1.537170,
        -4.423206,
        4.745470,
        0.427942,
        0.695278,
        -1.615600,
        -4.475942,
        4.813632,
        0.426479,
        0.703540,
        -1.729053,
        -4.618680,
        4.854463,
        0.423162,
        0.711846,
        -1.838624,
        -4.828746,
        4.823737,
        0.418309,
        0.720063,
        -2.368250,
        -3.106237,
        4.868096,
        0.390095,
        0.639573,
        -7.542244,
        -1.049282,
        -2.431321,
        0.013954,
        0.560034,
        0.000000,
        -1.724003,
        6.601390,
        0.499914,
        0.580147,
        -1.826614,
        -4.399531,
        4.399021,
        0.413200,
        0.695400,
        -1.929558,
        -4.411831,
        4.497052,
        0.409626,
        0.701823,
        -0.597442,
        -2.013686,
        5.866456,
        0.468080,
        0.601535,
        -1.405627,
        -1.714196,
        5.241087,
        0.422729,
        0.585985,
        -0.662449,
        -1.819321,
        5.863759,
        0.463080,
        0.593784,
        -2.342340,
        0.572222,
        4.294303,
        0.372120,
        0.473414,
        -3.327324,
        0.104863,
        4.113860,
        0.334562,
        0.496073,
        -1.726175,
        -0.919165,
        5.273355,
        0.411671,
        0.546965,
        -5.133204,
        7.485602,
        2.660442,
        0.242176,
        0.147676,
        -4.538641,
        6.319907,
        3.683424,
        0.290777,
        0.201446,
        -3.986562,
        5.109487,
        4.466315,
        0.327338,
        0.256527,
        -2.169681,
        -5.440433,
        4.455874,
        0.399510,
        0.748921,
        -1.395634,
        5.011963,
        5.316032,
        0.441728,
        0.261676,
        -1.619500,
        6.599217,
        4.921106,
        0.429765,
        0.187834,
        -1.891399,
        8.236377,
        4.274997,
        0.412198,
        0.108901,
        -4.195832,
        2.235205,
        3.375099,
        0.288955,
        0.398952,
        -5.733342,
        1.411738,
        2.431726,
        0.218937,
        0.435411,
        -1.859887,
        2.355757,
        3.843181,
        0.412782,
        0.398970,
        -4.988612,
        3.074654,
        3.083858,
        0.257135,
        0.355440,
        -1.303263,
        1.416453,
        4.831091,
        0.427685,
        0.437961,
        -1.305757,
        -0.672779,
        6.415959,
        0.448340,
        0.536936,
        -6.465170,
        0.937119,
        1.689873,
        0.178560,
        0.457554,
        -5.258659,
        0.945811,
        2.974312,
        0.247308,
        0.457194,
        -4.432338,
        0.722096,
        3.522615,
        0.286267,
        0.467675,
        -3.300681,
        0.861641,
        3.872784,
        0.332828,
        0.460712,
        -2.430178,
        1.131492,
        4.039035,
        0.368756,
        0.447207,
        -1.820731,
        1.467954,
        4.224124,
        0.398964,
        0.432655,
        -0.563221,
        2.307693,
        5.566789,
        0.476410,
        0.405806,
        -6.338145,
        -0.529279,
        1.881175,
        0.189241,
        0.523924,
        -5.587698,
        3.208071,
        2.687839,
        0.228962,
        0.348951,
        -0.242624,
        -1.462857,
        7.071491,
        0.490726,
        0.562401,
        -1.611251,
        0.339326,
        4.895421,
        0.404670,
        0.485133,
        -7.743095,
        2.364999,
        -2.005167,
        0.019469,
        0.401564,
        -1.391142,
        1.851048,
        4.448999,
        0.426243,
        0.420431,
        -1.785794,
        -0.978284,
        4.850470,
        0.396993,
        0.548797,
        -4.670959,
        2.664461,
        3.084075,
        0.266470,
        0.376977,
        -1.333970,
        -0.283761,
        6.097047,
        0.439121,
        0.518958,
        -7.270895,
        -2.890917,
        -2.252455,
        0.032314,
        0.644357,
        -1.856432,
        2.585245,
        3.757904,
        0.419054,
        0.387155,
        -0.923388,
        0.073076,
        6.671944,
        0.462783,
        0.505747,
        -5.000589,
        -6.135128,
        1.892523,
        0.238979,
        0.779745,
        -5.085276,
        -7.178590,
        0.714711,
        0.198221,
        0.831938,
        -7.159291,
        -0.811820,
        -0.072044,
        0.107550,
        0.540755,
        -5.843051,
        -5.248023,
        0.924091,
        0.183610,
        0.740257,
        -6.847258,
        3.662916,
        0.724695,
        0.134410,
        0.333683,
        -2.412942,
        -8.258853,
        4.119213,
        0.385764,
        0.883154,
        -0.179909,
        -1.689864,
        6.573301,
        0.490967,
        0.579378,
        -2.103655,
        -0.163946,
        4.566119,
        0.382385,
        0.508573,
        -6.407571,
        2.236021,
        1.560843,
        0.174399,
        0.397671,
        -3.670075,
        2.360153,
        3.635230,
        0.318785,
        0.396235,
        -3.177186,
        2.294265,
        3.775704,
        0.343364,
        0.400597,
        -2.196121,
        -4.598322,
        4.479786,
        0.396100,
        0.710217,
        -6.234883,
        -1.944430,
        1.663542,
        0.187885,
        0.588538,
        -1.292924,
        -9.295920,
        4.094063,
        0.430987,
        0.944065,
        -3.210651,
        -8.533278,
        2.802001,
        0.318993,
        0.898285,
        -4.068926,
        -7.993109,
        1.925119,
        0.266248,
        0.869701,
        0.000000,
        6.545390,
        5.027311,
        0.500023,
        0.190576,
        0.000000,
        -9.403378,
        4.264492,
        0.499977,
        0.954453,
        -2.724032,
        2.315802,
        3.777151,
        0.366170,
        0.398822,
        -2.288460,
        2.398891,
        3.697603,
        0.393207,
        0.395537,
        -1.998311,
        2.496547,
        3.689148,
        0.410373,
        0.391080,
        -6.130040,
        3.399261,
        2.038516,
        0.194993,
        0.342102,
        -2.288460,
        2.886504,
        3.775031,
        0.388665,
        0.362284,
        -2.724032,
        2.961810,
        3.871767,
        0.365962,
        0.355971,
        -3.177186,
        2.964136,
        3.876973,
        0.343364,
        0.355357,
        -3.670075,
        2.927714,
        3.724325,
        0.318785,
        0.358340,
        -4.018389,
        2.857357,
        3.482983,
        0.301415,
        0.363156,
        -7.555811,
        4.106811,
        -0.991917,
        0.058133,
        0.319076,
        -4.018389,
        2.483695,
        3.440898,
        0.301415,
        0.387449,
        0.000000,
        -2.521945,
        5.932265,
        0.499988,
        0.618434,
        -1.776217,
        -2.683946,
        5.213116,
        0.415838,
        0.624196,
        -1.222237,
        -1.182444,
        5.952465,
        0.445682,
        0.566077,
        -0.731493,
        -2.536683,
        5.815343,
        0.465844,
        0.620641,
        0.000000,
        3.271027,
        5.236015,
        0.499923,
        0.351524,
        -4.135272,
        -6.996638,
        2.671970,
        0.288719,
        0.819946,
        -3.311811,
        -7.660815,
        3.382963,
        0.335279,
        0.852820,
        -1.313701,
        -8.639995,
        4.702456,
        0.440512,
        0.902419,
        -5.940524,
        -6.223629,
        -0.631468,
        0.128294,
        0.791941,
        -1.998311,
        2.743838,
        3.744030,
        0.408772,
        0.373894,
        -0.901447,
        1.236992,
        5.754256,
        0.455607,
        0.451801,
        0.000000,
        -8.765243,
        4.891441,
        0.499877,
        0.908990,
        -2.308977,
        -8.974196,
        3.609070,
        0.375437,
        0.924192,
        -6.954154,
        -2.439843,
        -0.131163,
        0.114210,
        0.615022,
        -1.098819,
        -4.458788,
        5.120727,
        0.448662,
        0.695278,
        -1.181124,
        -4.579996,
        5.189564,
        0.448020,
        0.704632,
        -1.255818,
        -4.787901,
        5.237051,
        0.447112,
        0.715808,
        -1.325085,
        -5.106507,
        5.205010,
        0.444832,
        0.730794,
        -1.546388,
        -5.819392,
        4.757893,
        0.430012,
        0.766809,
        -1.953754,
        -4.183892,
        4.431713,
        0.406787,
        0.685673,
        -2.117802,
        -4.137093,
        4.555096,
        0.400738,
        0.681069,
        -2.285339,
        -4.051196,
        4.582438,
        0.392400,
        0.677703,
        -2.850160,
        -3.665720,
        4.484994,
        0.367856,
        0.663919,
        -5.278538,
        -2.238942,
        2.861224,
        0.247923,
        0.601333,
        -0.946709,
        1.907628,
        5.196779,
        0.452770,
        0.420850,
        -1.314173,
        3.104912,
        4.231404,
        0.436392,
        0.359887,
        -1.780000,
        2.860000,
        3.881555,
        0.416164,
        0.368714,
        -1.845110,
        -4.098880,
        4.247264,
        0.413386,
        0.692366,
        -5.436187,
        -4.030482,
        2.109852,
        0.228018,
        0.683572,
        -0.766444,
        3.182131,
        4.861453,
        0.468268,
        0.352671,
        -1.938616,
        -6.614410,
        4.521085,
        0.411362,
        0.804327,
        0.000000,
        1.059413,
        6.774605,
        0.499989,
        0.469825,
        -0.516573,
        1.583572,
        6.148363,
        0.479154,
        0.442654,
        0.000000,
        1.728369,
        6.316750,
        0.499974,
        0.439637,
        -1.246815,
        0.230297,
        5.681036,
        0.432112,
        0.493589,
        0.000000,
        -7.942194,
        5.181173,
        0.499886,
        0.866917,
        0.000000,
        -6.991499,
        5.153478,
        0.499913,
        0.821729,
        -0.997827,
        -6.930921,
        4.979576,
        0.456549,
        0.819201,
        -3.288807,
        -5.382514,
        3.795752,
        0.344549,
        0.745439,
        -2.311631,
        -1.566237,
        4.590085,
        0.378909,
        0.574010,
        -2.680250,
        -6.111567,
        4.096152,
        0.374293,
        0.780185,
        -3.832928,
        -1.537326,
        4.137731,
        0.319688,
        0.570738,
        -2.961860,
        -2.274215,
        4.440943,
        0.357155,
        0.604270,
        -4.386901,
        -2.683286,
        3.643886,
        0.295284,
        0.621581,
        -1.217295,
        -7.834465,
        4.969286,
        0.447750,
        0.862477,
        -1.542374,
        -0.136843,
        5.201008,
        0.410986,
        0.508723,
        -3.878377,
        -6.041764,
        3.311079,
        0.313951,
        0.775308,
        -3.084037,
        -6.809842,
        3.814195,
        0.354128,
        0.812553,
        -3.747321,
        -4.503545,
        3.726453,
        0.324548,
        0.703993,
        -6.094129,
        -3.205991,
        1.473482,
        0.189096,
        0.646300,
        -4.588995,
        -4.728726,
        2.983221,
        0.279777,
        0.714658,
        -6.583231,
        -3.941269,
        0.070268,
        0.133823,
        0.682701,
        -3.492580,
        -3.195820,
        4.130198,
        0.336768,
        0.644733,
        -1.255543,
        0.802341,
        5.307551,
        0.429884,
        0.466522,
        -1.126122,
        -0.933602,
        6.538785,
        0.455528,
        0.548623,
        -1.443109,
        -1.142774,
        5.905127,
        0.437114,
        0.558896,
        -0.923043,
        -0.529042,
        7.003423,
        0.467288,
        0.529925,
        -1.755386,
        3.529117,
        4.327696,
        0.414712,
        0.335220,
        -2.632589,
        3.713828,
        4.364629,
        0.377046,
        0.322778,
        -3.388062,
        3.721976,
        4.309028,
        0.344108,
        0.320151,
        -4.075766,
        3.675413,
        4.076063,
        0.312876,
        0.322332,
        -4.622910,
        3.474691,
        3.646321,
        0.283526,
        0.333190,
        -5.171755,
        2.535753,
        2.670867,
        0.241246,
        0.382786,
        -7.297331,
        0.763172,
        -0.048769,
        0.102986,
        0.468763,
        -4.706828,
        1.651000,
        3.109532,
        0.267612,
        0.424560,
        -4.071712,
        1.476821,
        3.476944,
        0.297879,
        0.433176,
        -3.269817,
        1.470659,
        3.731945,
        0.333434,
        0.433878,
        -2.527572,
        1.617311,
        3.865444,
        0.366427,
        0.426116,
        -1.970894,
        1.858505,
        3.961782,
        0.396012,
        0.416696,
        -1.579543,
        2.097941,
        4.084996,
        0.420121,
        0.410228,
        -7.664182,
        0.673132,
        -2.435867,
        0.007561,
        0.480777,
        -1.397041,
        -1.340139,
        5.630378,
        0.432949,
        0.569518,
        -0.884838,
        0.658740,
        6.233232,
        0.458639,
        0.479089,
        -0.767097,
        -0.968035,
        7.077932,
        0.473466,
        0.545744,
        -0.460213,
        -1.334106,
        6.787447,
        0.476088,
        0.563830,
        -0.748618,
        -1.067994,
        6.798303,
        0.468472,
        0.555057,
        -1.236408,
        -1.585568,
        5.480490,
        0.433991,
        0.582362,
        -0.387306,
        -1.409990,
        6.957705,
        0.483518,
        0.562984,
        -0.319925,
        -1.607931,
        6.508676,
        0.482483,
        0.577849,
        -1.639633,
        2.556298,
        3.863736,
        0.426450,
        0.389799,
        -1.255645,
        2.467144,
        4.203800,
        0.438999,
        0.396495,
        -1.031362,
        2.382663,
        4.615849,
        0.450067,
        0.400434,
        -4.253081,
        2.772296,
        3.315305,
        0.289712,
        0.368253,
        -4.530000,
        2.910000,
        3.339685,
        0.276670,
        0.363373,
        0.463928,
        0.955357,
        6.633583,
        0.517862,
        0.471948,
        4.253081,
        2.577646,
        3.279702,
        0.710288,
        0.380764,
        0.416106,
        -1.466449,
        6.447657,
        0.526227,
        0.573910,
        7.087960,
        5.434801,
        0.099620,
        0.895093,
        0.254141,
        2.628639,
        2.035898,
        3.848121,
        0.634070,
        0.409576,
        3.198363,
        1.985815,
        3.796952,
        0.661242,
        0.413025,
        3.775151,
        2.039402,
        3.646194,
        0.688880,
        0.409460,
        4.465819,
        2.422950,
        3.155168,
        0.725342,
        0.389131,
        2.164289,
        2.189867,
        3.851822,
        0.606630,
        0.403705,
        3.208229,
        3.223926,
        4.115822,
        0.654766,
        0.344011,
        2.673803,
        3.205337,
        4.092203,
        0.629906,
        0.346076,
        3.745193,
        3.165286,
        3.972409,
        0.680678,
        0.347265,
        4.161018,
        3.059069,
        3.719554,
        0.702097,
        0.353591,
        5.062006,
        1.934418,
        2.776093,
        0.752212,
        0.410805,
        2.266659,
        -7.425768,
        4.389812,
        0.602918,
        0.842863,
        4.445859,
        2.663991,
        3.173422,
        0.719902,
        0.375600,
        7.214530,
        2.263009,
        0.073150,
        0.893693,
        0.399960,
        5.799793,
        2.349546,
        2.204059,
        0.790082,
        0.391354,
        2.844939,
        -0.720868,
        4.433130,
        0.643998,
        0.534488,
        0.711452,
        -3.329355,
        5.877044,
        0.528249,
        0.650404,
        0.606033,
        -3.924562,
        5.444923,
        0.525850,
        0.680191,
        1.431615,
        -3.500953,
        5.496189,
        0.560215,
        0.657229,
        1.914910,
        -3.803146,
        5.028930,
        0.585384,
        0.666541,
        1.131043,
        -3.973937,
        5.189648,
        0.549626,
        0.680861,
        1.563548,
        -4.082763,
        4.842263,
        0.571228,
        0.682692,
        2.650112,
        -5.003649,
        4.188483,
        0.624852,
        0.728099,
        0.427049,
        -1.094134,
        7.360529,
        0.513050,
        0.547282,
        0.496396,
        -0.475659,
        7.440358,
        0.515097,
        0.527252,
        5.253307,
        3.881582,
        3.363159,
        0.742247,
        0.314507,
        1.718698,
        0.974609,
        4.558359,
        0.598631,
        0.454979,
        1.608635,
        -0.942516,
        5.814193,
        0.570338,
        0.548575,
        1.651267,
        -0.610868,
        5.581319,
        0.578632,
        0.533623,
        4.765501,
        -0.701554,
        3.534632,
        0.723087,
        0.532054,
        0.478306,
        0.295766,
        7.101013,
        0.516446,
        0.499639,
        3.734964,
        4.508230,
        4.550454,
        0.662801,
        0.282918,
        4.588603,
        4.302037,
        4.048484,
        0.703624,
        0.293271,
        6.279331,
        6.615427,
        1.425850,
        0.830705,
        0.193814,
        1.220941,
        4.142165,
        5.106035,
        0.552386,
        0.302568,
        2.193489,
        3.100317,
        4.000575,
        0.607610,
        0.353888,
        3.102642,
        -4.352984,
        4.095905,
        0.645429,
        0.696707,
        6.719682,
        -4.788645,
        -1.745401,
        0.932695,
        0.730105,
        1.193824,
        -1.306795,
        5.737747,
        0.557261,
        0.572826,
        0.729766,
        -1.593712,
        5.833208,
        0.542902,
        0.584792,
        2.456206,
        -4.342621,
        4.283884,
        0.618026,
        0.694711,
        2.204823,
        -4.304508,
        4.162499,
        0.607591,
        0.694203,
        4.985894,
        4.802461,
        3.751977,
        0.722943,
        0.271963,
        1.592294,
        -1.257709,
        5.456949,
        0.577414,
        0.563167,
        2.644548,
        4.524654,
        4.921559,
        0.614083,
        0.281387,
        2.760292,
        5.100971,
        5.015990,
        0.616907,
        0.255886,
        3.523964,
        8.005976,
        3.729163,
        0.668509,
        0.119914,
        5.599763,
        5.715470,
        2.724259,
        0.770092,
        0.232021,
        3.063932,
        6.566144,
        4.529981,
        0.635536,
        0.189249,
        5.720968,
        4.254584,
        2.830852,
        0.770391,
        0.299556,
        6.374393,
        4.785590,
        1.591691,
        0.826722,
        0.278755,
        0.672728,
        -3.688016,
        5.737804,
        0.527121,
        0.666198,
        1.262560,
        -3.787691,
        5.417779,
        0.553172,
        0.668527,
        1.732553,
        -3.952767,
        5.000579,
        0.577238,
        0.673890,
        1.043625,
        -1.464973,
        5.662455,
        0.554692,
        0.580066,
        2.321234,
        -4.329069,
        4.258156,
        0.611897,
        0.693961,
        2.056846,
        -4.477671,
        4.520883,
        0.596961,
        0.706540,
        2.153084,
        -4.276322,
        4.038093,
        0.596371,
        0.693953,
        0.946874,
        -1.035249,
        6.512274,
        0.539958,
        0.557139,
        1.469132,
        -4.036351,
        4.604908,
        0.568842,
        0.692366,
        1.024340,
        -3.989851,
        4.926693,
        0.547818,
        0.692366,
        0.533422,
        -3.993222,
        5.138202,
        0.524613,
        0.692366,
        0.769720,
        -6.095394,
        4.985883,
        0.534090,
        0.779141,
        0.699606,
        -5.291850,
        5.448304,
        0.527671,
        0.736226,
        0.669687,
        -4.949770,
        5.509612,
        0.526913,
        0.717857,
        0.630947,
        -4.695101,
        5.449371,
        0.526878,
        0.704626,
        0.583218,
        -4.517982,
        5.339869,
        0.526967,
        0.695278,
        1.537170,
        -4.423206,
        4.745470,
        0.572058,
        0.695278,
        1.615600,
        -4.475942,
        4.813632,
        0.573521,
        0.703540,
        1.729053,
        -4.618680,
        4.854463,
        0.576838,
        0.711846,
        1.838624,
        -4.828746,
        4.823737,
        0.581691,
        0.720063,
        2.368250,
        -3.106237,
        4.868096,
        0.609945,
        0.639910,
        7.542244,
        -1.049282,
        -2.431321,
        0.986046,
        0.560034,
        1.826614,
        -4.399531,
        4.399021,
        0.586800,
        0.695400,
        1.929558,
        -4.411831,
        4.497052,
        0.590372,
        0.701823,
        0.597442,
        -2.013686,
        5.866456,
        0.531915,
        0.601537,
        1.405627,
        -1.714196,
        5.241087,
        0.577268,
        0.585935,
        0.662449,
        -1.819321,
        5.863759,
        0.536915,
        0.593786,
        2.342340,
        0.572222,
        4.294303,
        0.627543,
        0.473352,
        3.327324,
        0.104863,
        4.113860,
        0.665586,
        0.495951,
        1.726175,
        -0.919165,
        5.273355,
        0.588354,
        0.546862,
        5.133204,
        7.485602,
        2.660442,
        0.757824,
        0.147676,
        4.538641,
        6.319907,
        3.683424,
        0.709250,
        0.201508,
        3.986562,
        5.109487,
        4.466315,
        0.672684,
        0.256581,
        2.169681,
        -5.440433,
        4.455874,
        0.600409,
        0.749005,
        1.395634,
        5.011963,
        5.316032,
        0.558266,
        0.261672,
        1.619500,
        6.599217,
        4.921106,
        0.570304,
        0.187871,
        1.891399,
        8.236377,
        4.274997,
        0.588166,
        0.109044,
        4.195832,
        2.235205,
        3.375099,
        0.711045,
        0.398952,
        5.733342,
        1.411738,
        2.431726,
        0.781070,
        0.435405,
        1.859887,
        2.355757,
        3.843181,
        0.587247,
        0.398932,
        4.988612,
        3.074654,
        3.083858,
        0.742870,
        0.355446,
        1.303263,
        1.416453,
        4.831091,
        0.572156,
        0.437652,
        1.305757,
        -0.672779,
        6.415959,
        0.551868,
        0.536570,
        6.465170,
        0.937119,
        1.689873,
        0.821442,
        0.457556,
        5.258659,
        0.945811,
        2.974312,
        0.752702,
        0.457182,
        4.432338,
        0.722096,
        3.522615,
        0.713757,
        0.467627,
        3.300681,
        0.861641,
        3.872784,
        0.667113,
        0.460673,
        2.430178,
        1.131492,
        4.039035,
        0.631101,
        0.447154,
        1.820731,
        1.467954,
        4.224124,
        0.600862,
        0.432473,
        0.563221,
        2.307693,
        5.566789,
        0.523481,
        0.405627,
        6.338145,
        -0.529279,
        1.881175,
        0.810748,
        0.523926,
        5.587698,
        3.208071,
        2.687839,
        0.771046,
        0.348959,
        0.242624,
        -1.462857,
        7.071491,
        0.509127,
        0.562718,
        1.611251,
        0.339326,
        4.895421,
        0.595293,
        0.485024,
        7.743095,
        2.364999,
        -2.005167,
        0.980531,
        0.401564,
        1.391142,
        1.851048,
        4.448999,
        0.573500,
        0.420000,
        1.785794,
        -0.978284,
        4.850470,
        0.602995,
        0.548688,
        4.670959,
        2.664461,
        3.084075,
        0.733530,
        0.376977,
        1.333970,
        -0.283761,
        6.097047,
        0.560611,
        0.519017,
        7.270895,
        -2.890917,
        -2.252455,
        0.967686,
        0.644357,
        1.856432,
        2.585245,
        3.757904,
        0.580985,
        0.387160,
        0.923388,
        0.073076,
        6.671944,
        0.537728,
        0.505385,
        5.000589,
        -6.135128,
        1.892523,
        0.760966,
        0.779753,
        5.085276,
        -7.178590,
        0.714711,
        0.801779,
        0.831938,
        7.159291,
        -0.811820,
        -0.072044,
        0.892441,
        0.540761,
        5.843051,
        -5.248023,
        0.924091,
        0.816351,
        0.740260,
        6.847258,
        3.662916,
        0.724695,
        0.865595,
        0.333687,
        2.412942,
        -8.258853,
        4.119213,
        0.614074,
        0.883246,
        0.179909,
        -1.689864,
        6.573301,
        0.508953,
        0.579438,
        2.103655,
        -0.163946,
        4.566119,
        0.617942,
        0.508316,
        6.407571,
        2.236021,
        1.560843,
        0.825608,
        0.397675,
        3.670075,
        2.360153,
        3.635230,
        0.681215,
        0.396235,
        3.177186,
        2.294265,
        3.775704,
        0.656636,
        0.400597,
        2.196121,
        -4.598322,
        4.479786,
        0.603900,
        0.710217,
        6.234883,
        -1.944430,
        1.663542,
        0.812086,
        0.588539,
        1.292924,
        -9.295920,
        4.094063,
        0.568013,
        0.944565,
        3.210651,
        -8.533278,
        2.802001,
        0.681008,
        0.898285,
        4.068926,
        -7.993109,
        1.925119,
        0.733752,
        0.869701,
        2.724032,
        2.315802,
        3.777151,
        0.633830,
        0.398822,
        2.288460,
        2.398891,
        3.697603,
        0.606793,
        0.395537,
        1.998311,
        2.496547,
        3.689148,
        0.589660,
        0.391062,
        6.130040,
        3.399261,
        2.038516,
        0.805016,
        0.342108,
        2.288460,
        2.886504,
        3.775031,
        0.611335,
        0.362284,
        2.724032,
        2.961810,
        3.871767,
        0.634038,
        0.355971,
        3.177186,
        2.964136,
        3.876973,
        0.656636,
        0.355357,
        3.670075,
        2.927714,
        3.724325,
        0.681215,
        0.358340,
        4.018389,
        2.857357,
        3.482983,
        0.698585,
        0.363156,
        7.555811,
        4.106811,
        -0.991917,
        0.941867,
        0.319076,
        4.018389,
        2.483695,
        3.440898,
        0.698585,
        0.387449,
        1.776217,
        -2.683946,
        5.213116,
        0.584177,
        0.624107,
        1.222237,
        -1.182444,
        5.952465,
        0.554318,
        0.566077,
        0.731493,
        -2.536683,
        5.815343,
        0.534154,
        0.620640,
        4.135272,
        -6.996638,
        2.671970,
        0.711218,
        0.819975,
        3.311811,
        -7.660815,
        3.382963,
        0.664630,
        0.852871,
        1.313701,
        -8.639995,
        4.702456,
        0.559100,
        0.902632,
        5.940524,
        -6.223629,
        -0.631468,
        0.871706,
        0.791941,
        1.998311,
        2.743838,
        3.744030,
        0.591234,
        0.373894,
        0.901447,
        1.236992,
        5.754256,
        0.544341,
        0.451584,
        2.308977,
        -8.974196,
        3.609070,
        0.624563,
        0.924192,
        6.954154,
        -2.439843,
        -0.131163,
        0.885770,
        0.615029,
        1.098819,
        -4.458788,
        5.120727,
        0.551338,
        0.695278,
        1.181124,
        -4.579996,
        5.189564,
        0.551980,
        0.704632,
        1.255818,
        -4.787901,
        5.237051,
        0.552888,
        0.715808,
        1.325085,
        -5.106507,
        5.205010,
        0.555168,
        0.730794,
        1.546388,
        -5.819392,
        4.757893,
        0.569944,
        0.767035,
        1.953754,
        -4.183892,
        4.431713,
        0.593203,
        0.685676,
        2.117802,
        -4.137093,
        4.555096,
        0.599262,
        0.681069,
        2.285339,
        -4.051196,
        4.582438,
        0.607600,
        0.677703,
        2.850160,
        -3.665720,
        4.484994,
        0.631938,
        0.663500,
        5.278538,
        -2.238942,
        2.861224,
        0.752033,
        0.601315,
        0.946709,
        1.907628,
        5.196779,
        0.547226,
        0.420395,
        1.314173,
        3.104912,
        4.231404,
        0.563544,
        0.359828,
        1.780000,
        2.860000,
        3.881555,
        0.583841,
        0.368714,
        1.845110,
        -4.098880,
        4.247264,
        0.586614,
        0.692366,
        5.436187,
        -4.030482,
        2.109852,
        0.771915,
        0.683578,
        0.766444,
        3.182131,
        4.861453,
        0.531597,
        0.352483,
        1.938616,
        -6.614410,
        4.521085,
        0.588371,
        0.804441,
        0.516573,
        1.583572,
        6.148363,
        0.520797,
        0.442565,
        1.246815,
        0.230297,
        5.681036,
        0.567985,
        0.493479,
        0.997827,
        -6.930921,
        4.979576,
        0.543283,
        0.819255,
        3.288807,
        -5.382514,
        3.795752,
        0.655317,
        0.745515,
        2.311631,
        -1.566237,
        4.590085,
        0.621009,
        0.574018,
        2.680250,
        -6.111567,
        4.096152,
        0.625560,
        0.780312,
        3.832928,
        -1.537326,
        4.137731,
        0.680198,
        0.570719,
        2.961860,
        -2.274215,
        4.440943,
        0.642764,
        0.604338,
        4.386901,
        -2.683286,
        3.643886,
        0.704663,
        0.621530,
        1.217295,
        -7.834465,
        4.969286,
        0.552012,
        0.862592,
        1.542374,
        -0.136843,
        5.201008,
        0.589072,
        0.508637,
        3.878377,
        -6.041764,
        3.311079,
        0.685945,
        0.775357,
        3.084037,
        -6.809842,
        3.814195,
        0.645735,
        0.812640,
        3.747321,
        -4.503545,
        3.726453,
        0.675343,
        0.703978,
        6.094129,
        -3.205991,
        1.473482,
        0.810858,
        0.646305,
        4.588995,
        -4.728726,
        2.983221,
        0.720122,
        0.714667,
        6.583231,
        -3.941269,
        0.070268,
        0.866152,
        0.682705,
        3.492580,
        -3.195820,
        4.130198,
        0.663187,
        0.644597,
        1.255543,
        0.802341,
        5.307551,
        0.570082,
        0.466326,
        1.126122,
        -0.933602,
        6.538785,
        0.544562,
        0.548376,
        1.443109,
        -1.142774,
        5.905127,
        0.562759,
        0.558785,
        0.923043,
        -0.529042,
        7.003423,
        0.531987,
        0.530140,
        1.755386,
        3.529117,
        4.327696,
        0.585271,
        0.335177,
        2.632589,
        3.713828,
        4.364629,
        0.622953,
        0.322779,
        3.388062,
        3.721976,
        4.309028,
        0.655896,
        0.320163,
        4.075766,
        3.675413,
        4.076063,
        0.687132,
        0.322346,
        4.622910,
        3.474691,
        3.646321,
        0.716482,
        0.333201,
        5.171755,
        2.535753,
        2.670867,
        0.758757,
        0.382787,
        7.297331,
        0.763172,
        -0.048769,
        0.897013,
        0.468769,
        4.706828,
        1.651000,
        3.109532,
        0.732392,
        0.424547,
        4.071712,
        1.476821,
        3.476944,
        0.702114,
        0.433163,
        3.269817,
        1.470659,
        3.731945,
        0.666525,
        0.433866,
        2.527572,
        1.617311,
        3.865444,
        0.633505,
        0.426088,
        1.970894,
        1.858505,
        3.961782,
        0.603876,
        0.416587,
        1.579543,
        2.097941,
        4.084996,
        0.579658,
        0.409945,
        7.664182,
        0.673132,
        -2.435867,
        0.992440,
        0.480777,
        1.397041,
        -1.340139,
        5.630378,
        0.567192,
        0.569420,
        0.884838,
        0.658740,
        6.233232,
        0.541366,
        0.478899,
        0.767097,
        -0.968035,
        7.077932,
        0.526564,
        0.546118,
        0.460213,
        -1.334106,
        6.787447,
        0.523913,
        0.563830,
        0.748618,
        -1.067994,
        6.798303,
        0.531529,
        0.555057,
        1.236408,
        -1.585568,
        5.480490,
        0.566036,
        0.582329,
        0.387306,
        -1.409990,
        6.957705,
        0.516311,
        0.563054,
        0.319925,
        -1.607931,
        6.508676,
        0.517472,
        0.577877,
        1.639633,
        2.556298,
        3.863736,
        0.573595,
        0.389807,
        1.255645,
        2.467144,
        4.203800,
        0.560698,
        0.395332,
        1.031362,
        2.382663,
        4.615849,
        0.549756,
        0.399751,
        4.253081,
        2.772296,
        3.315305,
        0.710288,
        0.368253,
        4.530000,
        2.910000,
        3.339685,
        0.723330,
        0.363373,
    ]
)
canonical_metric_landmarks = np.reshape(
    canonical_metric_landmarks, (canonical_metric_landmarks.shape[0] // 5, 5)
).T
canonical_metric_landmarks = canonical_metric_landmarks[:3, :]

procrustes_landmark_basis = [
    (4, 0.070909939706326),
    (6, 0.032100144773722),
    (10, 0.008446550928056),
    (33, 0.058724168688059),
    (54, 0.007667080033571),
    (67, 0.009078059345484),
    (117, 0.009791937656701),
    (119, 0.014565368182957),
    (121, 0.018591361120343),
    (127, 0.005197994410992),
    (129, 0.120625205338001),
    (132, 0.005560018587857),
    (133, 0.05328618362546),
    (136, 0.066890455782413),
    (143, 0.014816547743976),
    (147, 0.014262833632529),
    (198, 0.025462191551924),
    (205, 0.047252278774977),
    (263, 0.058724168688059),
    (284, 0.007667080033571),
    (297, 0.009078059345484),
    (346, 0.009791937656701),
    (348, 0.014565368182957),
    (350, 0.018591361120343),
    (356, 0.005197994410992),
    (358, 0.120625205338001),
    (361, 0.005560018587857),
    (362, 0.05328618362546),
    (365, 0.066890455782413),
    (372, 0.014816547743976),
    (376, 0.014262833632529),
    (420, 0.025462191551924),
    (425, 0.047252278774977),
]
landmark_weights = np.zeros((canonical_metric_landmarks.shape[1],))
for idx, weight in procrustes_landmark_basis:
    landmark_weights[idx] = weight


def log(name, f):
    
    if DEBUG.get_debug():
        print(f"{name} logged:", f)
        print()


def cpp_compare(name, np_matrix):
    
    if DEBUG.get_debug():
        # reorder cpp matrix as memory alignment is not correct
        cpp_matrix = np.load(f"{name}_cpp.npy")
        rows, cols = cpp_matrix.shape
        cpp_matrix = np.split(np.reshape(cpp_matrix, -1), cols)
        cpp_matrix = np.stack(cpp_matrix, 1)

        print(f"{name}:", np.sum(np.abs(cpp_matrix - np_matrix[:rows, :cols]) ** 2))
        print()


def get_metric_landmarks(screen_landmarks, pcf):
    
    screen_landmarks = project_xy(screen_landmarks, pcf)
    depth_offset = np.mean(screen_landmarks[2, :])
    
    intermediate_landmarks = screen_landmarks.copy()
    intermediate_landmarks = change_handedness(intermediate_landmarks)
    first_iteration_scale = estimate_scale(intermediate_landmarks)

    intermediate_landmarks = screen_landmarks.copy()
    intermediate_landmarks = move_and_rescale_z(
        pcf, depth_offset, first_iteration_scale, intermediate_landmarks
    )
    intermediate_landmarks = unproject_xy(pcf, intermediate_landmarks)
    intermediate_landmarks = change_handedness(intermediate_landmarks)
    second_iteration_scale = estimate_scale(intermediate_landmarks)

    metric_landmarks = screen_landmarks.copy()
    total_scale = first_iteration_scale * second_iteration_scale
    metric_landmarks = move_and_rescale_z(
        pcf, depth_offset, total_scale, metric_landmarks
    )
    metric_landmarks = unproject_xy(pcf, metric_landmarks)
    metric_landmarks = change_handedness(metric_landmarks)

    pose_transform_mat = solve_weighted_orthogonal_problem(
        canonical_metric_landmarks, metric_landmarks, landmark_weights
    )
    cpp_compare("pose_transform_mat", pose_transform_mat)

    inv_pose_transform_mat = np.linalg.inv(pose_transform_mat)
    inv_pose_rotation = inv_pose_transform_mat[:3, :3]
    inv_pose_translation = inv_pose_transform_mat[:3, 3]

    metric_landmarks = (
        inv_pose_rotation @ metric_landmarks + inv_pose_translation[:, None]
    )

    return metric_landmarks, pose_transform_mat


def project_xy(landmarks, pcf):
    x_scale = pcf.right - pcf.left
    y_scale = pcf.top - pcf.bottom
    x_translation = pcf.left
    y_translation = pcf.bottom

    landmarks[1, :] = 1.0 - landmarks[1, :]

    landmarks = landmarks * np.array([[x_scale, y_scale, x_scale]]).T
    landmarks = landmarks + np.array([[x_translation, y_translation, 0]]).T

    return landmarks


def change_handedness(landmarks):
    landmarks[2, :] *= -1.0

    return landmarks


def move_and_rescale_z(pcf, depth_offset, scale, landmarks):

    landmarks[2, :] = (landmarks[2, :] - depth_offset + pcf.near) / scale

    return landmarks


def unproject_xy(pcf, landmarks):
    landmarks[0, :] = landmarks[0, :] * landmarks[2, :] / pcf.near
    landmarks[1, :] = landmarks[1, :] * landmarks[2, :] / pcf.near

    return landmarks


def estimate_scale(landmarks):
    transform_mat = solve_weighted_orthogonal_problem(
        canonical_metric_landmarks, landmarks, landmark_weights
    )

    return np.linalg.norm(transform_mat[:, 0])


def extract_square_root(point_weights):

    return np.sqrt(point_weights)


def solve_weighted_orthogonal_problem(source_points, target_points, point_weights):
    sqrt_weights = extract_square_root(point_weights)
    transform_mat = internal_solve_weighted_orthogonal_problem(
        source_points, target_points, sqrt_weights
    )
    return transform_mat


def internal_solve_weighted_orthogonal_problem(sources, targets, sqrt_weights):

    cpp_compare("sources", sources)
    cpp_compare("targets", targets)

    # tranposed(A_w).
    weighted_sources = sources * sqrt_weights[None, :]
    # tranposed(B_w).
    weighted_targets = targets[:, :468] * sqrt_weights[None, :]

    cpp_compare("weighted_sources", weighted_sources)
    cpp_compare("weighted_targets", weighted_targets)

    # w = tranposed(j_w) j_w.
    total_weight = np.sum(sqrt_weights * sqrt_weights)
    log("total_weight", total_weight)


    twice_weighted_sources = weighted_sources * sqrt_weights[None, :]
    source_center_of_mass = np.sum(twice_weighted_sources, axis=1) / total_weight
    log("source_center_of_mass", source_center_of_mass)

    # tranposed((I - C) A_w) = tranposed(A_w) (I - C) =
    # tranposed(A_w) - tranposed(A_w) C = tranposed(A_w) - c_w tranposed(j_w).
    centered_weighted_sources = weighted_sources - np.matmul(
        source_center_of_mass[:, None], sqrt_weights[None, :]
    )
    cpp_compare("centered_weighted_sources", centered_weighted_sources)

    design_matrix = np.matmul(weighted_targets, centered_weighted_sources.T)
    cpp_compare("design_matrix", design_matrix)
    log("design_matrix_norm", np.linalg.norm(design_matrix))

    rotation = compute_optimal_rotation(design_matrix)

    scale = compute_optimal_scale(
        centered_weighted_sources, weighted_sources, weighted_targets, rotation
    )
    log("scale", scale)

    rotation_and_scale = scale * rotation

    pointwise_diffs = weighted_targets - np.matmul(rotation_and_scale, weighted_sources)
    cpp_compare("pointwise_diffs", pointwise_diffs)

    weighted_pointwise_diffs = pointwise_diffs * sqrt_weights[None, :]
    cpp_compare("weighted_pointwise_diffs", weighted_pointwise_diffs)

    translation = np.sum(weighted_pointwise_diffs, axis=1) / total_weight
    log("translation", translation)

    transform_mat = combine_transform_matrix(rotation_and_scale, translation)
    cpp_compare("transform_mat", transform_mat)

    return transform_mat


def compute_optimal_rotation(design_matrix):
    if np.linalg.norm(design_matrix) < 1e-9:
        print("Design matrix norm is too small!")

    u, _, vh = np.linalg.svd(design_matrix, full_matrices=True)

    postrotation = u
    prerotation = vh

    if np.linalg.det(postrotation) * np.linalg.det(prerotation) < 0:
        postrotation[:, 2] = -1 * postrotation[:, 2]

    cpp_compare("postrotation", postrotation)
    cpp_compare("prerotation", prerotation)

    rotation = np.matmul(postrotation, prerotation)

    cpp_compare("rotation", rotation)

    return rotation


def compute_optimal_scale(
    centered_weighted_sources, weighted_sources, weighted_targets, rotation
):
    rotated_centered_weighted_sources = np.matmul(rotation, centered_weighted_sources)

    numerator = np.sum(rotated_centered_weighted_sources * weighted_targets)
    denominator = np.sum(centered_weighted_sources * weighted_sources)

    if denominator < 1e-9:
        print("Scale expression denominator is too small!")
    if numerator / denominator < 1e-9:
        print("Scale is too small!")

    return numerator / denominator


def combine_transform_matrix(r_and_s, t):
    result = np.eye(4)
    result[:3, :3] = r_and_s
    result[:3, 3] = t
    return result


import cv2
import numpy as np

from face_geometry import *
from Utils import rotationMatrixToEulerAngles, draw_pose_info


JAW_LMS_NUMS = [61, 291, 199]


def _rmat2euler(rmat):
    rtr = np.transpose(rmat)
    r_identity = np.matmul(rtr, rmat)

    I = np.identity(3, dtype=rmat.dtype)
    if np.linalg.norm(r_identity - I) < 1e-6:
        sy = (rmat[:2, 0] ** 2).sum() ** 0.5
        singular = sy < 1e-6

        if not singular:  # check if it's a gimbal lock situation
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])

        else:  # if in gimbal lock, use different formula for yaw, pitch roll
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0

        if x > 0:
            x = np.pi - x
        else:
            x = -(np.pi + x)

        if z > 0:
            z = np.pi - z
        else:
            z = -(np.pi + z)

        return (np.array([x, y, z]) * 180.0 / np.pi).round(2)
    else:
        print("Isn't rotation matrix")


class HeadPoseEstimator:

    def __init__(self, camera_matrix=None, dist_coeffs=None, show_axis: bool = False):

        self.show_axis = show_axis
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.focal_length = None

        self.pcf_calculated = False

        self.model_lms_ids = self._get_model_lms_ids()

        self.NOSE_AXES_POINTS = np.array(
            [[7, 0, 10], [0, 7, 6], [0, 0, 14]], dtype=float
        )

    @staticmethod
    def _get_model_lms_ids():
        model_lms_ids = JAW_LMS_NUMS + [key for key, _ in procrustes_landmark_basis]
        model_lms_ids.sort()

        return model_lms_ids

    def get_pose(self, frame, landmarks, frame_size):

        rvec = None
        tvec = None
        model_img_lms = None
        eulers = None
        metric_lms = None

        if not self.pcf_calculated:
            self._get_camera_parameters(frame_size)

        model_img_lms = (
            np.clip(landmarks[self.model_lms_ids, :2], 0.0, 1.0) * frame_size
        )

        metric_lms = get_metric_landmarks(landmarks.T.copy(), self.pcf)[0].T

        model_metric_lms = metric_lms[self.model_lms_ids, :]

        (solve_pnp_success, rvec, tvec) = cv2.solvePnP(
            model_metric_lms,
            model_img_lms,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        tvec = tvec.round(2)

        if solve_pnp_success:
            rvec, tvec = cv2.solvePnPRefineVVS(
                model_metric_lms,
                model_img_lms,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
            )

            rvec1 = np.array([rvec[2, 0], rvec[0, 0], rvec[1, 0]]).reshape((3, 1))

            # cv2.Rodrigues: convert a rotation vector to a rotation matrix (also known as a Rodrigues rotation matrix)
            rmat, _ = cv2.Rodrigues(rvec1)

            eulers = _rmat2euler(rmat).reshape((-1, 1))



            self._draw_nose_axes(
                frame, rvec, tvec, model_img_lms
            )  # this will not show the lines

            return frame, eulers[0], eulers[1], eulers[2]

        else:
            return None, None, None, None

    def _draw_nose_axes(self, frame, rvec, tvec, model_img_lms):
        (nose_axes_point2D, _) = cv2.projectPoints(
            self.NOSE_AXES_POINTS, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        nose = tuple(model_img_lms[0, :2].astype(int))

        nose_x = tuple(nose_axes_point2D[0, 0].astype(int))
        nose_y = tuple(nose_axes_point2D[1, 0].astype(int))
        nose_z = tuple(nose_axes_point2D[2, 0].astype(int))

        cv2.line(frame, nose, nose_x, (255, 0, 0), 2)
        cv2.line(frame, nose, nose_y, (0, 255, 0), 2)
        cv2.line(frame, nose, nose_z, (0, 0, 255), 2)

    def _get_camera_parameters(self, frame_size):
        fr_w = frame_size[0]
        fr_h = frame_size[1]
        if self.camera_matrix is None:
            fr_center = (fr_w // 2, fr_h // 2)
            focal_length = fr_w
            self.camera_matrix = np.array(
                [
                    [focal_length, 0, fr_center[0]],
                    [0, focal_length, fr_center[1]],
                    [0, 0, 1],
                ],
                dtype="double",
            )
            self.focal_length = focal_length
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros((5, 1))

        self.pcf = PCF(frame_height=fr_h, frame_width=fr_w, fy=self.focal_length)

        self.pcf_calculated = True


import time


class AttentionScorer:

    def __init__(
        self,
        t_now,
        ear_thresh,
        gaze_thresh,
        perclos_thresh=0.2,
        roll_thresh=60,
        pitch_thresh=20,
        yaw_thresh=30,
        ear_time_thresh=4.0,
        gaze_time_thresh=2.0,
        pose_time_thresh=4.0,
        verbose=False,
    ):

        self.ear_thresh = ear_thresh
        self.gaze_thresh = gaze_thresh
        self.perclos_thresh = perclos_thresh
        self.roll_thresh = roll_thresh
        self.pitch_thresh = pitch_thresh
        self.yaw_thresh = yaw_thresh
        self.ear_time_thresh = ear_time_thresh
        self.gaze_time_thresh = gaze_time_thresh
        self.pose_time_thresh = pose_time_thresh
        self.verbose = verbose

        self.perclos_time_period = 60

        self.last_time_eye_opened = t_now
        self.last_time_looked_ahead = t_now
        self.last_time_attended = t_now
        self.closure_time = 0
        self.not_look_ahead_time = 0
        self.distracted_time = 0

        self.prev_time = t_now
        self.eye_closure_counter = 0


    def eval_scores(
        self, t_now, ear_score, gaze_score, head_roll, head_pitch, head_yaw
    ):
        # instantiating state of attention variables
        asleep = False
        looking_away = False
        distracted = False

        # Check if the EAR cumulative closure time has surpassed the threshold
        if self.closure_time >= self.ear_time_thresh:
            asleep = True
        else:
            asleep = False

        # Check if the gaze cumulative counter surpassed the threshold
        if self.not_look_ahead_time >= self.gaze_time_thresh:
            looking_away = True

        # Check if the pose cumulative counter surpassed the threshold
        if self.distracted_time >= self.pose_time_thresh:
            distracted = True

        # Update closure time based on the ear score
        if (ear_score is not None) and (ear_score <= self.ear_thresh):
            self.closure_time += t_now - self.last_time_eye_opened
            self.last_time_eye_opened = t_now
        elif ear_score is None or (
            ear_score is not None and ear_score > self.ear_thresh
        ):
            self.last_time_eye_opened = t_now
            self.closure_time = 0.0

        # Update not looking ahead time based on the gaze score
        if (gaze_score is not None) and (gaze_score > self.gaze_thresh):
            self.not_look_ahead_time = t_now - self.last_time_looked_ahead
        elif gaze_score is None or (
            gaze_score is not None and gaze_score <= self.gaze_thresh
        ):
            self.last_time_looked_ahead = t_now
            self.not_look_ahead_time = 0.0

        if (
            (head_roll is not None and abs(head_roll) > 30)
            or (head_pitch is not None and abs(head_pitch) > 30)
            or (head_yaw is not None and abs(head_yaw) > 30)
        ):
            self.distracted_time = t_now - self.last_time_attended
        else:
            self.last_time_attended = t_now
            self.distracted_time = 0.0

        if self.verbose:  # print additional info if verbose is True

            pass

        return asleep, looking_away, distracted

    def get_PERCLOS(self, t_now, fps, ear_score):

        delta = t_now - self.prev_time  # set delta timer
        tired = False  # set default value for the tired state of the driver

        all_frames_numbers_in_perclos_duration = int(self.perclos_time_period * fps)

        # if the ear_score is lower or equal than the threshold, increase the eye_closure_counter
        if (ear_score is not None) and (ear_score <= self.ear_thresh):
            self.eye_closure_counter += 1

        # compute the PERCLOS over a given time period
        perclos_score = (
            self.eye_closure_counter
        ) / all_frames_numbers_in_perclos_duration

        if (
            perclos_score >= self.perclos_thresh
        ):  # if the PERCLOS score is higher than a threshold, tired = True
            tired = True

        if (
            delta >= self.perclos_time_period
        ):  # at every end of the given time period, reset the counter and the timer
            self.eye_closure_counter = 0
            self.prev_time = t_now

        return tired, perclos_score

