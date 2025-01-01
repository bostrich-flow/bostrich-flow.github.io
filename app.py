from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
import os
import threading
from datetime import datetime
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import base64

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

progress = {"value": 0}  # Progress tracking

# Kalman Filter Class (As Is)
class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        return self.kalman.correct(measurement)

    def predict(self):
        prediction = self.kalman.predict()
        return prediction[0], prediction[1]

background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=10, detectShadows=False)

def apply_schlieren_effect(frame, background_frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_background, gray_frame)
    enhanced_diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_diff = clahe.apply(enhanced_diff)
    enhanced_diff = cv2.convertScaleAbs(enhanced_diff, alpha=1.5, beta=0)
    return enhanced_diff

def apply_colormap(image):
    colormap = plt.get_cmap('plasma')
    norm = Normalize(vmin=0, vmax=255)
    colormap_img = colormap(norm(image))
    colormap_img = (colormap_img[:, :, :3] * 255).astype(np.uint8)
    return colormap_img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    # Start processing in a separate thread
    thread = threading.Thread(target=process_video, args=(video_path,))
    thread.start()

    return jsonify({"message": "Video uploaded and processing started."}), 200

def process_video(video_path):
    global progress
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = []

    ret, background_frame = cap.read()
    if not ret:
        print("Error: Unable to read the video file.")
        return

    background_frame_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        schlieren_frame = apply_schlieren_effect(frame, background_frame)
        colormap_frame = apply_colormap(schlieren_frame)
        processed_frames.append(colormap_frame)

        frame_count += 1
        progress["value"] = int((frame_count / total_frames) * 100)

    cap.release()

    # Save processed video
    output_path = os.path.join(PROCESSED_FOLDER, f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    height, width, _ = processed_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for frame in processed_frames:
        out.write(frame)
    out.release()

    progress["value"] = 100
    print(f"Processing complete. Video saved at {output_path}.")

@app.route("/progress", methods=["GET"])
def get_progress():
    return jsonify(progress), 200

@app.route("/processed/<filename>")
def get_processed_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
