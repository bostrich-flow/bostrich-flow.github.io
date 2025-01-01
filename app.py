from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/process_video', methods=['POST'])
def process_video():
    file = request.files['video']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    cap = cv2.VideoCapture(file_path)
    output_path = os.path.join(UPLOAD_FOLDER, 'processed_' + file.filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        out.write(processed_frame)

    cap.release()
    out.release()

    return send_file(output_path, as_attachment=True)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    frame_data = data['frame'].split(',')[1]
    frame = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    processed_frame = cv2.flip(frame, 1)

    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify(f"data:image/jpeg;base64,{processed_base64}")

if __name__ == '__main__':
    app.run(debug=True)
