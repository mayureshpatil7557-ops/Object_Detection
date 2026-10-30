"""
VisionAI - Flask Backend
Real-Time Object Detection & Tracking using YOLOv8 + OpenCV
"""

import cv2
import time
import threading
import os
import json
from flask import Flask, render_template, Response, request, jsonify, stream_with_context
from detector import Detector
from tracker import CentroidTracker

# ── Flask App ─────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# ── Global State ──────────────────────────────────────────
state = {
    'running':          False,
    'source':           'webcam',     # 'webcam' or 'file'
    'confidence':       0.50,
    'tracking':         True,
    'fps':              0,
    'objects':          0,
    'tracks':           0,
    'inference_ms':     0,
    'detections':       [],
    'class_counts':     {},
    'total_detected':   0,
    'seen_classes':     set(),
    'max_simultaneous': 0,
    'model_ready':      False,
    'model_status':     'Loading YOLOv8 model...',
}

cap        = None
cap_lock   = threading.Lock()
state_lock = threading.Lock()
detector   = None
tracker    = CentroidTracker(max_disappeared=25, max_distance=100)

# FPS tracking
_fps_times  = []
_last_frame  = None
_frame_lock  = threading.Lock()


# ── Model Init (background thread) ───────────────────────
def load_model():
    global detector
    try:
        with state_lock:
            state['model_status'] = 'Downloading YOLOv8n weights (~6 MB)...'
        detector = Detector(model_size='yolov8n')
        with state_lock:
            state['model_ready'] = True
            state['model_status'] = 'Model ready'
        print('[VisionAI] YOLOv8n model loaded ✓')
    except Exception as e:
        with state_lock:
            state['model_status'] = f'Model error: {str(e)}'
        print(f'[VisionAI] Model load error: {e}')

threading.Thread(target=load_model, daemon=True).start()


# ── Video Capture Thread ───────────────────────────────────
def capture_loop():
    global cap, _last_frame
    while True:
        with state_lock:
            running = state['running']
        if not running:
            time.sleep(0.05)
            continue

        with cap_lock:
            if cap is None or not cap.isOpened():
                time.sleep(0.05)
                continue
            ret, frame = cap.read()

        if not ret:
            # Loop video file
            with cap_lock:
                if cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.03)
            continue

        # Run detection
        t0 = time.perf_counter()
        try:
            with state_lock:
                conf = state['confidence']
                tracking_on = state['tracking']

            detections = detector.detect(frame, conf) if detector else []
            inf_ms = int((time.perf_counter() - t0) * 1000)

            # Tracking
            tracked = {}
            if tracking_on and detections:
                tracked = tracker.update(detections)
            elif not tracking_on:
                tracker.reset()

            # Draw
            annotated = draw_frame(frame, detections, tracked, tracking_on)

            # Update stats
            fps = compute_fps()
            class_counts = {}
            for d in detections:
                class_counts[d['label']] = class_counts.get(d['label'], 0) + 1

            det_list = [
                {
                    'label': d['label'],
                    'conf':  round(d['score'] * 100),
                    'color': d['color'],
                    'track_id': d.get('track_id'),
                }
                for d in detections[:12]
            ]

            with state_lock:
                state['fps']           = fps
                state['objects']       = len(detections)
                state['tracks']        = len(tracked)
                state['inference_ms']  = inf_ms
                state['detections']    = det_list
                state['class_counts']  = class_counts
                state['total_detected'] += len(detections)
                for d in detections:
                    state['seen_classes'].add(d['label'])
                if len(detections) > state['max_simultaneous']:
                    state['max_simultaneous'] = len(detections)

        except Exception as e:
            print(f'[VisionAI] Detection error: {e}')
            annotated = frame

        # Encode JPEG
        _, buf = cv2.imencode('.jpg', annotated,
                              [cv2.IMWRITE_JPEG_QUALITY, 82])
        with _frame_lock:
            _last_frame = buf.tobytes()


threading.Thread(target=capture_loop, daemon=True).start()


# ── Drawing ────────────────────────────────────────────────
def draw_frame(frame, detections, tracked, tracking_on):
    img = frame.copy()
    h, w = img.shape[:2]

    # Build centroid→id map
    id_map = {}
    if tracking_on:
        for tid, obj in tracked.items():
            cx = int(obj['centroid']['x'])
            cy = int(obj['centroid']['y'])
            id_map[(cx, cy)] = tid

    for d in detections:
        x1, y1, bw, bh = [int(v) for v in d['bbox']]
        x2, y2 = x1 + bw, y1 + bh
        color_hex = d['color']
        color_bgr = hex_to_bgr(color_hex)
        label = d['label']
        conf  = d['score']

        # Find track ID
        cx = int(x1 + bw / 2)
        cy = int(y1 + bh / 2)
        track_id = None
        if tracking_on:
            best_dist, best_id = 80, None
            for (mx, my), tid in id_map.items():
                dist = ((mx - cx)**2 + (my - cy)**2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            track_id = best_id
            d['track_id'] = track_id

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

        # Corner accents
        clen = 16
        thickness = 3
        for pts in [
            [(x1, y1+clen), (x1, y1), (x1+clen, y1)],
            [(x2-clen, y1), (x2, y1), (x2, y1+clen)],
            [(x1, y2-clen), (x1, y2), (x1+clen, y2)],
            [(x2-clen, y2), (x2, y2), (x2, y2-clen)],
        ]:
            for i in range(len(pts)-1):
                cv2.line(img, pts[i], pts[i+1], color_bgr, thickness)

        # Label text
        id_txt  = f'#{track_id} ' if track_id is not None else ''
        tag     = f'{id_txt}{label} {int(conf*100)}%'
        font    = cv2.FONT_HERSHEY_DUPLEX
        scale   = 0.52
        thick   = 1
        (tw, th), bl = cv2.getTextSize(tag, font, scale, thick)
        ty = y1 - 8 if y1 - 8 > th + 4 else y1 + th + 8
        # Background pill
        cv2.rectangle(img, (x1, ty - th - 5), (x1 + tw + 10, ty + 3),
                      color_bgr, cv2.FILLED)
        # Text
        lum = 0.299 * int(color_hex[1:3],16) + \
              0.587 * int(color_hex[3:5],16) + \
              0.114 * int(color_hex[5:7],16)
        txt_color = (20, 20, 20) if lum > 128 else (255, 255, 255)
        cv2.putText(img, tag, (x1 + 5, ty - 2), font, scale, txt_color, thick)

        # Centroid dot
        if tracking_on and track_id is not None:
            cv2.circle(img, (cx, cy), 4, color_bgr, -1)
            cv2.circle(img, (cx, cy), 6, color_bgr, 1)

    # Overlay HUD
    draw_hud(img, w, h)
    return img


def draw_hud(img, w, h):
    """Draw refined translucent glass HUD badge on top-left"""
    with state_lock:
        fps   = state['fps']
        objs  = state['objects']
        inf   = state['inference_ms']
    overlay = img.copy()
    cv2.rectangle(overlay, (12, 12), (180, 78), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(overlay, (12, 12), (180, 78), (220, 226, 235), 1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    font = cv2.FONT_HERSHEY_DUPLEX
    cobalt = (219, 99, 37)  # #2563eb in BGR
    slate  = (71, 85, 105)  # #475569 in BGR
    cv2.putText(img, f'FPS: {fps}',        (20, 32), font, 0.48, cobalt, 1)
    cv2.putText(img, f'Objects: {objs}',   (20, 50), font, 0.48, slate,  1)
    cv2.putText(img, f'Latency: {inf}ms',  (20, 68), font, 0.48, slate,  1)


# ── Helpers ────────────────────────────────────────────────
def hex_to_bgr(h):
    h = h.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)

def compute_fps():
    now = time.perf_counter()
    _fps_times.append(now)
    cutoff = now - 1.0
    while _fps_times and _fps_times[0] < cutoff:
        _fps_times.pop(0)
    return len(_fps_times)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Flask Routes ───────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with _frame_lock:
                frame = _last_frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS max
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def api_stats():
    with state_lock:
        return jsonify({
            'running':          state['running'],
            'model_ready':      state['model_ready'],
            'model_status':     state['model_status'],
            'fps':              state['fps'],
            'objects':          state['objects'],
            'tracks':           state['tracks'],
            'inference_ms':     state['inference_ms'],
            'detections':       state['detections'],
            'class_counts':     state['class_counts'],
            'total_detected':   state['total_detected'],
            'unique_classes':   len(state['seen_classes']),
            'max_simultaneous': state['max_simultaneous'],
        })


@app.route('/api/start', methods=['POST'])
def api_start():
    global cap
    data = request.get_json() or {}
    source = data.get('source', 'webcam')

    if not state['model_ready']:
        return jsonify({'ok': False, 'msg': 'Model not ready yet'}), 503

    with cap_lock:
        if cap is not None:
            cap.release()
        if source == 'webcam':
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            path = data.get('path', '')
            if not os.path.exists(path):
                return jsonify({'ok': False, 'msg': 'Video file not found'}), 404
            cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            cap = None
            return jsonify({'ok': False, 'msg': 'Could not open video source'}), 500

    tracker.reset()
    with state_lock:
        state['running']          = True
        state['source']           = source
        state['total_detected']   = 0
        state['seen_classes']     = set()
        state['max_simultaneous'] = 0

    return jsonify({'ok': True})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    global cap
    with state_lock:
        state['running'] = False

    with cap_lock:
        if cap is not None:
            cap.release()
            cap = None

    tracker.reset()
    with _frame_lock:
        global _last_frame
        _last_frame = None

    return jsonify({'ok': True})


@app.route('/api/settings', methods=['POST'])
def api_settings():
    data = request.get_json() or {}
    with state_lock:
        if 'confidence' in data:
            state['confidence'] = float(data['confidence'])
        if 'tracking' in data:
            state['tracking'] = bool(data['tracking'])
            if not state['tracking']:
                tracker.reset()
    return jsonify({'ok': True})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'video' not in request.files:
        return jsonify({'ok': False, 'msg': 'No file'}), 400
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'ok': False, 'msg': 'Invalid file type'}), 400

    filename = f"upload_{int(time.time())}_{file.filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    return jsonify({'ok': True, 'path': save_path})


# ── Run ────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 55)
    print('  VisionAI — Object Detection & Tracking')
    print('  http://127.0.0.1:5000')
    print('=' * 55)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
