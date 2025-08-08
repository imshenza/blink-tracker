import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from collections import deque
import random
import mediapipe as mp

# ---- Custom CSS ----
st.markdown("""
           <link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
    <style>
     div.stButton > button {
        background-color: #1E90FF;  /* Dodger Blue */
        color: white;
        font-size: 18px;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: bold;
        transition: background-color 0.3s ease;
        border: none;
        cursor: pointer;
        width: 100%;  /* optional: full width of container */
        max-width: 200px;  /* optional */
        display: block;
        margin: 0 auto;
    }
    div.stButton > button:hover {
        background-color: #0F75D8;
    }
     div.stButton > button:active {
        background-color: #0B5FAD;
    }
    h1{
        font-size:24px;
    }
    h1, h4 {
        font-family: 'Roboto', sans-serif;
    }
    .player-card {box-shadow: 0 2px 12px #44444430; transition:0.4s; margin-bottom:15px;}
    .player-card:hover {box-shadow: 0 4px 24px #44444480;}
    hr {margin-top:12px;margin-bottom:12px;}
    .stat-label {color: #555; font-weight: bold;}
    .blink-emoji {font-size:1.5em;}
    .msg-banner {background-color: #f3f8fd; border-radius:10px; padding:10px 22px; margin-bottom:10px; font-size:1.1em; color:#276bfd;}
    </style>
""", unsafe_allow_html=True)

# Motivational messages for long no-blink intervals
MESSAGES = [
    "Are you meditating or programming? 🧘",
    "Blinking is overrated anyway. 😜",
    "Are you trying out for a staring contest championship?",
    "Careful! You might dry out your eyeballs! 😮",
    "Legend says blinking slows you down...",
    "Steel Eyes Mode: ACTIVATED! 👁️🦾",
    "Unblinking productivity! ⚡",
    "Who needs coffee when you have eyes like that?! ☕👀",
    "NASA called: they're researching your focus.",
    "Stare harder, become wiser. 😆"
]

PLAYER_COLORS = [
    "#A16D68", "#8C7EB3", "#88A166", "#AF7B7B", "#92A8D1", "#F29C9A"
]

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=6,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    points = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in eye_indices]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear, points

class PlayerBlinkTracker:
    def __init__(self, player_id):
        self.player_id = player_id
        self.blink_count = 0
        self.last_blink_time = None
        self.blink_timestamps = []
        self.no_blink_start = time.time()
        self.longest_no_blink = 0
        self.blink_queue = deque()
        self.message = ""
        self.show_fun_message = False
        self.message_expire_time = 0
        self.eye_closed = False

    def update_ear(self, ear):
        EAR_THRESHOLD = 0.25
        if ear < EAR_THRESHOLD:
            if not self.eye_closed:
                self.register_blink()
                self.eye_closed = True
        else:
            self.eye_closed = False

    def register_blink(self):
        now = time.time()
        self.blink_count += 1
        self.blink_timestamps.append(now)
        if self.last_blink_time:
            no_blink_duration = now - self.last_blink_time
            if no_blink_duration > self.longest_no_blink:
                self.longest_no_blink = no_blink_duration
        self.last_blink_time = now
        self.no_blink_start = now
        self.blink_queue.append(now)
        while self.blink_queue and now - self.blink_queue[0] > 0.5:
            self.blink_queue.popleft()
        self.message_expire_time = 0
        self.update_messages()

    def update_no_blink(self):
        now = time.time()
        no_blink_duration = now - self.no_blink_start
        if no_blink_duration > 3:
            if now > self.message_expire_time:
                self.message = random.choice(MESSAGES)
                self.message_expire_time = now + 5
                self.show_fun_message = True
        else:
            self.message = ""
            self.message_expire_time = 0
            self.show_fun_message = False

    def update_messages(self):
        if len(self.blink_queue) > 3:
            self.message = "Rapid Fire! 🔥😵"
            self.message_expire_time = 0
            self.show_fun_message = False
        elif self.message_expire_time > time.time():
            pass
        else:
            self.message = ""
            self.message_expire_time = 0
            self.show_fun_message = False

    def average_blinks_per_minute(self):
        if len(self.blink_timestamps) < 2:
            return 0.0
        duration = self.blink_timestamps[-1] - self.blink_timestamps[0]
        if duration == 0:
            return 0.0
        bpm = (len(self.blink_timestamps) - 1) / (duration / 60)
        return round(bpm, 2)

def main():
    st.set_page_config(page_title="Blink Tracker", layout="wide")

    st.markdown("""
        <h1 style='text-align:center; color:#fff; margin-bottom:0.3em;'>Blink Tracker</h1>
        <hr style='border:2px solid #186eed'>
    """, unsafe_allow_html=True)

    for key, default in {
        'tracking': False,
        'players': dict(),
        'frame_count': 0,
        'blink_log': [],
        'start_time': None,
        'show_results': False,
        'just_stopped': False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.session_state.tracking:
            if st.button("■ Stop Tracking", key='toggle_btn'):
                st.session_state.tracking = False
                st.session_state.just_stopped = True
                st.success("Tracking stopped.")
        else:
            if st.button("▶ Start Tracking", key='toggle_btn'):
                st.session_state.tracking = True
                st.session_state.players = dict()
                st.session_state.frame_count = 0
                st.session_state.blink_log = []
                st.session_state.start_time = time.time()
                st.session_state.show_results = False
                st.session_state.just_stopped = False
                st.success("Tracking started! Please look at the camera.")


    with col3:
        if st.session_state.just_stopped:
            st.session_state.show_results = True
            st.session_state.just_stopped = False
        if st.session_state.show_results:
            if st.button("Hide Results"):
                st.session_state.show_results = False

    video_col, stats_col = st.columns([1.2, 1.7])
    video_col.markdown("#### Live Camera Feed 🎥")
    video_placeholder = video_col.empty()

    if st.session_state.tracking:
        run_tracker(video_placeholder, stats_col)
        st.session_state.show_results = False
    else:
        if st.session_state.show_results:
            show_summary(stats_col)
        else:
            video_col.info("Start tracking to see the live video feed and player stats here.")

def run_tracker(video_placeholder, stats_col):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        return

    stats_container = stats_col.empty()
    player_positions = {}

    try:
        while st.session_state.tracking:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(frame_rgb)
            faces = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    x_coords = [lm.x for lm in face_landmarks.landmark]
                    y_coords = [lm.y for lm in face_landmarks.landmark]
                    x_min = int(min(x_coords) * frame_width)
                    y_min = int(min(y_coords) * frame_height)
                    x_max = int(max(x_coords) * frame_width)
                    y_max = int(max(y_coords) * frame_height)
                    faces.append((x_min, y_min, x_max - x_min, y_max - y_min))
            faces = sorted(faces, key=lambda x: x[0])

            def center(face):
                return (face[0] + face[2] // 2, face[1] + face[3] // 2)

            DISTANCE_THRESHOLD = 100

            new_positions = {}
            assigned_pids = set()
            new_players = {}

            for face in faces:
                cx, cy = center(face)
                min_dist = float('inf')
                matched_pid = None
                for pid, pos in player_positions.items():
                    if pid in assigned_pids:
                        continue
                    dist = (cx - pos[0])**2 + (cy - pos[1])**2
                    if dist < min_dist:
                        min_dist = dist
                        matched_pid = pid
                if min_dist < DISTANCE_THRESHOLD**2 and matched_pid is not None:
                    new_positions[matched_pid] = (cx, cy)
                    assigned_pids.add(matched_pid)
                    new_players[matched_pid] = st.session_state.players[matched_pid]
                else:
                    new_id = 1
                    while new_id in assigned_pids or new_id in new_players:
                        new_id += 1
                    new_positions[new_id] = (cx, cy)
                    new_players[new_id] = PlayerBlinkTracker(new_id)
                    assigned_pids.add(new_id)

            st.session_state.players = new_players
            player_positions = new_positions

            eyes_present_curr = {}

            for i, (x, y, w, h) in enumerate(faces):
                cx, cy = center((x,y,w,h))
                player_id = None
                for pid, pos in player_positions.items():
                    if abs(pos[0] - cx) < 10 and abs(pos[1] - cy) < 10:
                        player_id = pid
                        break
                if player_id is None:
                    continue

                color = tuple(int(c) for c in hex_to_bgr(PLAYER_COLORS[(player_id-1) % len(PLAYER_COLORS)]))
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame_rgb, f"Player {player_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                face_landmarks = results.multi_face_landmarks[i]

                left_ear, left_eye_points = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDX, frame_width, frame_height)
                right_ear, right_eye_points = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_IDX, frame_width, frame_height)
                ear = (left_ear + right_ear) / 2.0

                pts_left = np.array(left_eye_points, np.int32)
                cv2.polylines(frame_rgb, [pts_left], isClosed=True, color=(0,255,0), thickness=2)

                pts_right = np.array(right_eye_points, np.int32)
                cv2.polylines(frame_rgb, [pts_right], isClosed=True, color=(0,255,0), thickness=2)

                st.session_state.players[player_id].update_ear(ear)
                eyes_present_curr[player_id] = ear >= 0.25

                st.session_state.players[player_id].update_no_blink()

            display_player_stats(stats_container)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            time.sleep(1/15)
    finally:
        cap.release()

def display_player_stats(stats_col):
    players = st.session_state.players
    if not players:
        stats_col.info("No players detected yet.")
        return

    stats_col.markdown("<strong style='font-size:1.15em; color:#186eed;'>Player Stats 🎯</strong>", unsafe_allow_html=True)
    player_ids = list(players.keys())
    max_per_row = 3

    for i in range(0, len(player_ids), max_per_row):
        row_ids = player_ids[i:i+max_per_row]
        cols = stats_col.columns(len(row_ids))

        for col, pid in zip(cols, row_ids):
            tracker = players[pid]
            color = PLAYER_COLORS[(pid-1) % len(PLAYER_COLORS)]
            col.markdown(
                f"""
                <div class="player-card" style="background-color:{color}; padding:16px; border-radius:10px; margin-bottom:10px; color:white;">
                    <div style="display:flex; align-items:center;">
                        <span style="font-size:1.3em; font-weight:bold;">Player {pid}</span>
                    </div>
                    <hr>
                    <div class="stat-label">Total Blinks:</div>
                    <div style="font-size:1.3em;">{tracker.blink_count} <span class="blink-emoji">👀</span></div>
                    <div class="stat-label">Average BPM:</div>
                    <div>{tracker.average_blinks_per_minute()}</div>
                    <div class="stat-label">Longest No Blink:</div>
                    <div>{int(tracker.longest_no_blink)} seconds</div>
                    {"<div class='msg-banner'>" + tracker.message + "</div>" if tracker.message else ""}
                </div>
                """,
                unsafe_allow_html=True
            )

def show_summary(stats_col):
    players = st.session_state.players
    blink_log = st.session_state.blink_log

    if not players:
        stats_col.write("No data to show.")
        return

    stats_col.markdown("<strong style='font-size:1.15em; color:#186eed;'>Session Summary Report 📊</strong>", unsafe_allow_html=True)

    min_blinks = min([tracker.blink_count for tracker in players.values()]) if players else 0
    winners = [pid for pid, tracker in players.items() if tracker.blink_count == min_blinks]

    for pid, tracker in players.items():
        color = PLAYER_COLORS[(pid-1) % len(PLAYER_COLORS)]
        winner_tag = ""
        if pid in winners:
            winner_tag = "<strong style='color:#FFD700;'> 👑 Winner!</strong>"
        stats_col.markdown(
            f"""
            <div class="player-card" style="background-color:{color}; padding:15px; border-radius:10px; margin-bottom:15px; color:white;">
                <h3 style="margin-bottom:5px;">Player {pid} {winner_tag}</h3>
                <ul>
                    <li><b>Total Blinks:</b> {tracker.blink_count}</li>
                    <li><b>Average Blinks/Min:</b> {tracker.average_blinks_per_minute()}</li>
                    <li><b>Longest No Blink:</b> {int(tracker.longest_no_blink)} seconds</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )

    if len(blink_log) == 0:
        stats_col.write("No blinks recorded in this session.")
        return

    start_time = blink_log[0][0]
    end_time = blink_log[-1][0]
    bins = np.arange(start_time, end_time + 10, 10)

    fig, ax = plt.subplots(figsize=(8, 4))

    for pid in players.keys():
        times = [t for (t, p) in blink_log if p == pid]
        counts, _ = np.histogram(times, bins)
        cumulative_counts = np.cumsum(counts)
        ax.plot(bins[:-1] - start_time, cumulative_counts, label=f"Player {pid}", color=PLAYER_COLORS[(pid-1) % len(PLAYER_COLORS)])

    ax.set_title("Cumulative Blinks Over Time")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Blinks")
    ax.legend()
    ax.grid(True)

    stats_col.pyplot(fig)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"blink_tracker_session_{timestamp_str}.csv"

    with open(filename, "w") as f:
        f.write("Player,Total Blinks,Average Blinks/Min,Longest No Blink (s)\n")
        for pid, tracker in players.items():
            f.write(f"{pid},{tracker.blink_count},{tracker.average_blinks_per_minute()},{int(tracker.longest_no_blink)}\n")
        f.write("\nBlink Log:\nTimestamp,Player\n")
        for ts, pid in blink_log:
            f.write(f"{datetime.datetime.fromtimestamp(ts)},{pid}\n")

    stats_col.success(f"Session stats saved to {filename}")

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

if __name__ == "__main__":
    main()
