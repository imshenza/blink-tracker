import cv2
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
from collections import deque

# ------------------ Load Haar cascades ------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Colors for players (cycled if > len)
PLAYER_COLORS = [
    "#FF6F61",  # Red-ish
    "#6B5B95",  # Purple
    "#88B04B",  # Green
    "#F7CAC9",  # Pink
    "#92A8D1",  # Blue
    "#955251",  # Brown
]

# ------------------ Blink Tracker Classes ------------------
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
        while self.blink_queue and now - self.blink_queue[0] > 2:
            self.blink_queue.popleft()

        self.update_messages()

    def update_no_blink(self):
        now = time.time()
        no_blink_duration = now - self.no_blink_start
        if no_blink_duration >= 20:
            self.message = "Steel Eyes Mode Activated 👁️🦾"
        else:
            if len(self.blink_queue) <= 3:
                self.message = ""

    def update_messages(self):
        if len(self.blink_queue) > 3:
            self.message = "Rapid Fire! 🔥😵"
        else:
            self.message = ""

    def average_blinks_per_minute(self):
        if len(self.blink_timestamps) < 2:
            return 0.0
        duration = self.blink_timestamps[-1] - self.blink_timestamps[0]
        if duration == 0:
            return 0.0
        bpm = (len(self.blink_timestamps) - 1) / (duration / 60)
        return round(bpm, 2)

# ------------------ Main App ------------------
def main():
    st.set_page_config(page_title="👁️ Blink Tracker 3000", layout="wide")
    st.markdown(
        """
        <h1 style='text-align:center; color:#4B0082;'>👁️ Blink Tracker 3000</h1>
        <h4 style='text-align:center; color:#6A5ACD;'>Track your blinks live with fun messages! Supports multiple faces & players.</h4>
        <hr style='border:2px solid #6A5ACD'>
        """, unsafe_allow_html=True
    )
    
    # Initialize session state variables
    for key, default in {
        'tracking': False,
        'players': dict(),
        'frame_count': 0,
        'blink_log': [],
        'start_time': None,
        'show_results': False
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Controls area
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        start_stop_label = "Stop Tracking" if st.session_state.tracking else "Start Tracking"
        if st.button(start_stop_label, key='start_stop_btn'):
            if not st.session_state.tracking:
                # Starting Tracking - Reset
                st.session_state.tracking = True
                st.session_state.players = dict()
                st.session_state.frame_count = 0
                st.session_state.blink_log = []
                st.session_state.start_time = time.time()
                st.session_state.show_results = False
                st.success("Tracking started! Please look at the camera.")
            else:
                # Stopping Tracking
                st.session_state.tracking = False
                st.success("Tracking stopped.")

    with col2:
        if st.button("View Results", disabled=st.session_state.tracking or len(st.session_state.blink_log) == 0):
            st.session_state.show_results = True

    with col3:
        if st.session_state.show_results:
            if st.button("Hide Results"):
                st.session_state.show_results = False

    # Layout for video and player stats side by side
    video_col, stats_col = st.columns([3, 1])
    
    # Video placeholder
    video_placeholder = video_col.empty()
    
    if st.session_state.tracking:
        run_tracker(video_placeholder, stats_col)
        st.session_state.show_results = False
    else:
        # Show summary only if show_results is True
        if st.session_state.show_results:
            show_summary(stats_col)
        else:
            video_col.info("Start tracking to see the live video feed and stats here.")

# ------------------ Tracking Loop ------------------
def run_tracker(video_placeholder, stats_col):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        return

    eyes_present_prev = dict()

    # Create a placeholder container for player stats outside the loop
    stats_container = stats_col.empty()

    try:
        while st.session_state.tracking:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            faces = sorted(faces, key=lambda x: x[0])

            current_face_count = len(faces)
            existing_player_count = len(st.session_state.players)
            if current_face_count > existing_player_count:
                for i in range(existing_player_count + 1, current_face_count + 1):
                    st.session_state.players[i] = PlayerBlinkTracker(i)

            eyes_present_curr = dict()

            for i, (x, y, w, h) in enumerate(faces):
                player_id = i + 1
                face_gray = gray[y:y+h, x:x+w]
                face_color = frame[y:y+h, x:x+w]

                eyes = eye_cascade.detectMultiScale(face_gray)

                color = tuple(int(c) for c in hex_to_bgr(PLAYER_COLORS[(player_id-1) % len(PLAYER_COLORS)]))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, f"Player {player_id}", (x, y-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

                eyes_present_curr[player_id] = len(eyes) > 0

                if player_id in eyes_present_prev:
                    if eyes_present_prev[player_id] and not eyes_present_curr[player_id]:
                        st.session_state.players[player_id].register_blink()
                        st.session_state.blink_log.append((time.time(), player_id))

                st.session_state.players[player_id].update_no_blink()

            eyes_present_prev = eyes_present_curr

            # Update player stats inside the single container
            display_player_stats(stats_container)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")

            time.sleep(1/15)

    finally:
        cap.release()

def display_player_stats(stats_col):
    players = st.session_state.players
    if not players:
        stats_col.info("No players detected yet.")
        return

    stats_col.markdown("### Player Stats 🎯")
    for pid, tracker in players.items():
        color = PLAYER_COLORS[(pid-1) % len(PLAYER_COLORS)]
        with stats_col.container():
            stats_col.markdown(
                f"""
                <div style="background-color:{color}; padding:10px; border-radius:8px; margin-bottom:10px; color:white;">
                    <h4 style='margin-bottom:5px;'>Player {pid}</h4>
                    <p><b>Total Blinks:</b> {tracker.blink_count}</p>
                    <p><b>Avg BPM:</b> {tracker.average_blinks_per_minute()}</p>
                    <p><b>Longest No Blink:</b> {int(tracker.longest_no_blink)} seconds</p>
                    <p><i>{tracker.message}</i></p>
                </div>
                """, unsafe_allow_html=True
            )

def show_summary(stats_col):
    players = st.session_state.players
    blink_log = st.session_state.blink_log
    
    if not players:
        stats_col.write("No data to show.")
        return
    
    stats_col.markdown("## Session Summary Report 📊")
    
    # Show player summary in colorful boxes
    min_blinks = min([tracker.blink_count for tracker in players.values()])
    winners = [pid for pid, tracker in players.items() if tracker.blink_count == min_blinks]
    
    for pid, tracker in players.items():
        color = PLAYER_COLORS[(pid-1) % len(PLAYER_COLORS)]
        winner_tag = ""
        if pid in winners:
            winner_tag = "<strong style='color:#FFD700;'> 👑 Winner!</strong>"
        stats_col.markdown(
            f"""
            <div style="background-color:{color}; padding:15px; border-radius:10px; margin-bottom:15px; color:white;">
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

    # Plot cumulative blinks over time
    start_time = blink_log[0][0]
    end_time = blink_log[-1][0]

    bins = np.arange(start_time, end_time + 10, 10)

    fig, ax = plt.subplots(figsize=(8,4))

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

    # Save session stats to CSV
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"blink_tracker_session_{timestamp_str}.csv"

    with open(filename, 'w') as f:
        f.write("Player,Total Blinks,Average Blinks/Min,Longest No Blink (s)\n")
        for pid, tracker in players.items():
            f.write(f"{pid},{tracker.blink_count},{tracker.average_blinks_per_minute()},{int(tracker.longest_no_blink)}\n")
        f.write("\nBlink Log:\nTimestamp,Player\n")
        for ts, pid in blink_log:
            f.write(f"{datetime.datetime.fromtimestamp(ts)},{pid}\n")

    stats_col.success(f"Session stats saved to {filename}")

def hex_to_bgr(hex_color):
    """Convert hex color string to BGR tuple for OpenCV"""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

if __name__ == "__main__":
    main()
