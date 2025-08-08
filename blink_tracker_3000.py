import cv2
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
from collections import deque

# ------------------ Load Haar cascades ------------------
# Using OpenCV's pre-trained classifiers for face and eyes detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ------------------ Blink Tracker Classes ------------------
class PlayerBlinkTracker:
    """
    Tracks blink data for a single detected face/player.
    """
    def __init__(self, player_id):
        self.player_id = player_id
        self.blink_count = 0
        self.last_blink_time = None
        self.blink_timestamps = []  # store blink event times
        self.no_blink_start = time.time()
        self.longest_no_blink = 0
        self.blink_queue = deque()  # store recent blink times for rapid fire detection
        self.message = ""  # playful message
    
    def register_blink(self):
        now = time.time()
        self.blink_count += 1
        self.blink_timestamps.append(now)
        
        # Calculate no blink duration
        if self.last_blink_time:
            no_blink_duration = now - self.last_blink_time
            if no_blink_duration > self.longest_no_blink:
                self.longest_no_blink = no_blink_duration
        self.last_blink_time = now
        
        # Reset no blink timer
        self.no_blink_start = now
        
        # Maintain blink queue for last 2 seconds to detect rapid blinks
        self.blink_queue.append(now)
        while self.blink_queue and now - self.blink_queue[0] > 2:
            self.blink_queue.popleft()
        
        # Update playful messages
        self.update_messages()
    
    def update_no_blink(self):
        # Check how long no blink has been happening
        now = time.time()
        no_blink_duration = now - self.no_blink_start
        if no_blink_duration >= 20:
            self.message = "Steel Eyes Mode Activated 👁️🦾"
        else:
            # If no special message set by blinks, clear message
            if len(self.blink_queue) <= 3:
                self.message = ""
    
    def update_messages(self):
        # If more than 3 blinks in last 2 seconds
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
    st.title("👁️ Blink Tracker 3000")
    st.write("Track your blinks live with fun messages! Supports multiple faces & players.")
    
    # Button to start/stop tracking
    if 'tracking' not in st.session_state:
        st.session_state.tracking = False
    if 'players' not in st.session_state:
        st.session_state.players = dict()
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'blink_log' not in st.session_state:
        # blink_log stores tuples of (timestamp, player_id)
        st.session_state.blink_log = []
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None

    start_stop_btn = st.button("Start Tracking" if not st.session_state.tracking else "Stop Tracking")
    
    # Video display placeholder
    video_placeholder = st.empty()
    
    # When Start Tracking is pressed
    if start_stop_btn:
        if not st.session_state.tracking:
            # Reset all trackers
            st.session_state.tracking = True
            st.session_state.players = dict()
            st.session_state.frame_count = 0
            st.session_state.blink_log = []
            st.session_state.start_time = time.time()
            st.success("Tracking started! Please look at the camera.")
        else:
            # Stop tracking
            st.session_state.tracking = False

    # Only track if tracking enabled
    if st.session_state.tracking:
        run_tracker(video_placeholder)
    else:
        if st.session_state.start_time is not None:
            # Show summary after stopping
            show_summary()
            st.session_state.start_time = None

def run_tracker(video_placeholder):
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        return
    
    # Used to detect blink by eye aspect ratio (simple heuristic)
    # Here, since Haar cascades don't detect eyes closed, we will use eye detection presence to guess blinks.
    # If eyes disappear for a frame, count as blink.
    
    # To keep track of eyes presence per player between frames
    eyes_present_prev = dict()
    
    # Run until stop button is clicked or webcam closed
    try:
        while st.session_state.tracking:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break
            
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Sort faces left to right (for stable player labeling)
            faces = sorted(faces, key=lambda x: x[0])
            
            # Update players dictionary: Add new players if faces detected more than existing
            current_face_count = len(faces)
            existing_player_count = len(st.session_state.players)
            if current_face_count > existing_player_count:
                for i in range(existing_player_count + 1, current_face_count + 1):
                    st.session_state.players[i] = PlayerBlinkTracker(i)
            
            # Track blinks and draw boxes
            eyes_present_curr = dict()
            
            for i, (x, y, w, h) in enumerate(faces):
                player_id = i + 1
                face_gray = gray[y:y+h, x:x+w]
                face_color = frame[y:y+h, x:x+w]
                
                # Detect eyes within face region
                eyes = eye_cascade.detectMultiScale(face_gray)
                
                # Draw face rectangle + label player
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
                cv2.putText(frame, f"Player {player_id}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                
                # Draw rectangles around eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
                
                # Check eyes presence
                eyes_present_curr[player_id] = len(eyes) > 0
                
                # Blink detection logic:
                # If eyes present last frame but now eyes missing => blink
                if player_id in eyes_present_prev:
                    if eyes_present_prev[player_id] and not eyes_present_curr[player_id]:
                        st.session_state.players[player_id].register_blink()
                        st.session_state.blink_log.append( (time.time(), player_id) )
                
                # Update no blink message if eyes present (or not)
                st.session_state.players[player_id].update_no_blink()
            
            # Update eyes_present_prev for next iteration
            eyes_present_prev = eyes_present_curr
            
            # Show stats and messages on frame
            y0 = 30
            dy = 30
            for player_id, tracker in st.session_state.players.items():
                text1 = f"Player {player_id} - Total Blinks: {tracker.blink_count}"
                text2 = f"Avg BPM: {tracker.average_blinks_per_minute()}"
                text3 = f"Longest No Blink: {int(tracker.longest_no_blink)}s"
                text4 = f"Message: {tracker.message}"
                cv2.putText(frame, text1, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, text2, (10, y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, text3, (10, y0+2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, text4, (10, y0+3*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y0 += 110
            
            # Convert frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Show frame in Streamlit
            video_placeholder.image(frame_rgb)
            
            # Limit frame rate to ~15 FPS
            time.sleep(1/15)
            
    finally:
        cap.release()

def show_summary():
    st.subheader("Session Summary Report")
    players = st.session_state.players
    blink_log = st.session_state.blink_log
    
    if not players:
        st.write("No data to show.")
        return
    
    # Display stats table
    data = []
    for pid, tracker in players.items():
        data.append({
            "Player": pid,
            "Total Blinks": tracker.blink_count,
            "Average Blinks/Min": tracker.average_blinks_per_minute(),
            "Longest No Blink (s)": int(tracker.longest_no_blink)
        })
    df_stats = pd.DataFrame(data)
    st.dataframe(df_stats)
    
    # Prepare blink counts over time for chart
    # For each player, create a time series of cumulative blinks per 10-second interval
    if len(blink_log) == 0:
        st.write("No blinks recorded in this session.")
        return
    
    start_time = blink_log[0][0]
    end_time = blink_log[-1][0]
    duration = end_time - start_time
    
    # Create time bins every 10 seconds
    bins = np.arange(start_time, end_time + 10, 10)
    
    # Plot cumulative blinks for each player over time
    fig, ax = plt.subplots()
    
    for pid in players.keys():
        times = [t for (t,p) in blink_log if p == pid]
        counts, _ = np.histogram(times, bins)
        cumulative_counts = np.cumsum(counts)
        ax.plot(bins[:-1] - start_time, cumulative_counts, label=f"Player {pid}")
    
    ax.set_title("Cumulative Blinks Over Time")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Blinks")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Save session stats to CSV
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"blink_tracker_session_{timestamp_str}.csv"
    
    # Save player stats and blink log
    with open(filename, 'w') as f:
        f.write("Player,Total Blinks,Average Blinks/Min,Longest No Blink (s)\n")
        for pid, tracker in players.items():
            f.write(f"{pid},{tracker.blink_count},{tracker.average_blinks_per_minute()},{int(tracker.longest_no_blink)}\n")
        f.write("\nBlink Log:\nTimestamp,Player\n")
        for ts, pid in blink_log:
            f.write(f"{datetime.datetime.fromtimestamp(ts)},{pid}\n")
    
    st.success(f"Session stats saved to {filename}")

if __name__ == "__main__":
    main()
