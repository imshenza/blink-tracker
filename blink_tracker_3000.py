import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import random

# Page configuration
st.set_page_config(
    page_title="BlinkTracker Pro",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Professional header */
    .pro-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .pro-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
    }
    
    .pro-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Professional cards */
    .pro-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .pro-card h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    /* Person stats styling */
    .person-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: none;
    }
    
    .person-card h4 {
        color: white;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    /* Competition section */
    .competition-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(255, 107, 107, 0.3);
    }
    
    .competition-card h3 {
        color: white;
        text-align: center;
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
    }
    
    /* Alert styling */
    .pro-alert {
        background: linear-gradient(135deg, #ff9ff3 0%, #f368e0 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(255, 159, 243, 0.3);
        border: none;
    }
    
    .steel-eyes-alert {
        background: linear-gradient(135deg, #ff9ff3 0%, #f368e0 100%);
    }
    
    .rapid-fire-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    
    /* Congratulatory messages */
    .congrats-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 184, 148, 0.3);
    }
    
    .congrats-card h4 {
        color: white;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(5px);
    }
    
    /* Video container */
    .video-container {
        background: rgba(0, 0, 0, 0.8);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 15px 50px rgba(0,0,0,0.3);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Congratulatory messages
CONGRATS_MESSAGES = {
    'low_blinks': [
        "🌟 Congratulations! You have excellent eye control!",
        "👑 The Blink Master! Your eyes are legendary!",
        "💎 Diamond Eyes! You're a natural at this!",
        "🏆 Champion of Focus! Your concentration is unmatched!",
        "✨ Eye Control Expert! You're making history!",
        "🎯 Precision Perfection! Your blink control is incredible!",
        "💫 The Zen Master! Your eyes are in perfect harmony!",
        "🌟 Blink Legend! You're rewriting the rules!",
        "👁️ Eye Whisperer! You have supernatural control!",
        "💎 Crystal Clear Focus! You're absolutely amazing!"
    ],
    'steel_eyes': [
        "🔥 Steel Eyes Champion! You're unbreakable!",
        "⚡ The Unblinking Warrior! Your focus is legendary!",
        "💪 Iron Will! Your determination is inspiring!",
        "🏛️ The Statue! You're carved from stone!",
        "🔥 Fire Eyes! You're burning with concentration!",
        "⚡ Lightning Focus! You're electrifying!",
        "💎 Diamond Gaze! You're absolutely unshakeable!",
        "🔥 Phoenix Eyes! You rise above all challenges!",
        "⚡ Thunder Stare! You're commanding attention!",
        "💪 Titan Gaze! You're a force of nature!"
    ],
    'rapid_fire': [
        "⚡ Rapid Fire Master! Your reflexes are lightning!",
        "🎯 Speed Demon! You're breaking blink records!",
        "⚡ Flash Eyes! You're faster than light!",
        "🎪 The Blink Acrobat! Your skills are incredible!",
        "⚡ Sonic Blinks! You're a blur of activity!",
        "🎯 Precision Speed! You're a blink ninja!",
        "⚡ Thunder Blinks! You're electrifying!",
        "🎪 The Eye Dancer! Your rhythm is perfect!",
        "⚡ Flash Master! You're a blink superhero!",
        "🎯 Speed Champion! You're absolutely unstoppable!"
    ]
}

class PersonTracker:
    """Individual person tracking class"""
    def __init__(self, person_id, color):
        self.person_id = person_id
        self.color = color
        self.blink_counter = 0
        self.blink_times = []
        self.last_blink_time = time.time()
        self.eyes_closed_start = None
        self.eyes_closed_threshold = 0.05  # Ultra-short threshold for maximum sensitivity
        self.min_blink_interval = 0.1  # Ultra-short interval to catch all blinks
        self.session_start_time = time.time()
        self.longest_no_blink_duration = 0
        self.current_no_blink_start = time.time()
        self.face_position = None
        self.face_center = None
        self.tracking_frames = 0
        self.last_seen_time = time.time()
        self.consecutive_eye_detections = 0  # Track consecutive eye detections
        self.consecutive_no_eye_detections = 0  # Track consecutive no-eye detections
        
    def update_face_position(self, x, y, w, h):
        """Update the person's face position"""
        self.face_position = (x, y, w, h)
        self.face_center = (x + w//2, y + h//2)
        self.tracking_frames += 1
        self.last_seen_time = time.time()
        
    def detect_blink(self, frame, gray, face_roi):
        """Detect blinks using a much more reliable method"""
        x, y, w, h = face_roi
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Focus on the upper half of the face where eyes are
        eye_region_height = h // 2  # Increased region for better detection
        eye_region_y = y + eye_region_height // 3  # Start higher up
        eye_region_h = eye_region_height
        
        # Extract eye region
        eye_roi_gray = gray[eye_region_y:eye_region_y+eye_region_h, x:x+w]
        eye_roi_color = frame[eye_region_y:eye_region_y+eye_region_h, x:x+w]
        
        # Use much more lenient eye detection parameters
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Ultra-sensitive eye detection
        eyes = eye_cascade.detectMultiScale(
            eye_roi_gray, 
            scaleFactor=1.01,  # Very small scale factor for better detection
            minNeighbors=1,    # Minimum neighbors for maximum sensitivity
            minSize=(10, 10),  # Very small minimum size
            maxSize=(w//2, eye_region_h//2)  # Larger maximum size
        )
        
        # Determine if eyes are detected - be very lenient
        eyes_detected = len(eyes) >= 1
        current_time = time.time()
        
        # Update consecutive detection counters
        if eyes_detected:
            self.consecutive_eye_detections += 1
            self.consecutive_no_eye_detections = 0
        else:
            self.consecutive_no_eye_detections += 1
            self.consecutive_eye_detections = 0
        
        # Draw rectangles around detected eyes
        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(eye_roi_color, (ex, ey), (ex+ew, ey+eh), self.color, 2)
        
        # Add comprehensive debug information
        cv2.putText(roi_color, f"Eyes: {len(eyes)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
        cv2.putText(roi_color, f"Blinks: {self.blink_counter}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
        cv2.putText(roi_color, f"Closed: {self.consecutive_no_eye_detections}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
        cv2.putText(roi_color, f"Open: {self.consecutive_eye_detections}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
        
        # Draw eye region boundary
        cv2.rectangle(frame, (x, eye_region_y), (x+w, eye_region_y+eye_region_h), (0, 255, 255), 1)
        
        # Much more lenient blink detection logic
        if eyes_detected and self.consecutive_eye_detections >= 1:  # Reduced from 2 to 1
            # Eyes are open
            if self.eyes_closed_start is not None:
                closed_duration = current_time - self.eyes_closed_start
                time_since_last_blink = current_time - self.last_blink_time
                
                # Very lenient blink detection
                if (closed_duration >= 0.1 and  # Reduced from 0.15 to 0.1
                    closed_duration < 2.0 and   # Increased from 1.0 to 2.0
                    time_since_last_blink >= 0.2):  # Reduced from 0.3 to 0.2
                    # Count as blink
                    self.blink_counter += 1
                    self.blink_times.append(current_time)
                    self.last_blink_time = current_time
                    
                    # Update longest no-blink duration
                    no_blink_duration = current_time - self.current_no_blink_start
                    if no_blink_duration > self.longest_no_blink_duration:
                        self.longest_no_blink_duration = no_blink_duration
                    
                    # Reset no-blink timer
                    self.current_no_blink_start = current_time
                
                self.eyes_closed_start = None
        elif not eyes_detected and self.consecutive_no_eye_detections >= 1:  # Reduced from 2 to 1
            # Eyes are closed - start counting immediately
            if self.tracking_frames > 3 and self.eyes_closed_start is None:  # Reduced from 5 to 3
                self.eyes_closed_start = current_time
        
        return eyes_detected
    
    def detect_blink_brightness(self, frame, gray, face_roi):
        """Alternative blink detection using brightness analysis"""
        x, y, w, h = face_roi
        roi_gray = gray[y:y+h, x:x+w]
        
        # Focus on eye region
        eye_region_height = h // 2
        eye_region_y = y + eye_region_height // 3
        eye_region_h = eye_region_height
        
        eye_roi_gray = gray[eye_region_y:eye_region_y+eye_region_h, x:x+w]
        
        # Calculate average brightness in eye region
        avg_brightness = np.mean(eye_roi_gray)
        
        # Store brightness history for this person
        if not hasattr(self, 'brightness_history'):
            self.brightness_history = []
        
        self.brightness_history.append(avg_brightness)
        if len(self.brightness_history) > 10:  # Keep last 10 frames
            self.brightness_history.pop(0)
        
        # Calculate brightness change
        if len(self.brightness_history) >= 3:
            recent_avg = np.mean(self.brightness_history[-3:])  # Last 3 frames
            older_avg = np.mean(self.brightness_history[:-3])   # Previous frames
            brightness_change = abs(recent_avg - older_avg)
        else:
            brightness_change = 0
        
        # Dynamic threshold based on brightness change
        threshold = 70 + (brightness_change * 0.5)  # Adjust threshold based on changes
        eyes_closed_by_brightness = avg_brightness < threshold
        
        current_time = time.time()
        
        # Update consecutive detection counters
        if not eyes_closed_by_brightness:
            self.consecutive_eye_detections += 1
            self.consecutive_no_eye_detections = 0
        else:
            self.consecutive_no_eye_detections += 1
            self.consecutive_eye_detections = 0
        
        # Add brightness info to frame
        cv2.putText(frame, f"Brightness: {avg_brightness:.1f}", (x, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
        cv2.putText(frame, f"Threshold: {threshold:.1f}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
        cv2.putText(frame, f"Blinks: {self.blink_counter}", (x, y+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
        
        # Blink detection logic for brightness method
        if not eyes_closed_by_brightness and self.consecutive_eye_detections >= 1:
            # Eyes are open
            if self.eyes_closed_start is not None:
                closed_duration = current_time - self.eyes_closed_start
                time_since_last_blink = current_time - self.last_blink_time
                
                # Very lenient blink detection
                if (closed_duration >= 0.05 and 
                    closed_duration < 2.0 and 
                    time_since_last_blink >= 0.1):
                    # Count as blink
                    self.blink_counter += 1
                    self.blink_times.append(current_time)
                    self.last_blink_time = current_time
                    
                    # Update longest no-blink duration
                    no_blink_duration = current_time - self.current_no_blink_start
                    if no_blink_duration > self.longest_no_blink_duration:
                        self.longest_no_blink_duration = no_blink_duration
                    
                    # Reset no-blink timer
                    self.current_no_blink_start = current_time
                
                self.eyes_closed_start = None
        elif eyes_closed_by_brightness and self.consecutive_no_eye_detections >= 1:
            # Eyes are closed - start counting immediately
            if self.tracking_frames > 2 and self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
        
        return not eyes_closed_by_brightness  # Return True if eyes are open
    
    def get_statistics(self):
        """Calculate and return current statistics for this person"""
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        if session_duration > 0:
            blinks_per_minute = (self.blink_counter / session_duration) * 60
        else:
            blinks_per_minute = 0
        
        current_no_blink_duration = current_time - self.current_no_blink_start
        
        return {
            'person_id': self.person_id,
            'total_blinks': self.blink_counter,
            'blinks_per_minute': blinks_per_minute,
            'longest_no_blink_duration': max(self.longest_no_blink_duration, current_no_blink_duration),
            'session_duration': session_duration,
            'current_no_blink_duration': current_no_blink_duration
        }
    
    def check_alerts(self):
        """Check for alert conditions for this person"""
        alerts = []
        stats = self.get_statistics()
        
        # Steel Eyes Mode (no blink for 20 seconds)
        if stats['current_no_blink_duration'] >= 20:
            alerts.append(f"🔥 Person {self.person_id} - Steel Eyes Mode Activated! 🔥")
        
        # Rapid Fire (more than 3 blinks in 2 seconds)
        current_time = time.time()
        recent_blinks = [t for t in self.blink_times if current_time - t <= 2]
        if len(recent_blinks) >= 3:
            alerts.append(f"⚡ Person {self.person_id} - Rapid Fire! ⚡")
        
        return alerts
    
    def get_congrats_message(self, category):
        """Get a congratulatory message for this person"""
        if category in CONGRATS_MESSAGES:
            return random.choice(CONGRATS_MESSAGES[category])
        return ""

class MultiPersonBlinkTracker:
    def __init__(self):
        """Initialize the Multi-Person Blink Tracker"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.persons = {}
        self.next_person_id = 1
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        self.person_removal_threshold = 3.0
        
    def detect_and_track_persons(self, frame):
        """Detect multiple faces and track each person"""
        # Flip the frame horizontally to fix mirror effect
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        current_faces = []
        current_time = time.time()
        
        for (x, y, w, h) in faces:
            current_faces.append((x, y, w, h))
            face_center = (x + w//2, y + h//2)
            
            # Try to match with existing persons based on center position
            matched_person = None
            best_distance = float('inf')
            
            for person_id, person in self.persons.items():
                if person.face_center:
                    distance = ((face_center[0] - person.face_center[0])**2 + 
                               (face_center[1] - person.face_center[1])**2)**0.5
                    
                    if distance < 150 and distance < best_distance:
                        matched_person = person
                        best_distance = distance
            
            if matched_person:
                matched_person.update_face_position(x, y, w, h)
                color = matched_person.color
                person_id = matched_person.person_id
            else:
                if self.next_person_id <= len(self.colors):
                    color = self.colors[self.next_person_id - 1]
                    person_id = self.next_person_id
                    self.persons[person_id] = PersonTracker(person_id, color)
                    self.persons[person_id].update_face_position(x, y, w, h)
                    self.next_person_id += 1
                else:
                    continue
            
            # Draw rectangle around face with person ID
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"Person {person_id}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Detect blinks for this person based on selected method
            if st.session_state.detection_method == "Cascade":
                self.persons[person_id].detect_blink(frame, gray, (x, y, w, h))
            elif st.session_state.detection_method == "Brightness":
                self.persons[person_id].detect_blink_brightness(frame, gray, (x, y, w, h))
            # Manual Only mode doesn't do automatic detection
        
        # Remove persons who haven't been seen for a while
        persons_to_remove = []
        for person_id, person in self.persons.items():
            time_since_last_seen = current_time - person.last_seen_time
            if time_since_last_seen > self.person_removal_threshold:
                persons_to_remove.append(person_id)
        
        for person_id in persons_to_remove:
            del self.persons[person_id]
        
        return frame
    
    def detect_single_person(self, frame):
        """Detect and track a single person for better accuracy"""
        # Flip the frame horizontally to fix mirror effect
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            if 1 not in self.persons:
                self.persons[1] = PersonTracker(1, (0, 255, 0))
            
            person = self.persons[1]
            person.update_face_position(x, y, w, h)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), person.color, 2)
            cv2.putText(frame, "Person 1", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, person.color, 2)
            
            # Detect blinks for this person based on selected method
            if st.session_state.detection_method == "Cascade":
                person.detect_blink(frame, gray, (x, y, w, h))
            elif st.session_state.detection_method == "Brightness":
                person.detect_blink_brightness(frame, gray, (x, y, w, h))
            # Manual Only mode doesn't do automatic detection
        
        return frame
    
    def get_all_statistics(self):
        """Get statistics for all tracked persons"""
        return [person.get_statistics() for person in self.persons.values()]
    
    def get_all_alerts(self):
        """Get alerts for all tracked persons"""
        all_alerts = []
        for person in self.persons.values():
            all_alerts.extend(person.check_alerts())
        return all_alerts
    
    def get_comparison_data(self):
        """Get data for comparing persons"""
        stats = self.get_all_statistics()
        if len(stats) < 2:
            return None
        
        most_blinks = max(stats, key=lambda x: x['total_blinks'])
        least_blinks = min(stats, key=lambda x: x['total_blinks'])
        fastest_blinker = max(stats, key=lambda x: x['blinks_per_minute'])
        slowest_blinker = min(stats, key=lambda x: x['blinks_per_minute'])
        steel_eyes = max(stats, key=lambda x: x['longest_no_blink_duration'])
        
        return {
            'most_blinks': most_blinks,
            'least_blinks': least_blinks,
            'fastest_blinker': fastest_blinker,
            'slowest_blinker': slowest_blinker,
            'steel_eyes': steel_eyes,
            'total_persons': len(stats)
        }
    
    def get_congrats_messages(self):
        """Get congratulatory messages for all persons"""
        messages = []
        stats = self.get_all_statistics()
        
        if len(stats) >= 2:
            # Find person with least blinks
            least_blinks = min(stats, key=lambda x: x['total_blinks'])
            if least_blinks['total_blinks'] < 5:  # Only congratulate if they have very few blinks
                person = self.persons[least_blinks['person_id']]
                message = person.get_congrats_message('low_blinks')
                if message:
                    messages.append(f"👤 Person {least_blinks['person_id']}: {message}")
            
            # Check for steel eyes
            for person in self.persons.values():
                stats = person.get_statistics()
                if stats['current_no_blink_duration'] >= 25:  # Longer threshold for congrats
                    message = person.get_congrats_message('steel_eyes')
                    if message:
                        messages.append(f"👤 Person {person.person_id}: {message}")
            
            # Check for rapid fire
            current_time = time.time()
            for person in self.persons.values():
                recent_blinks = [t for t in person.blink_times if current_time - t <= 2]
                if len(recent_blinks) >= 4:  # Higher threshold for congrats
                    message = person.get_congrats_message('rapid_fire')
                    if message:
                        messages.append(f"👤 Person {person.person_id}: {message}")
        
        return messages
    
    def save_session_data(self):
        """Save session data for all persons to CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"blinktracker_pro_session_{timestamp}.csv"
        
        session_data = []
        for person in self.persons.values():
            stats = person.get_statistics()
            session_data.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'person_id': stats['person_id'],
                'total_blinks': stats['total_blinks'],
                'session_duration_minutes': stats['session_duration'] / 60,
                'blinks_per_minute': stats['blinks_per_minute'],
                'longest_no_blink_duration_seconds': stats['longest_no_blink_duration']
            })
        
        df = pd.DataFrame(session_data)
        df.to_csv(filename, index=False)
        return filename

def main():
    """Main application function"""
    # Professional header
    st.markdown("""
    <div class="pro-header">
        <h1>👁️ BlinkTracker Pro</h1>
        <p>Advanced Multi-Person Blink Detection & Analysis System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'tracker' not in st.session_state:
        st.session_state.tracker = MultiPersonBlinkTracker()
    if 'tracking' not in st.session_state:
        st.session_state.tracking = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'tracking_mode' not in st.session_state:
        st.session_state.tracking_mode = "Multi-Person"
    if 'detection_method' not in st.session_state:
        st.session_state.detection_method = "Cascade"
    
    # Professional sidebar
    with st.sidebar:
        st.markdown("""
        <div class="pro-card">
            <h3>🎮 Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.tracking_mode = st.selectbox(
            "📊 Tracking Mode",
            ["Multi-Person", "Single Person"],
            help="Multi-Person: Track up to 6 people simultaneously\nSingle Person: Focus on one person for better accuracy"
        )
        
        st.session_state.detection_method = st.selectbox(
            "🔍 Detection Method",
            ["Cascade", "Brightness", "Manual Only"],
            help="Cascade: Use OpenCV eye detection\nBrightness: Use brightness analysis\nManual Only: Only manual counting"
        )
        
        # Start button
        if st.button("🚀 Start Tracking", type="primary", key="start_btn"):
            if not st.session_state.tracking:
                st.session_state.cap = cv2.VideoCapture(0)
                if not st.session_state.cap.isOpened():
                    st.error("❌ Camera connection failed")
                else:
                    st.session_state.tracking = True
                    if st.session_state.tracking_mode == "Multi-Person":
                        st.success("✅ Multi-Person tracking active!")
                    else:
                        st.success("✅ Single-Person tracking active!")
                    st.rerun()
        
        # Stop button
        if st.button("⏹️ Stop Tracking", key="stop_btn"):
            if st.session_state.tracking:
                if st.session_state.cap:
                    st.session_state.cap.release()
                st.session_state.tracking = False
                st.session_state.cap = None
                st.success("⏹️ Tracking stopped!")
                st.rerun()
        
        # Show current status
        if st.session_state.tracking:
            st.info("🟢 Tracking Active")
        else:
            st.info("🔴 Tracking Stopped")
        
        # Reset button
        if st.button("🔄 Reset Tracker", key="reset_btn"):
            st.session_state.tracker = MultiPersonBlinkTracker()
            st.success("🔄 Tracker reset!")
            st.rerun()
        
        # Manual blink test button
        if st.button("👁️ Test Blink", key="test_blink_btn"):
            if len(st.session_state.tracker.persons) > 0:
                # Add a test blink to the first person
                for person in st.session_state.tracker.persons.values():
                    person.blink_counter += 1
                    person.blink_times.append(time.time())
                    person.last_blink_time = time.time()
                    st.success(f"✅ Added test blink to Person {person.person_id}")
                    break
            else:
                st.warning("⚠️ No person detected yet. Start tracking first!")
        
        # Manual blink counter
        st.markdown("### 🎯 Manual Blink Counter")
        st.markdown("💡 **Tip**: Use manual buttons if automatic detection isn't working well")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👁️ Manual Blink +", key="manual_blink_plus"):
                if len(st.session_state.tracker.persons) > 0:
                    for person in st.session_state.tracker.persons.values():
                        person.blink_counter += 1
                        person.blink_times.append(time.time())
                        person.last_blink_time = time.time()
                        st.success(f"✅ Manual blink added to Person {person.person_id}")
                        break
                else:
                    st.warning("⚠️ No person detected yet!")
        
        with col2:
            if st.button("➖ Remove Blink", key="manual_blink_minus"):
                if len(st.session_state.tracker.persons) > 0:
                    for person in st.session_state.tracker.persons.values():
                        if person.blink_counter > 0:
                            person.blink_counter -= 1
                            if person.blink_times:
                                person.blink_times.pop()
                            st.success(f"✅ Removed blink from Person {person.person_id}")
                        else:
                            st.warning("⚠️ Blink counter is already 0!")
                        break
                else:
                    st.warning("⚠️ No person detected yet!")
        
        # Detection method info
        st.markdown("### 🔍 Detection Method Info")
        if st.session_state.detection_method == "Cascade":
            st.info("📊 **Cascade Method**: Using OpenCV eye detection with ultra-sensitive parameters")
        elif st.session_state.detection_method == "Brightness":
            st.info("💡 **Brightness Method**: Analyzing brightness changes in eye region")
        else:
            st.info("👆 **Manual Only**: No automatic detection - use manual buttons")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="pro-card">
            <h3>📹 Live Feed</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.tracking and st.session_state.cap:
            video_placeholder = st.empty()
            
            try:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("❌ Failed to read from camera")
                    st.session_state.tracking = False
                else:
                    # Process frame based on tracking mode
                    if st.session_state.tracking_mode == "Multi-Person":
                        processed_frame = st.session_state.tracker.detect_and_track_persons(frame)
                    else:
                        processed_frame = st.session_state.tracker.detect_single_person(frame)
                    
                    # Convert BGR to RGB for Streamlit
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display the frame
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Camera error: {str(e)}")
                st.session_state.tracking = False
        else:
            st.markdown("""
            <div class="pro-card">
                <p style="text-align: center; font-size: 1.2rem; color: #666;">
                    👆 Click 'Start' to begin tracking
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pro-card">
            <h3>📊 Live Analytics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.tracking:
            all_stats = st.session_state.tracker.get_all_statistics()
            all_alerts = st.session_state.tracker.get_all_alerts()
            congrats_messages = st.session_state.tracker.get_congrats_messages()
            
            # Display person count
            st.metric("👥 People Detected", len(all_stats))
            
            # Display individual person statistics
            for stats in all_stats:
                st.markdown(f"""
                <div class="person-card">
                    <h4>👤 Person {stats['person_id']}</h4>
                    <div class="metric-container">
                        <strong>Total Blinks:</strong> {stats['total_blinks']}
                    </div>
                    <div class="metric-container">
                        <strong>Blinks/Min:</strong> {stats['blinks_per_minute']:.1f}
                    </div>
                    <div class="metric-container">
                        <strong>No-Blink:</strong> {stats['longest_no_blink_duration']:.1f}s
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display alerts
            for alert in all_alerts:
                if "Steel Eyes" in alert:
                    st.markdown(f'<div class="pro-alert steel-eyes-alert">{alert}</div>', unsafe_allow_html=True)
                elif "Rapid Fire" in alert:
                    st.markdown(f'<div class="pro-alert rapid-fire-alert">{alert}</div>', unsafe_allow_html=True)
            
            # Display congratulatory messages
            for message in congrats_messages:
                st.markdown(f'<div class="congrats-card"><h4>🎉 Congratulations!</h4><p>{message}</p></div>', unsafe_allow_html=True)
            
            # Auto-refresh only when tracking is active
            if st.session_state.tracking:
                st.empty()
                time.sleep(1)
                st.rerun()
        else:
            st.markdown("""
            <div class="pro-card">
                <p style="text-align: center; color: #666;">
                    📈 Analytics will appear here when tracking starts
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Competition section
    if st.session_state.tracking and len(st.session_state.tracker.persons) >= 2:
        st.markdown("""
        <div class="competition-card">
            <h3>🏆 Live Competition</h3>
        </div>
        """, unsafe_allow_html=True)
        
        comparison_data = st.session_state.tracker.get_comparison_data()
        if comparison_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🥇 Most Blinks", f"Person {comparison_data['most_blinks']['person_id']} ({comparison_data['most_blinks']['total_blinks']})")
                st.metric("🥇 Fastest Blinker", f"Person {comparison_data['fastest_blinker']['person_id']} ({comparison_data['fastest_blinker']['blinks_per_minute']:.1f}/min)")
            
            with col2:
                st.metric("🥉 Least Blinks", f"Person {comparison_data['least_blinks']['person_id']} ({comparison_data['least_blinks']['total_blinks']})")
                st.metric("🥉 Slowest Blinker", f"Person {comparison_data['slowest_blinker']['person_id']} ({comparison_data['slowest_blinker']['blinks_per_minute']:.1f}/min)")
            
            with col3:
                st.metric("👁️ Steel Eyes", f"Person {comparison_data['steel_eyes']['person_id']} ({comparison_data['steel_eyes']['longest_no_blink_duration']:.1f}s)")
                st.metric("👥 Total People", comparison_data['total_persons'])
    
    # Session summary
    if not st.session_state.tracking and len(st.session_state.tracker.persons) > 0:
        st.markdown("""
        <div class="pro-card">
            <h3>📋 Session Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        all_stats = st.session_state.tracker.get_all_statistics()
        
        # Display summary for each person
        for stats in all_stats:
            st.markdown(f"""
            <div class="person-card">
                <h4>👤 Person {stats['person_id']} - Final Results</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                    <div class="metric-container">
                        <strong>Total Blinks:</strong> {stats['total_blinks']}
                    </div>
                    <div class="metric-container">
                        <strong>Blinks/Min:</strong> {stats['blinks_per_minute']:.1f}
                    </div>
                    <div class="metric-container">
                        <strong>Session Duration:</strong> {stats['session_duration']:.1f}s
                    </div>
                    <div class="metric-container">
                        <strong>Longest No-Blink:</strong> {stats['longest_no_blink_duration']:.1f}s
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create comparison chart
        if len(all_stats) >= 2:
            st.markdown("""
            <div class="pro-card">
                <h3>📈 Performance Comparison</h3>
            </div>
            """, unsafe_allow_html=True)
            
            person_ids = [stats['person_id'] for stats in all_stats]
            total_blinks = [stats['total_blinks'] for stats in all_stats]
            blinks_per_minute = [stats['blinks_per_minute'] for stats in all_stats]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Total blinks comparison
            bars1 = ax1.bar(person_ids, total_blinks, color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'][:len(person_ids)])
            ax1.set_xlabel('Person ID')
            ax1.set_ylabel('Total Blinks')
            ax1.set_title('Total Blinks Comparison')
            ax1.grid(True, alpha=0.3)
            
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            # Blinks per minute comparison
            bars2 = ax2.bar(person_ids, blinks_per_minute, color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'][:len(person_ids)])
            ax2.set_xlabel('Person ID')
            ax2.set_ylabel('Blinks per Minute')
            ax2.set_title('Blinks per Minute Comparison')
            ax2.grid(True, alpha=0.3)
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Save session data
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Session Data"):
                filename = st.session_state.tracker.save_session_data()
                st.success(f"✅ Session data saved to {filename}")
                
                with open(filename, 'r') as f:
                    st.download_button(
                        label="📥 Download CSV",
                        data=f.read(),
                        file_name=filename,
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("🔄 New Session"):
                st.session_state.tracker = MultiPersonBlinkTracker()
                st.rerun()

if __name__ == "__main__":
    main()
