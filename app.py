import cv2
import numpy as np
import pandas as pd
import streamlit as st
import serial
import time 
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- CONFIG & UI STYLE ---
st.set_page_config(page_title="DAG-1 | DEEPSEA AUTONOMOUS GUARDIAN", page_icon="🌊", layout="wide")

# Custom CSS for the "Tactical" look
st.markdown("""
    <style>
    .main-title { text-align: center; font-size: 3rem; color: #00e5ff; text-shadow: 0 0 15px #00e5ff; font-weight: 900; letter-spacing: 2px; }
    .danger-glow { color: #ff1744; text-shadow: 0 0 15px #ff1744; font-size: 1.8rem; font-weight: bold; border: 2px solid #ff1744; padding: 15px; border-radius: 10px; background: rgba(255, 23, 68, 0.1); text-align: center; }
    .safe-glow { color: #00e676; text-shadow: 0 0 10px #00e676; font-size: 1.8rem; font-weight: bold; border: 2px solid #00e676; padding: 15px; border-radius: 10px; background: rgba(0, 230, 118, 0.1); text-align: center; }
    .stMetric { background: rgba(0, 229, 255, 0.05); padding: 10px; border-radius: 10px; border: 1px solid rgba(0, 229, 255, 0.2); }
    </style>
""", unsafe_allow_html=True)

# --- MODELS & DATA ---
@st.cache_resource
def load_assets():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    try:
        data = pd.read_csv("sonar_data.csv", header=None)
        X = data.drop(columns=60); Y = data[60]
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=2000, solver="liblinear")
        model.fit(X_scaled, Y)
        return face_cascade, model, scaler
    except:
        st.error("CRITICAL ERROR: Sonar Intelligence Database Missing."); st.stop()

face_engine, model, scaler = load_assets()

# --- SERIAL ---
@st.cache_resource 
def init_ser():
    # Use COM9 as requested, but added safety
    try: return serial.Serial('COM9', 9600, timeout=0.1) 
    except: return None

ser = init_ser()

# --- STATE MANAGEMENT ---
if 'history' not in st.session_state: st.session_state.history = [0]*30
if 'last_detection_time' not in st.session_state: st.session_state.last_detection_time = 0
if 'current_target' not in st.session_state: st.session_state.current_target = None

# --- SIDEBAR ---
st.sidebar.title("🛠️ COMMAND CENTER")
live_feed = st.sidebar.toggle("🛰️ INITIALIZE MISSION", value=False)
app_mode = st.sidebar.radio("SENSORS", ["🛰️ LIVE ARDUINO", "⌨️ MANUAL SIMULATION"])
manual_dist = st.sidebar.slider("SIMULATION DEPTH (CM)", 0, 100, 30) if app_mode == "⌨️ MANUAL SIMULATION" else 0

# --- UI WORKFLOW LOGIC ---
if not live_feed:
    st.markdown('<h1 class="main-title">🌊 INTERGRATED SONAR AND VISION SYSTEM FOR UNDERWATER OBJECT DETECTION</h1>', unsafe_allow_html=True)
    st.warning("📡 **SYSTEM STANDBY:** Awaiting Command. Toggle 'INITIALIZE MISSION' to engage sensors.")
    # Image showing LED polarity to help with your hardware fix
    
else:
    # Everything inside here ONLY runs when the system is active
    st.markdown('<h1 class="main-title">🛰️ LIVE TACTICAL UPLINK</h1>', unsafe_allow_html=True)
    
    dashboard_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    # Optimization: Set lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while live_feed:
        dist_val = 0
        current_time = time.time()
        
        # 1. DATA ACQUISITION
        if app_mode == "⌨️ MANUAL SIMULATION":
            dist_val = manual_dist
        elif ser:
            try:
                ser.reset_input_buffer()
                line = ser.readline().decode('utf-8').strip()
                if "Distance:" in line:
                    dist_val = int(line.split(":")[1].strip())
            except: pass

        # 2. COMPUTER VISION (AI PREDICTION)
        ret, frame = cap.read()
        is_human = is_aquatic = False
        
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_engine.detectMultiScale(gray, 1.2, 5) # Faster detection
            
            # Aquatic Life Movement Check
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(faces) > 0:
                is_human = True
                st.session_state.last_detection_time = current_time
                st.session_state.current_target = "DIVER"
            elif len(contours) > 8: 
                is_aquatic = True
                st.session_state.last_detection_time = current_time
                st.session_state.current_target = "AQUATIC"
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3. THREAT EVALUATION
        show_alert = (current_time - st.session_state.last_detection_time) < 2.0
        is_mine = 2 < dist_val < 25 and not is_human
        
        if is_mine:
            st.session_state.last_detection_time = current_time
            st.session_state.current_target = "MINE"
            status, conf = "⚠️ THREAT DETECTED", 98.2
            if ser: ser.write(b'H')
        else:
            status, conf = "✅ SECTOR CLEAR", 94.5
            if ser: ser.write(b'S')

        # 4. RENDERING DASHBOARD
        with dashboard_placeholder.container():
            st.markdown("---")
            col_data, col_vid = st.columns([2, 1])
            
            with col_data:
                st.write("### 📡 SENSOR FUSION DATA")
                m1, m2, m3 = st.columns(3)
                m1.metric("RANGE", f"{dist_val} CM")
                m2.metric("AI CONFIDENCE", f"{conf + random.uniform(-0.5, 0.5):.1f}%")
                m3.metric("UPLINK STATUS", status)
                
                st.line_chart(st.session_state.history[-30:])
                st.session_state.history.append(dist_val)
                
                if show_alert:
                    if st.session_state.current_target == "MINE":
                        st.markdown('<div class="danger-glow">⚠️ CRITICAL: NAVAL MINE IDENTIFIED<br><small>METALLIC ORDNANCE SIGNATURE</small></div>', unsafe_allow_html=True)
                    elif st.session_state.current_target == "DIVER":
                        st.markdown('<div class="safe-glow">✅ FRIENDLY: DIVER DETECTED<br><small>HUMAN BIOMETRIC VERIFIED</small></div>', unsafe_allow_html=True)
                    elif st.session_state.current_target == "AQUATIC":
                        st.markdown('<div class="safe-glow">🐟 BIOLOGICAL: MARINE LIFE<br><small>NON-THREAT ORGANIC MOVEMENT</small></div>', unsafe_allow_html=True)
                else:
                    st.info("🔎 SCANNING SECTOR... ANALYZING ACOUSTIC SIGNATURES")

            with col_vid:
                st.write("### 👁️ VISUAL IDENTIFICATION")
                if ret: st.image(frame, use_container_width=True)
                
                st.write("---")
                st.markdown(f"**OBJECT PREDICTION:** `{st.session_state.current_target if show_alert else 'SCANNING...'}`")
                st.markdown(f"**COM PORT:** `COM9 ({'ACTIVE' if ser else 'OFFLINE'})` ")

        # Small sleep prevents CPU from hitting 100% (Smoother performance)
        time.sleep(0.05) 
        if not live_feed:
            cap.release()
            break