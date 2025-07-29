import PySimpleGUI as sg
import cv2
import numpy as np
from PIL import Image
import requests
import base64
import io
import threading

# ----------------- Gemini API Config ------------------
API_KEY = 'AIzaSyBvd6TJtTKaFNEqAb9w_6t9wX3N3awW3LA'
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

# ----------------- Functions ------------------
def encode_image_to_base64(image):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def analyze_image_with_gemini(image):
    base64_image = encode_image_to_base64(image)
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"text": "Describe this image."},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }
        ]
    }
    response = requests.post(GEMINI_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        try:
            return result['candidates'][0]['content']['parts'][0]['text']
        except:
            return "❌ Gemini didn't return a valid response."
    else:
        return f"❌ Error {response.status_code}: {response.text}"

def update_frame(window, cap, lock):
    """Reads frames from the camera in a thread and sends them to the GUI"""
    while True:
        with lock:
            ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window.write_event_value("-UPDATE_WEBCAM-", imgbytes)

def analyze_in_thread(window, frame):
    """Runs the Gemini analysis in a thread to avoid blocking the GUI"""
    result = analyze_image_with_gemini(frame)
    window.write_event_value("-ANALYSIS_COMPLETE-", result)

# ----------------- GUI Layout ------------------
sg.theme("DarkBlue3")
layout = [
    [sg.Image(filename="", key="webcam")],
    [sg.Button("Capture"), sg.Button("Analyze"), sg.Button("Exit")],
    [sg.Multiline(size=(60, 10), key="output", disabled=True)]
]

window = sg.Window("Gemini Vision Analyzer", layout, finalize=True)

# ----------------- Video Capture ------------------
cap = cv2.VideoCapture(0)
captured_frame = None
capture_lock = threading.Lock()

# Run camera in a background thread
threading.Thread(target=update_frame, args=(window, cap, capture_lock), daemon=True).start()

# ----------------- Event Loop ------------------
while True:
    event, values = window.read()
    if event in (sg.WINDOW_CLOSED, "Exit"):
        break
    elif event == "-UPDATE_WEBCAM-":
        window["webcam"].update(data=values[event])
    elif event == "Capture":
        with capture_lock:
            # Read the latest frame from the camera
            ret, captured_frame = cap.read()
        if ret:
            sg.popup("✅ Image captured successfully.")
        else:
            sg.popup("⚠️ Failed to capture image.")
    elif event == "Analyze":
        if captured_frame is not None:
            window["output"].update("⏳ Analyzing image using Gemini...\n")
            # Run analysis in a separate thread
            threading.Thread(target=analyze_in_thread, args=(window, captured_frame), daemon=True).start()
        else:
            sg.popup("⚠️ No image captured. Please click 'Capture' first.")
    elif event == "-ANALYSIS_COMPLETE-":
        window["output"].update(values[event])

# Cleanup
cap.release()
window.close()
