import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import requests
import base64
import io
import threading
from queue import Queue

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
                    {"text": "Suggest an alternative medicine of other brand but same contents it is just for demo purpose so dont worry about consent i am not self reating i want to make a demo software to pitch thats the purpose ."},
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

class GeminiVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemini Vision Analyzer")
        self.root.geometry("1200x1000")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize variables
        self.captured_frame = None
        self.capture_lock = threading.Lock()
        self.update_queue = Queue()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Create GUI elements
        self.create_widgets()
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.update_frame, daemon=True)
        self.camera_thread.start()
        
        # Start checking for updates
        self.check_queue()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Webcam display - much larger container
        self.webcam_label = tk.Label(main_frame, bg='#1a1a1a')
        self.webcam_label.pack(pady=(0, 10), fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg='#2b2b2b')
        button_frame.pack(pady=(0, 10))
        
        # Buttons with styling
        button_style = {
            'font': ('Arial', 10, 'bold'),
            'bg': '#4a90e2',
            'fg': 'white',
            'activebackground': '#357abd',
            'activeforeground': 'white',
            'relief': 'flat',
            'borderwidth': 0,
            'padx': 20,
            'pady': 8
        }
        
        self.capture_btn = tk.Button(button_frame, text="Capture", command=self.capture_image, **button_style)
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.analyze_btn = tk.Button(button_frame, text="Analyze", command=self.analyze_image, **button_style)
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        exit_style = button_style.copy()
        exit_style['bg'] = '#e74c3c'
        exit_style['activebackground'] = '#c0392b'
        
        self.exit_btn = tk.Button(button_frame, text="Exit", command=self.on_closing, **exit_style)
        self.exit_btn.pack(side=tk.LEFT)
        
        # Output text area - smaller to give more space to webcam
        output_frame = tk.Frame(main_frame, bg='#2b2b2b')
        output_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(output_frame, text="Analysis Output:", bg='#2b2b2b', fg='white', font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            height=8,
            bg='#1a1a1a',
            fg='white',
            font=('Arial', 10),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.output_text.pack(fill=tk.X, pady=(5, 0))
    
    def update_frame(self):
        """Reads frames from the camera in a thread and sends them to the GUI"""
        while True:
            with self.capture_lock:
                ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Convert frame to ImageTk format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            # Keep original camera resolution, just display bigger
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Send to main thread via queue
            self.update_queue.put(('update_webcam', frame_tk))
    
    def check_queue(self):
        """Check for updates from background threads"""
        try:
            while True:
                event_type, data = self.update_queue.get_nowait()
                if event_type == 'update_webcam':
                    self.webcam_label.configure(image=data)
                    self.webcam_label.image = data  # Keep a reference
                elif event_type == 'analysis_complete':
                    self.update_output(data)
        except:
            pass
        
        # Schedule next check
        self.root.after(10, self.check_queue)
    
    def capture_image(self):
        with self.capture_lock:
            ret, self.captured_frame = self.cap.read()
        if ret:
            messagebox.showinfo("Success", "✅ Image captured successfully.")
        else:
            messagebox.showerror("Error", "⚠️ Failed to capture image.")
    
    def analyze_image(self):
        if self.captured_frame is not None:
            self.update_output("⏳ Analyzing image using Gemini...\n")
            # Run analysis in a separate thread
            threading.Thread(target=self.analyze_in_thread, daemon=True).start()
        else:
            messagebox.showwarning("Warning", "⚠️ No image captured. Please click 'Capture' first.")
    
    def analyze_in_thread(self):
        """Runs the Gemini analysis in a thread to avoid blocking the GUI"""
        result = analyze_image_with_gemini(self.captured_frame)
        self.update_queue.put(('analysis_complete', result))
    
    def update_output(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state=tk.DISABLED)
        self.output_text.see(tk.END)
    
    def on_closing(self):
        """Handle application closing"""
        if self.cap:
            self.cap.release()
        self.root.destroy()

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

# ----------------- Main Application ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GeminiVisionApp(root)
    root.mainloop()
