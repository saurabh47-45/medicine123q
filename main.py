import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
import time
import detection
import gemini

class VideoStreamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medicine Detection System")
        self.root.configure(bg='#2c3e50')  # Dark blue-gray background
        
        # Set window size and make it resizable
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(2)
        self.running = True
        
        # Motion detection variables
        self.can_detect = True
        self.motion_threshold = 5000  # Change this value to adjust sensitivity
        self.prev_frame = None
        
        # Contrast adjustment parameters
        self.contrast_alpha = 0.8  # Contrast control (1.0 = normal, <1.0 = reduced contrast)
        self.brightness_beta = 50   # Brightness control (0 = normal, positive = brighter)
        
        # Countdown variables
        self.countdown_active = False
        self.countdown_start_time = 0
        self.countdown_duration = 5.0  # 5 second countdown
        
        # Processing state
        self.processing = False
        
        # Create and setup UI
        self.setup_ui()
        
        # Start video stream
        self.start_video()
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Setup the user interface with dashboard layout"""
        
        # Main container with padding
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Top section - Camera video (left) and Status panels (right)
        top_section = tk.Frame(main_container, bg='#2c3e50')
        top_section.pack(fill="x", pady=(0, 20))  # Changed from fill="both", expand=True
        
        # Configure grid weights for responsive layout
        top_section.grid_columnconfigure(0, weight=2)  # Camera gets more space
        top_section.grid_columnconfigure(1, weight=1)  # Status panels get less space
        # Remove the row weight so it doesn't expand vertically
        
        # Left side - Camera Video Frame
        camera_frame = tk.Frame(top_section, 
                               bg="#34495e",
                               relief="solid",
                               bd=3,
                               height=400)  # Fixed height
        camera_frame.grid(row=0, column=0, sticky="ew", padx=(0, 10))  # Changed from "nsew" to "ew"
        camera_frame.grid_propagate(False)  # Prevent frame from shrinking
        
        # Camera title
        camera_title = tk.Label(camera_frame,
                               text="Camera Video",
                               font=("Arial", 20, "bold"),
                               fg="white",
                               bg="#34495e")
        camera_title.pack(pady=(15, 10))
        
        # Video display label
        self.video_label = tk.Label(camera_frame, 
                                   bg="#34495e",
                                   relief="sunken",
                                   bd=2)
        self.video_label.pack(padx=15, pady=(0, 15))
        
        # Right side - Status panels container
        status_container = tk.Frame(top_section, bg='#2c3e50')
        status_container.grid(row=0, column=1, sticky="ew")  # Changed from "nsew" to "ew"
        
        # System Status Panel
        system_frame = tk.Frame(status_container,
                               bg="#34495e",
                               relief="solid",
                               bd=3,
                               height=180)  # Fixed height
        system_frame.pack(fill="x", pady=(0, 15))
        system_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        system_title = tk.Label(system_frame,
                               text="System Status",
                               font=("Arial", 18, "bold"),
                               fg="white",
                               bg="#34495e")
        system_title.pack(pady=(15, 10))
        
        self.status_label = tk.Label(system_frame, 
                                    text="Ready for motion detection",
                                    font=("Arial", 16, "bold"),
                                    fg="#2ecc71",
                                    bg="#34495e")
        self.status_label.pack(pady=(0, 15))
        
        # Validation Result Panel
        validation_frame = tk.Frame(status_container,
                                   bg="#34495e",
                                   relief="solid",
                                   bd=3,
                                   height=180)  # Fixed height
        validation_frame.pack(fill="x")
        validation_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        validation_title = tk.Label(validation_frame,
                                   text="Validation Result",
                                   font=("Arial", 18, "bold"),
                                   fg="white",
                                   bg="#34495e")
        validation_title.pack(pady=(15, 10))
        
        self.medicine_label = tk.Label(validation_frame, 
                                      text="Waiting for detection...",
                                      font=("Arial", 16, "bold"),
                                      fg="#f39c12",
                                      bg="#34495e")
        self.medicine_label.pack(pady=(0, 15))
        
        # Bottom section - AI Recommendation (full width)
        ai_frame = tk.Frame(main_container,
                           bg="#34495e",
                           relief="solid",
                           bd=3,
                           height=250)  # Increased height
        ai_frame.pack(fill="both", expand=True, pady=(0, 0))
        ai_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        ai_title = tk.Label(ai_frame,
                           text="AI Recommendation",
                           font=("Arial", 20, "bold"),
                           fg="white",
                           bg="#34495e")
        ai_title.pack(pady=(15, 10))
        
        # Create a frame for the recommendation content
        recommendation_content = tk.Frame(ai_frame, bg="#34495e")
        recommendation_content.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        
        self.alternative_label = tk.Label(recommendation_content, 
                                         text="",
                                         font=("Arial", 16),
                                         fg="#ecf0f1",
                                         bg="#34495e",
                                         justify="left",
                                         anchor="nw")
        self.alternative_label.pack(fill="both", expand=True)
    
    def adjust_contrast_brightness(self, frame):
        """Adjust contrast and brightness of the frame"""
        adjusted = cv2.convertScaleAbs(frame, alpha=self.contrast_alpha, beta=self.brightness_beta)
        return adjusted
    
    def start_video(self):
        """Start the video capture thread"""
        self.video_thread = threading.Thread(target=self.update_frame, daemon=True)
        self.video_thread.start()
    
    def update_medicine_label(self, count):
        """Update medicine validation label based on count"""
        if count >= 9:
            self.medicine_label.config(text="Valid Medicine", fg="#2ecc71")
        else:
            self.medicine_label.config(text="Invalid Medicine", fg="#e74c3c")
    
    def update_status_label(self, status):
        """Update status label"""
        if status == "Analyzing...":
            self.status_label.config(text=status, fg="#f39c12")
        elif status == "Error occurred":
            self.status_label.config(text=status, fg="#e74c3c")
        elif status == "":
            self.status_label.config(text="Ready for motion detection", fg="#2ecc71")
        else:
            self.status_label.config(text=status, fg="#3498db")
    
    def update_alternative_label(self, text):
        """Update alternative label with gemini output"""
        if text:
            # Format the text as a numbered list if it's not already formatted
            if not text.strip().startswith(('1.', '•', '-')):
                # Simple formatting - split by periods or newlines and create numbered list
                sentences = [s.strip() for s in text.replace('\n', '. ').split('.') if s.strip()]
                if sentences:
                    formatted_text = ""
                    for i, sentence in enumerate(sentences[:4], 1):  # Limit to 4 items
                        formatted_text += f"{i}. {sentence}\n"
                    self.alternative_label.config(text=formatted_text.strip())
                else:
                    self.alternative_label.config(text=text)
            else:
                self.alternative_label.config(text=text)
        else:
            self.alternative_label.config(text="1. ________\n2. ________\n3. ________\n4. ________")
    
    def draw_countdown_overlay(self, frame):
        """Draw countdown timer on the video frame"""
        if not self.countdown_active:
            return frame
        
        current_time = time.time()
        elapsed = current_time - self.countdown_start_time
        remaining = max(0, self.countdown_duration - elapsed)
        
        if remaining <= 0:
            self.countdown_active = False
            self.can_detect = True  # Re-enable detection when countdown finishes
            return frame
        
        # Create overlay
        overlay = frame.copy()
        
        # Calculate countdown number
        countdown_num = int(remaining) + 1
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Draw semi-transparent background circle
        center = (w // 2, h // 2)
        radius = min(w, h) // 8
        cv2.circle(overlay, center, radius, (200, 200, 200), -1)
        cv2.circle(overlay, center, radius, (0, 100, 200), 4)
        
        # Draw countdown text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w, h) / 200  # Scale font based on frame size
        thickness = max(2, int(min(w, h) / 200))
        
        text = str(countdown_num)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        
        # Draw text
        cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        # Draw "Next detection in:" text above
        info_text = "Next detection in:"
        info_font_scale = font_scale * 0.5
        info_thickness = max(1, thickness // 2)
        info_size = cv2.getTextSize(info_text, font, info_font_scale, info_thickness)[0]
        info_x = center[0] - info_size[0] // 2
        info_y = center[1] - radius - 20
        
        cv2.putText(overlay, info_text, (info_x, info_y), font, info_font_scale, (0, 0, 0), info_thickness)
        
        # Blend overlay with original frame
        alpha = 0.7
        blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return blended
    
    def process_detection(self):
        """Handle the entire detection pipeline in a separate thread"""
        def detection_pipeline():
            try:
                # Step 1: Wait 3 seconds
                time.sleep(3)
                
                # Step 2: Capture fresh frame after wait
                ret, fresh_frame = self.cap.read()
                if not ret:
                    print("Failed to capture fresh frame")
                    self.processing = False
                    self.can_detect = True
                    return
                
                # Apply contrast reduction to the frame used for detection
                fresh_frame = self.adjust_contrast_brightness(fresh_frame)
                
                # Step 3: Show analyzing status
                self.root.after(0, self.update_status_label, "Analyzing...")
                
                # Step 4: Run detection with fresh frame
                print("Running detection...")
                count = detection.detect(fresh_frame)
                print(f"Detection result: {count}")
                
                # Step 5: Update medicine label
                self.root.after(0, self.update_medicine_label, count)
                
                # Step 6: Get gemini alternative with fresh frame
                print("Getting gemini response...")
                text = gemini.get_alternative(fresh_frame)
                print(f"Gemini response: {text}")
                
                # Step 7: Update alternative label
                self.root.after(0, self.update_alternative_label, text)
                
                # Step 8: Reset status label
                self.root.after(0, self.update_status_label, "")
                
                # Step 9: Start countdown
                self.countdown_active = True
                self.countdown_start_time = time.time()
                
                # Mark processing as complete
                self.processing = False
                
            except Exception as e:
                print(f"Error in detection pipeline: {e}")
                # Reset everything on error
                self.root.after(0, self.update_status_label, "Error occurred")
                self.processing = False
                self.can_detect = True
        
        # Start the pipeline in a separate thread
        pipeline_thread = threading.Thread(target=detection_pipeline, daemon=True)
        pipeline_thread.start()
    
    def detect_motion(self, current_frame):
        """Detect motion between current and previous frame"""
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return False
        
        # Skip detection if we're processing or in countdown
        if self.processing or self.countdown_active or not self.can_detect:
            self.prev_frame = current_frame.copy()
            return False
        
        # Calculate absolute difference between frames
        diff = cv2.absdiff(self.prev_frame, current_frame)
        
        # Convert to grayscale
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate total white pixels (motion areas)
        motion_area = cv2.countNonZero(thresh)
        
        # Update previous frame
        self.prev_frame = current_frame.copy()
        
        # Debug: Print motion area to see if motion is being detected
        if motion_area > 1000:  # Lower threshold for debugging
            print(f"Motion area: {motion_area} (threshold: {self.motion_threshold})")
        
        # Check if motion exceeds threshold
        if motion_area > self.motion_threshold:
            print(f"Motion detected! Motion area: {motion_area}")
            
            # Start processing pipeline
            self.processing = True
            self.can_detect = False  # Disable detection during processing
            
            # Start the detection process
            self.process_detection()
            
            return True
        
        return False
    
    def update_frame(self):
        """Update video frame in a separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Apply contrast reduction to the display frame
                frame_adjusted = self.adjust_contrast_brightness(frame)
                
                # Detect motion (using adjusted frame for consistency)
                self.detect_motion(frame_adjusted)
                
                # Add countdown overlay if active
                frame_with_overlay = self.draw_countdown_overlay(frame_adjusted)
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update the label in main thread
                self.root.after(0, self.update_video_label, photo)
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)
    
    def update_video_label(self, photo):
        """Update video label with new frame"""
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # Keep a reference
    
    def on_closing(self):
        """Handle window close event"""
        self.running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = VideoStreamApp(root)
    
    # You can adjust these parameters:
    # Motion threshold (lower = more sensitive, higher = less sensitive)
    app.motion_threshold = 5000
    
    # Contrast and brightness adjustment
    app.contrast_alpha = 0.8  # Reduce contrast (0.0 = no contrast, 1.0 = normal, >1.0 = increased)
    app.brightness_beta = 50   # Adjust brightness (0 = normal, positive = brighter, negative = darker)
    
    root.mainloop()

if __name__ == "__main__":
    main()