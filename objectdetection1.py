import tkinter as tk
from tkinter import ttk, messagebox
from ultralytics import YOLO
import cv2
import serial
import serial.tools.list_ports
import threading
import time
from PIL import Image, ImageTk

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection with Arduino Control")
        self.root.geometry("800x600")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize variables
        self.arduino = None
        self.is_connected = False
        self.detection_active = False
        self.model = None
        self.cap = None
        self.detection_thread = None
        
        # Load YOLO model
        try:
            self.model = YOLO("yolov8n.pt")
            print("YOLO model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
            return
        
        # Create GUI
        self.create_widgets()
        
        # Auto-detect and connect to Arduino
        self.auto_connect_arduino()
    
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Object Detection Arduino Controller", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2b2b2b')
        title_label.pack(pady=(0, 20))
        
        # Arduino connection status frame
        status_frame = tk.Frame(main_frame, bg='#2b2b2b')
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(status_frame, text="Arduino Status:", font=('Arial', 12, 'bold'), 
                fg='white', bg='#2b2b2b').pack(side=tk.LEFT)
        
        self.status_label = tk.Label(status_frame, text="Disconnected", 
                                   font=('Arial', 12), fg='red', bg='#2b2b2b')
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Port selection frame
        port_frame = tk.Frame(main_frame, bg='#2b2b2b')
        port_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(port_frame, text="Serial Port:", font=('Arial', 10), 
                fg='white', bg='#2b2b2b').pack(side=tk.LEFT)
        
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, 
                                      values=self.get_serial_ports(), width=15)
        self.port_combo.pack(side=tk.LEFT, padx=(10, 10))
        
        connect_btn = tk.Button(port_frame, text="Connect", command=self.connect_arduino,
                               bg='#4a90e2', fg='white', font=('Arial', 10))
        connect_btn.pack(side=tk.LEFT)
        
        refresh_btn = tk.Button(port_frame, text="Refresh Ports", command=self.refresh_ports,
                               bg='#6c757d', fg='white', font=('Arial', 10))
        refresh_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Object detection control frame
        control_frame = tk.Frame(main_frame, bg='#2b2b2b')
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(control_frame, text="Object Detection:", font=('Arial', 12, 'bold'), 
                fg='white', bg='#2b2b2b').pack(anchor=tk.W)
        
        self.detection_status = tk.Label(control_frame, text="Stopped", 
                                       font=('Arial', 11), fg='orange', bg='#2b2b2b')
        self.detection_status.pack(anchor=tk.W, pady=(5, 10))
        
        # Button frame
        button_frame = tk.Frame(control_frame, bg='#2b2b2b')
        button_frame.pack(fill=tk.X)
        
        self.start_btn = tk.Button(button_frame, text="Start Detection", 
                                  command=self.start_detection,
                                  bg='#28a745', fg='white', font=('Arial', 10, 'bold'),
                                  padx=20, pady=8)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.reset_btn = tk.Button(button_frame, text="Send 0 (Reset)", 
                                  command=self.send_zero,
                                  bg='#dc3545', fg='white', font=('Arial', 10, 'bold'),
                                  padx=20, pady=8, state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT)
        
        # Log frame
        log_frame = tk.Frame(main_frame, bg='#2b2b2b')
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        tk.Label(log_frame, text="Activity Log:", font=('Arial', 12, 'bold'), 
                fg='white', bg='#2b2b2b').pack(anchor=tk.W)
        
        # Log text area with scrollbar
        log_container = tk.Frame(log_frame, bg='#2b2b2b')
        log_container.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.log_text = tk.Text(log_container, bg='#1a1a1a', fg='white', 
                               font=('Consolas', 9), wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(log_container, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def get_serial_ports(self):
        """Get list of available serial ports"""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    
    def refresh_ports(self):
        """Refresh the list of available serial ports"""
        ports = self.get_serial_ports()
        self.port_combo['values'] = ports
        self.log_message("Serial ports refreshed")
    
    def auto_connect_arduino(self):
        """Try to automatically connect to Arduino"""
        ports = self.get_serial_ports()
        for port in ports:
            if self.try_connect_port(port):
                self.port_var.set(port)
                break
    
    def try_connect_port(self, port):
        """Try to connect to a specific port"""
        try:
            test_arduino = serial.Serial(port, 9600, timeout=2)
            time.sleep(2)  # Wait for Arduino to initialize
            test_arduino.close()
            return True
        except:
            return False
    
    def connect_arduino(self):
        """Connect to Arduino on selected port"""
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("Warning", "Please select a serial port")
            return
        
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.close()
            
            self.arduino = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            
            self.is_connected = True
            self.status_label.config(text="Connected", fg='green')
            self.log_message(f"Connected to Arduino on {port}")
            
            # Enable buttons
            self.start_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.is_connected = False
            self.status_label.config(text="Connection Failed", fg='red')
            self.log_message(f"Failed to connect to {port}: {e}")
            messagebox.showerror("Connection Error", f"Failed to connect to {port}\n{e}")
    
    def start_detection(self):
        """Start object detection"""
        if not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to Arduino first")
            return
        
        if self.detection_active:
            return
        
        self.detection_active = True
        self.detection_status.config(text="Running...", fg='green')
        self.start_btn.config(state=tk.DISABLED)
        self.log_message("Object detection started")
        
        # Start detection in separate thread
        self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
        self.detection_thread.start()
    
    def run_detection(self):
        """Run object detection in separate thread"""
        try:
            self.cap = cv2.VideoCapture(0)
            
            while self.detection_active:
                ret, frame = self.cap.read()
                if not ret:
                    self.log_message("Failed to read from camera")
                    break
                
                # Run YOLO detection
                results = self.model(frame)[0]
                
                # Check for objects
                object_detected = False
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls_id = result
                    if conf > 0.5:  # Confidence threshold
                        class_name = self.model.names[int(cls_id)]
                        object_detected = True
                        self.log_message(f"Object detected: {class_name} (confidence: {conf:.2f})")
                        break
                
                if object_detected:
                    self.send_one()
                    break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except Exception as e:
            self.log_message(f"Detection error: {e}")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
    
    def send_one(self):
        """Send '1' to Arduino and stop detection"""
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(b'1')
                self.log_message("Sent '1' to Arduino - LED should turn ON")
                
                # Stop detection
                self.detection_active = False
                self.detection_status.config(text="Object Found - Stopped", fg='orange')
                self.reset_btn.config(state=tk.NORMAL)
                
            except Exception as e:
                self.log_message(f"Failed to send '1': {e}")
    
    def send_zero(self):
        """Send '0' to Arduino and restart detection"""
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(b'0')
                self.log_message("Sent '0' to Arduino - LED should turn OFF")
                
                # Re-enable start button
                self.start_btn.config(state=tk.NORMAL)
                self.reset_btn.config(state=tk.DISABLED)
                self.detection_status.config(text="Stopped", fg='orange')
                
            except Exception as e:
                self.log_message(f"Failed to send '0': {e}")
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Also print to console
        print(formatted_message.strip())
    
    def on_closing(self):
        """Handle application closing"""
        self.detection_active = False
        
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.root.destroy()

# ----------------- Main Application ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
