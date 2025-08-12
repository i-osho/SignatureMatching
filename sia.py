from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

root_path = r"C:/Users/osho/Documents/University/Sem 4/Internship/SignatureMatching"

def load_and_preprocess_image(img_path, image_size=(224, 224)):
    """Load and preprocess image for model input"""
    img = image.load_img(img_path, target_size=image_size, color_mode='grayscale')
    img = image.img_to_array(img) / 255.0  # Normalize image
    return img

def predict_signatures(model, img_path1, img_path2, image_size=(224, 224)):
    """Compare two signatures using model"""
    # Load and preprocess images
    img1 = load_and_preprocess_image(img_path1, image_size)
    img2 = load_and_preprocess_image(img_path2, image_size)
    
    # Expand dimensions to match input shape
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    
    # Make prediction
    prediction = model.predict([img1, img2])[0][0]  # best_sia.keras
    
    return prediction

class SignatureMatcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Matching Tool - best_sia.keras")
        self.root.geometry("800x600")
        
        # Load model
        self.load_model()
        
        # Initialize variables
        self.ref_img_path = None
        self.test_img_path = None
        
        self.setup_ui()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(f"{root_path}/models/best_sia.keras")
            print("Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def setup_ui(self):
        """Setup the GUI components"""
        # Main title
        title_label = tk.Label(self.root, text="Signature Matching", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Frame for buttons and images
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=True, fill="both", padx=20, pady=10)
        
        # Upload buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.btn_ref = tk.Button(button_frame, text="Select Reference Signature", 
                                command=self.load_ref_img, width=20)
        self.btn_ref.pack(side="left", padx=10)
        
        self.btn_test = tk.Button(button_frame, text="Select Test Signature", 
                                 command=self.load_test_img, width=20)
        self.btn_test.pack(side="left", padx=10)
        
        # Images display frame
        images_frame = tk.Frame(main_frame)
        images_frame.pack(pady=20, expand=True, fill="both")
        
        # Reference image frame
        ref_frame = tk.Frame(images_frame)
        ref_frame.pack(side="left", expand=True, fill="both", padx=10)
        
        tk.Label(ref_frame, text="Reference Signature", font=("Arial", 12, "bold")).pack()
        self.ref_display = tk.Label(ref_frame, text="No reference file selected", 
                                   bg="lightgray", width=30, height=15)
        self.ref_display.pack(expand=True, fill="both", pady=5)
        
        self.ref_label = tk.Label(ref_frame, text="", font=("Arial", 10))
        self.ref_label.pack()
        
        # Test image frame
        test_frame = tk.Frame(images_frame)
        test_frame.pack(side="right", expand=True, fill="both", padx=10)
        
        tk.Label(test_frame, text="Test Signature", font=("Arial", 12, "bold")).pack()
        self.test_display = tk.Label(test_frame, text="No test file selected", 
                                    bg="lightgray", width=30, height=15)
        self.test_display.pack(expand=True, fill="both", pady=5)
        
        self.test_label = tk.Label(test_frame, text="", font=("Arial", 10))
        self.test_label.pack()
        
        # Compare button
        self.compare_btn = tk.Button(main_frame, text="Compare Signatures",
                                    command=self.compare_signatures, state=tk.DISABLED,
                                    font=("Arial", 12, "bold"), width=20)
        self.compare_btn.pack(pady=20)
        
        # Result label
        self.result_label = tk.Label(main_frame, text="Similarity Score: ", 
                                    font=("Arial", 14, "bold"))
        self.result_label.pack(pady=10)
    
    def load_ref_img(self):
        """Select reference image"""
        path = filedialog.askopenfilename(
            title="Select Reference Signature",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            self.ref_img_path = path
            self.display_image(path, self.ref_display)
            self.ref_label.config(text=os.path.basename(path))
            self.check_ready()
    
    def load_test_img(self):
        """Select test image"""
        path = filedialog.askopenfilename(
            title="Select Test Signature",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            self.test_img_path = path
            self.display_image(path, self.test_display)
            self.test_label.config(text=os.path.basename(path))
            self.check_ready()
    
    def display_image(self, file_path, label):
        """Display image in the label"""
        try:
            image = Image.open(file_path)
            image = image.resize((250, 150), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label.config(image=photo, text="")
            label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def check_ready(self):
        """Enable compare button when both images are selected"""
        if self.ref_img_path and self.test_img_path:
            self.compare_btn.config(state=tk.NORMAL)
    
    def compare_signatures(self):
        """Compare the selected signatures"""
        if not self.ref_img_path or not self.test_img_path:
            messagebox.showwarning("Warning", "Please select both signature images!")
            return
        
        try:
            # Show progress
            self.result_label.config(text="Comparing signatures...", fg="blue")
            self.root.update()
            
            score = predict_signatures(self.model, self.ref_img_path, self.test_img_path)
            
            # Display result with prediction
            if score > 0.7:
                result_text = f"✅ Genuine - Score: {score:.4f}"
                color = "green"
            else:
                result_text = f"❌ Forged - Score: {score:.4f}"
                color = "red"
            
            self.result_label.config(text=result_text, fg=color)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compare signatures: {str(e)}")
            self.result_label.config(text="Comparison failed", fg="red")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = SignatureMatcherGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()


