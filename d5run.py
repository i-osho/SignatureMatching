import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from dataset5 import SiameseNetwork, predict
import torchvision.transforms as transforms
import warnings
import os
import glob

model_path = r"94.00%-epoch4-bs16-lr0.0001.pth"

warnings.filterwarnings('ignore', category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

class SignatureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Matching - {}".format(model_path))
        self.root.geometry("800x600")
        self.ref_img_path = None
        self.test_img_path = None
        self.ref_img_tk = None
        self.test_img_tk = None

        self.setup_ui()

    def setup_ui(self):
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
                                    command=self.compare, state=tk.DISABLED,
                                    font=("Arial", 12, "bold"), width=20)
        self.compare_btn.pack(pady=20)
        
        # Result label
        self.result_label = tk.Label(main_frame, text="Similarity Score: ", 
                                    font=("Arial", 14, "bold"))
        self.result_label.pack(pady=10)

    def load_ref_img(self):
        path = filedialog.askopenfilename(title="Select Reference Signature", 
                                         filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.ref_img_path = path
            self.display_image(path, self.ref_display, "Reference Signature")
            self.ref_label.config(text=os.path.basename(path))
            self.check_ready()

    def load_test_img(self):
        path = filedialog.askopenfilename(title="Select Test Signature", 
                                         filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.test_img_path = path
            self.display_image(path, self.test_display, "Test Signature")
            self.test_label.config(text=os.path.basename(path))
            self.check_ready()

    def display_image(self, file_path, label, title):
        try:
            image = Image.open(file_path)
            image = image.resize((250, 150), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label.config(image=photo, text="")
            label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def check_ready(self):
        if self.ref_img_path and self.test_img_path:
            self.compare_btn.config(state=tk.NORMAL)

    def compare(self):
        if not self.ref_img_path or not self.test_img_path:
            messagebox.showwarning("Warning", "Please upload both signature images.")
            return
            
        try:
            # Show progress
            self.result_label.config(text="Comparing signatures...", fg="blue")
            self.root.update()
            
            score = predict(model, self.ref_img_path, self.test_img_path, transform, device)
            
            # Display result with color coding
            self.result_label.config(text=f"Similarity Score: {score:.3f}")
            self.result_label.config(fg="green")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compare signatures:\n{e}")
            self.result_label.config(text="Comparison failed", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureApp(root)
    root.mainloop()