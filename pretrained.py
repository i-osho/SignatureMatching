import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.spatial.distance import cosine
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore', category=FutureWarning)


# Define a simple Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)

    def forward(self, x):
        return self.resnet(x)

class SignatureComparator:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Matching")
        self.root.geometry("800x600")
        
        # Initialize model
        self.model = SiameseNetwork()
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.image1 = None
        self.image2 = None
        self.image1_path = None
        self.image2_path = None
        
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
        
        self.upload_btn1 = tk.Button(button_frame, text="Upload Signature 1", 
                                    command=self.upload_image1, width=15)
        self.upload_btn1.pack(side="left", padx=10)
        
        self.upload_btn2 = tk.Button(button_frame, text="Upload Signature 2", 
                                    command=self.upload_image2, width=15)
        self.upload_btn2.pack(side="left", padx=10)
        
        # Images display frame
        images_frame = tk.Frame(main_frame)
        images_frame.pack(pady=20, expand=True, fill="both")
        
        # Image 1 frame
        img1_frame = tk.Frame(images_frame)
        img1_frame.pack(side="left", expand=True, fill="both", padx=10)
        
        tk.Label(img1_frame, text="Signature 1", font=("Arial", 12, "bold")).pack()
        self.img1_label = tk.Label(img1_frame, text="No image selected", 
                                  bg="lightgray", width=30, height=15)
        self.img1_label.pack(expand=True, fill="both", pady=5)
        
        self.img1_filename = tk.Label(img1_frame, text="", font=("Arial", 10))
        self.img1_filename.pack()
        
        # Image 2 frame
        img2_frame = tk.Frame(images_frame)
        img2_frame.pack(side="right", expand=True, fill="both", padx=10)
        
        tk.Label(img2_frame, text="Signature 2", font=("Arial", 12, "bold")).pack()
        self.img2_label = tk.Label(img2_frame, text="No image selected", 
                                  bg="lightgray", width=30, height=15)
        self.img2_label.pack(expand=True, fill="both", pady=5)
        
        self.img2_filename = tk.Label(img2_frame, text="", font=("Arial", 10))
        self.img2_filename.pack()
        
        # Compare button - initially disabled
        self.compare_btn = tk.Button(main_frame, text="Compare Signatures", 
                                    command=self.compare_signatures, state=tk.DISABLED,
                                    font=("Arial", 12, "bold"), width=20)
        self.compare_btn.pack(pady=20)
        
        # Result label
        self.result_label = tk.Label(main_frame, text="", font=("Arial", 14, "bold"))
        self.result_label.pack(pady=10)

    def upload_image1(self):
        file_path = filedialog.askopenfilename(
            title="Select Signature 1",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_path:
            self.image1_path = file_path
            self.display_image(file_path, self.img1_label, "Signature 1")
            self.img1_filename.config(text=os.path.basename(file_path))
            self.check_ready()

    def upload_image2(self):
        file_path = filedialog.askopenfilename(
            title="Select Signature 2",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_path:
            self.image2_path = file_path
            self.display_image(file_path, self.img2_label, "Signature 2")
            self.img2_filename.config(text=os.path.basename(file_path))
            self.check_ready()

    def check_ready(self):
        if self.image1_path and self.image2_path:
            self.compare_btn.config(state=tk.NORMAL)

    def display_image(self, file_path, label, title):
        try:
            image = Image.open(file_path)
            image = image.resize((250, 150), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label.config(image=photo, text="")
            label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def get_image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(image)
        return features.squeeze().cpu().numpy()
    
    def compare_signatures(self):
        try:
            # Show progress
            self.result_label.config(text="Comparing signatures...", fg="blue")
            self.root.update()
            
            # Get embeddings
            embedding1 = self.get_image_embedding(self.image1_path)
            embedding2 = self.get_image_embedding(self.image2_path)
            
            # Calculate similarity
            similarity = 1 - cosine(embedding1, embedding2)
            
            # Display result
            self.result_label.config(text=f"Similarity Score: {similarity:.3f}")
            self.result_label.config(fg="green")
                
        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed: {str(e)}")
            self.result_label.config(text="Comparison failed", fg="red")

def main():
    root = tk.Tk()
    app = SignatureComparator(root)
    root.mainloop()

if __name__ == "__main__":
    main()