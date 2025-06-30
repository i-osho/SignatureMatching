import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
# import matplotlib.pyplot as plt

class SignatureDataset(Dataset):
    def __init__(self, root_dirs, transform=None, train=True, test_size=0.2):
        self.root_dirs = root_dirs
        self.transform = transform
        self.writers_data = {}  # Dictionary to store signatures by writer
        
        # Load images organized by writer
        for root_dir in root_dirs:
            if not os.path.exists(root_dir):
                print(f"⚠️ Directory not found: {root_dir}")
                continue
                
            real_folder = os.path.join(root_dir, "real")
            forge_folder = os.path.join(root_dir, "forge")
            
            # Parse genuine signatures
            if os.path.exists(real_folder):
                for file in os.listdir(real_folder):
                    if file.endswith('.png'):
                        # Extract writer ID from filename
                        writer_id = self._extract_writer_id(file)
                        if writer_id not in self.writers_data:
                            self.writers_data[writer_id] = {'genuine': [], 'forged': []}
                        self.writers_data[writer_id]['genuine'].append(os.path.join(real_folder, file))
            
            # Parse forged signatures
            if os.path.exists(forge_folder):
                for file in os.listdir(forge_folder):
                    if file.endswith('.png'):
                        writer_id = self._extract_writer_id(file)
                        if writer_id not in self.writers_data:
                            self.writers_data[writer_id] = {'genuine': [], 'forged': []}
                        self.writers_data[writer_id]['forged'].append(os.path.join(forge_folder, file))
        
        # Print dataset statistics
        total_genuine = sum(len(data['genuine']) for data in self.writers_data.values())
        total_forged = sum(len(data['forged']) for data in self.writers_data.values())
        print(f"Found {len(self.writers_data)} writers")
        print(f"Total genuine signatures: {total_genuine}")
        print(f"Total forged signatures: {total_forged}")
        
        if len(self.writers_data) == 0:
            raise RuntimeError("No writers found in the dataset")
        
        # Generate pairs for training
        self.pairs = []
        self.labels = []
        
        for writer_id, data in self.writers_data.items():
            genuine_sigs = data['genuine']
            forged_sigs = data['forged']
            
            # Create positive pairs (genuine-genuine) within same writer
            for i in range(len(genuine_sigs)):
                for j in range(i+1, len(genuine_sigs)):
                    self.pairs.append((genuine_sigs[i], genuine_sigs[j]))
                    self.labels.append(1)
            
            # Create negative pairs (genuine-forged) within same writer
            for genuine_sig in genuine_sigs:
                for forged_sig in forged_sigs:
                    self.pairs.append((genuine_sig, forged_sig))
                    self.labels.append(0)
        
        # Add cross-writer negative pairs (genuine from writer A vs genuine from writer B)
        writer_ids = list(self.writers_data.keys())
        for i, writer_a in enumerate(writer_ids):
            for j, writer_b in enumerate(writer_ids[i+1:], i+1):
                genuine_a = self.writers_data[writer_a]['genuine']
                genuine_b = self.writers_data[writer_b]['genuine']
                
                # Add 3 random cross-writer pairs to avoid explosion
                for _ in range(min(3, len(genuine_a), len(genuine_b))):
                    sig_a = random.choice(genuine_a)
                    sig_b = random.choice(genuine_b)
                    self.pairs.append((sig_a, sig_b))
                    self.labels.append(0)
        
        print(f"Generated {len(self.pairs)} pairs")
        print(f"   - Positive pairs (same writer genuine): {sum(self.labels)}")
        print(f"   - Negative pairs (forged/different writers): {len(self.labels) - sum(self.labels)}")
        
        # Train-test split
        indices = list(range(len(self.pairs)))
        random.shuffle(indices)
        split_idx = max(1, int(len(indices) * test_size))
        
        self.indices = indices[split_idx:] if train else indices[:split_idx]
        print(f"{'Train' if train else 'Validation'} samples: {len(self.indices)}")
    
    def _extract_writer_id(self, filename):
        """Extract writer ID from filename based on common naming conventions."""
        # Remove file extension
        base_name = filename.split('.')[0]
        
        # Method 1: Check for underscore separated format (e.g., "001_01.png" or "writer_001_01.png")
        if '_' in base_name:
            parts = base_name.split('_')
            for part in parts:
                if part.isdigit() and len(part) >= 2:
                    return int(part)
        
        # Method 2: Extract first few digits (e.g., "00101.png" -> writer 001)
        digits = ''.join(c for c in base_name if c.isdigit())
        if len(digits) >= 3:
            return int(digits[:3])
        elif len(digits) >= 2:
            return int(digits[:2])
        elif len(digits) >= 1:
            return int(digits[0])
        
        # Fallback: use hash of filename for consistency
        return hash(base_name) % 1000
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        img1_path, img2_path = self.pairs[index]
        
        try:
            img1 = Image.open(img1_path).convert('L')
            img2 = Image.open(img2_path).convert('L')
        except Exception as e:
            print(f"Error loading images: {img1_path}, {img2_path}")
            print(f"Error: {e}")
            # Return a dummy pair in case of error
            img1 = Image.new('L', (128, 128), 255)
            img2 = Image.new('L', (128, 128), 255)
        
        label = self.labels[index]
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

# Siamese Network Architecture
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 96, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Fourth convolutional block
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Fifth convolutional block
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward_one(self, x):
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        
        # L1 distance (Manhattan distance)
        distance = torch.abs(output1 - output2)
        prediction = self.classifier(distance)
        return prediction

# Training function with improved logging
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (img1, img2, labels) in enumerate(train_loader):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(img1, img2).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(img1)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)] '
                  f'Loss: {loss.item():.6f} Acc: {100. * correct / total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Average Loss: {avg_loss:.6f} Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# Enhanced validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    print(f'Validation Loss: {avg_loss:.6f} Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# Prediction function
def predict(model, ref_path, test_path, transform, device):
    model.eval()
    
    ref_img = Image.open(ref_path).convert('L')
    test_img = Image.open(test_path).convert('L')
    
    ref_img = transform(ref_img).unsqueeze(0).to(device)
    test_img = transform(test_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(ref_img, test_img).item()
    
    return output

# Main execution
if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.00005
    model_path = None
    
    # Your dataset directories
    root_dirs = ["dataset5"]
    
    # Enhanced transformations with data augmentation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(degrees=5),  # Small rotation for robustness
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SignatureDataset(root_dirs, transform, train=True, test_size=0.15)
    val_dataset = SignatureDataset(root_dirs, transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]), train=False, test_size=0.15)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = SiameseNetwork().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    patience_counter = 0
    max_patience = 15
    
    print(f"Starting training for {num_epochs} epochs...")
    print("-" * 80)
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(val_accuracy)
        
        # Save best model with detailed naming
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            if model_path and os.path.exists(model_path):
                os.remove(model_path)
            model_path = f"{best_accuracy:.2f}%-epoch{epoch}-bs{batch_size}-lr{learning_rate}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✅ New best model saved! Accuracy: {best_accuracy:.2f}% at epoch {epoch}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {max_patience} epochs without improvement")
            break
        
        print("-" * 80)
    
    print(f"🎉 Training complete! Best validation accuracy: {best_accuracy:.2f}%")
    print(f"📁 Best model saved as: {model_path}")
