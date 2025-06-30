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
        self.genuine_images = []
        self.forged_images = []
        
        # Load images from the hierarchical structure
        for root_dir in root_dirs:
            if not os.path.exists(root_dir):
                print(f"⚠️ Directory not found: {root_dir}")
                continue
            real_folder = os.path.join(root_dir, "real")
            
            forge_folder = os.path.join(root_dir, "forge")
            
            # Load genuine signatures
            if real_folder and os.path.exists(real_folder):
                for file in os.listdir(real_folder):
                    if file.endswith('.png'):
                        self.genuine_images.append(os.path.join(real_folder, file))
            
            # Load forged signatures
            if os.path.exists(forge_folder):
                for file in os.listdir(forge_folder):
                    if file.endswith('.png'):
                        self.forged_images.append(os.path.join(forge_folder, file))
        
        print(f"Found {len(self.genuine_images)} genuine signatures")
        print(f"Found {len(self.forged_images)} forged signatures")
        
        if len(self.genuine_images) == 0 or len(self.forged_images) == 0:
            raise RuntimeError("Need both genuine and forged signatures for training")
        
        # Generate pairs for training
        self.pairs = []
        self.labels = []
        
        # Create positive pairs (genuine-genuine)
        for i in range(len(self.genuine_images)):
            for j in range(i+1, min(i+6, len(self.genuine_images))):  # Limit to 5 pairs per image
                self.pairs.append((self.genuine_images[i], self.genuine_images[j]))
                self.labels.append(1)
        
        # Create negative pairs (genuine-forged)
        for genuine_img in self.genuine_images:
            # Sample 3-5 forged images per genuine image
            num_negatives = min(5, len(self.forged_images))
            selected_forged = random.sample(self.forged_images, num_negatives)
            for forged_img in selected_forged:
                self.pairs.append((genuine_img, forged_img))
                self.labels.append(0)
        
        print(f"Generated {len(self.pairs)} pairs")
        print(f"   - Genuine pairs: {sum(self.labels)}")
        print(f"   - Forged pairs: {len(self.labels) - sum(self.labels)}")
        
        # Train-test split
        indices = list(range(len(self.pairs)))
        random.shuffle(indices)
        split_idx = max(1, int(len(indices) * test_size))
        
        self.indices = indices[split_idx:] if train else indices[:split_idx]
        print(f"{'Train' if train else 'Validation'} samples: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        img1 = Image.open(self.pairs[index][0]).convert('L')
        img2 = Image.open(self.pairs[index][1]).convert('L')
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
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 1),
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
        
        # L1 distance
        distance = torch.abs(output1 - output2)
        prediction = self.classifier(distance)
        return prediction

# Training function
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
        
        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(img1)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.6f} Acc: {100. * correct / total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Average Loss: {avg_loss:.6f} Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# Validation function
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
    num_epochs = 50
    learning_rate = 0.0001
    model_path = None
    
    # Your dataset directories
    root_dirs = ["dataset1", "dataset2", "dataset3", "dataset4"]
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SignatureDataset(root_dirs, transform, train=True)
    val_dataset = SignatureDataset(root_dirs, transform, train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    model = SiameseNetwork().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    best_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step()
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            if model_path:
                os.remove(model_path)
            model_path = f"{best_accuracy:.2f}%-{batch_size}-{epoch}-{learning_rate}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved! Accuracy: {best_accuracy:.2f}%")
        
        print("-" * 60)
    
    print(f"Training complete! Best validation accuracy: {best_accuracy:.2f}%")
