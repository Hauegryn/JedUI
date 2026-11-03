import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# 1. Load MobileNetV3 models pretrained on ImageNet
mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
mobilenet_v3_small = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# Note: torchvision updated 'pretrained=True' to 'weights=...' in newer versions, 
# but if your version supports pretrained=True, that works too.

# 2. Modify the final layer for your custom number of classes (e.g., 10 gestures)
mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=4)

# 3. Define data transformations matching our preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),          # Randomly flips images left-to-right
    transforms.RandomRotation(degrees=15),           # Tilts the hand slightly
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Simulates restaurant lighting changes
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Load dataset using raw string 'r' to prevent escape character bugs
# (Remember: preprocessed_dataset must contain subfolders for each gesture class!)
train_dataset = datasets.ImageFolder(
    root=r'C:\Users\starm\Desktop\UNI\Semester Projectы\SPRO3\preprocessed_dataset', 
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenet_v3_large.parameters(), lr=0.001)

# 6. Training loop
num_epochs = 10
mobilenet_v3_large.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = mobilenet_v3_large(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Fixed print statement (kept on a single line)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Optional: Save your trained model weights for later live recognition
torch.save(mobilenet_v3_large.state_dict(), "gesture_model.pth")
print("Model training complete and weights saved!")