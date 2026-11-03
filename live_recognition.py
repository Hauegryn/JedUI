import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

# 1. Define your class labels (update these to match your actual gesture folders!)
# Example: If folder names in preprocessed_dataset are 'pointing', 'clicking', 'idle', etc.
class_names = ["Click_OK", "Pointing_Cursor", "Thumbs_Down", "Thumbs_Up"]

# 2. Set up device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3. Re-initialize the model architecture and match the final classification layer
model = models.mobilenet_v3_large(weights=None)  # No need to download ImageNet weights again
model.classifier[3] = nn.Linear(in_features=1280, out_features=len(class_names))

# 4. Load your trained weights from Step 3
model.load_state_dict(torch.load("gesture_model.pth", map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode (disables dropout/batchnorm updates)

# 5. Define the exact same image transformations used during training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 6. Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("--- Live Gesture Recognition Started ---")
print("Press 'q' to quit.")

with torch.no_grad():  # Disable gradient tracking for faster inference
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # OpenCV reads frames in BGR format. Convert to RGB for the model transform
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transformations and add a batch dimension [1, 3, 224, 224]
        input_tensor = transform(rgb_frame).unsqueeze(0).to(device)

        # Perform inference
        outputs = model(input_tensor)
        
        # Convert raw logits to probabilities using Softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the highest confidence score and its corresponding class index
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        gesture_name = class_names[predicted_idx.item()]
        conf_score = confidence.item() * 100

        # Draw the prediction text and confidence on the live video window
        text = f"Gesture: {gesture_name} ({conf_score:.1f}%)"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the live feed window
        cv2.imshow("Kiosk Live Gesture Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up resources
cap.release()
cv2.destroyAllWindows()