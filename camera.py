import cv2
import os

# 1. Create a folder to store our captured dataset images
output_dir = "gesture_dataset"
os.makedirs(output_dir, exist_ok=True)

# 2. Initialize the camera 
# (0 represents the default connected camera, like your USB webcam or Pi Camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("--- Data Collection Started ---")
print("Press 's' to save an image frame.")
print("Press 'q' to quit the program.")

img_count = 0

while True:
    # Read a frame from the camera feed
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the live video feed in a window
    cv2.imshow("Kiosk Gesture Capture", frame)

    # Wait for a key press (1 millisecond delay)
    key = cv2.waitKey(1) & 0xFF

    # If user presses 's', save the current frame as an image file
    if key == ord('s'):
        filename = os.path.join(output_dir, f"sample_{img_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        img_count += 1

    # If user presses 'q', break out of the loop and close
    elif key == ord('q'):
        print("Exiting data capture...")
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
