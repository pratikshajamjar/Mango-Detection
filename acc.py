import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Prompt user for image input
image_path = input("Enter the path of the image: ")

# Check if the file exists
if not os.path.exists(image_path):
    print("Error: The specified file does not exist!")
    exit()

# Load YOLO model
model_path = "best.pt"  # Default trained model
if not os.path.exists(model_path):
    print("Error: Model file not found!")
    exit()

model = YOLO(model_path)

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image! Check the file path.")
    exit()

# Perform object detection
results = model.predict(image)

# Load class names if available
class_list = []
if os.path.exists("coco.txt"):
    with open("coco.txt", "r") as my_file:
        class_list = my_file.read().strip().split("\n")

# Dictionary to store object counts
object_counts = {}

# Define class-specific colors
color_map = {
    "ripe": (0, 255, 255),      # Yellow
    "unripe": (0, 255, 0),      # Green
    "overripe": (42, 42, 165),  # Brown (BGR format)
    "default": (255, 0, 0)      # Default blue for unknown classes
}

# Process results and draw bounding boxes
total_objects = 0
correct_predictions = 0  # For accuracy calculation

# Sample ground truth labels for evaluation (replace with real values)
ground_truth_labels = ["ripe", "unripe", "overripe"]  # Modify as needed

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  
        conf = float(box.conf[0].item())  
        cls = int(box.cls[0].item())  
        
        # Get class name if available
        class_name = class_list[cls] if 0 <= cls < len(class_list) else f"Unknown-{cls}"

        object_counts[class_name] = object_counts.get(class_name, 0) + 1
        total_objects += 1

        # Check if the prediction is correct
        if class_name in ground_truth_labels:
            correct_predictions += 1

        # Select color for the text based on the class
        text_color = color_map.get(class_name.lower(), color_map["default"])

        # Draw bounding boxes and labels (Reduced font size)
        label = f"{class_name} ({conf:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

# Calculate Accuracy
accuracy = (correct_predictions / total_objects) * 100 if total_objects > 0 else 0
print(f"Model Accuracy: {accuracy:.2f}%")

# Display object count on the image
text_x, text_y = 10, 20
for obj, count in object_counts.items():
    text = f"{obj}: {count}"
    
    # Select color for count text
    text_color = color_map.get(obj.lower(), color_map["default"])
    
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    text_y += 20  

# Add total objects and accuracy
cv2.putText(image, f"Total objects: {total_objects}", (text_x, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
text_y += 20
cv2.putText(image, f"Accuracy: {accuracy:.2f}%", (text_x, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Display the image (Use OpenCV if GUI is available, else use Matplotlib)
try:
    cv2.imshow('YOLO Object Detection', image)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()
except Exception:
    print("OpenCV display not available. Using Matplotlib instead.")

    # Convert BGR (OpenCV format) to RGB for correct Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image_rgb)
    plt.axis("off")  # Hide axis
    plt.show()
