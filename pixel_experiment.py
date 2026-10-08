import cv2
from ultralytics import YOLO

# Load your fine-tuned NIDAR model
model = YOLO(r"C:\my project works\droneclub\nidar\tsk2pxl\runs\detect\NIDAR_Vision\sard_finetune_v1-8\weights\best.pt")

# Load the high-res field image
# UPDATE THIS PATH to the exact image you just tested
original_img = cv2.imread(r"img\t3.jpg")

# Baseline calibration: Estimate the pixel height of the survivor in the unscaled image
original_target_height = 100

cv2.namedWindow("NIDAR Pixel Test")
# Generate the slider starting at 100px, dropping to the 20px limit
cv2.createTrackbar("Target Pixels", "NIDAR Pixel Test", 40,40, lambda x: None)
cv2.setTrackbarMin("Target Pixels", "NIDAR Pixel Test", 10)

while True:
    # 1. Read the current slider value
    current_pixels = cv2.getTrackbarPos("Target Pixels", "NIDAR Pixel Test")
    
    # 2. Calculate the dynamic scaling factor
    scale_factor = current_pixels / original_target_height
    
    # 3. Shrink the image to simulate higher drone altitude
    resized_img = cv2.resize(original_img, None, fx=scale_factor, fy=scale_factor)
    
    # 4. Run inference on the scaled-down image
    results = model(resized_img)
    annotated_frame = results[0].plot()
    
    # 5. Display the live result
    cv2.imshow("NIDAR Pixel Test", annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()