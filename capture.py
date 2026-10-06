import cv2
import os
import json
import sys
from datetime import datetime


# --------------------------------------------------
# 1. Read class name from command line
# --------------------------------------------------

if len(sys.argv) != 2:
    print("Usage:")
    print("python capture.py <class_name>")
    print()
    print("Example:")
    print("python capture.py navodhya")
    sys.exit(1)

class_name = sys.argv[1]


# --------------------------------------------------
# 2. Read valid classes from config.json
# --------------------------------------------------

with open("config.json", "r") as f:
    config = json.load(f)

class_names = config["class_names"]

if class_name not in class_names:
    print(f"Invalid class: {class_name}")
    print(f"Available classes: {class_names}")
    sys.exit(1)


# --------------------------------------------------
# 3. Create output folder
# --------------------------------------------------

output_dir = os.path.join("raw", class_name)

os.makedirs(output_dir, exist_ok=True)

print(f"Saving images to: {output_dir}")


# --------------------------------------------------
# 4. Load Haar Cascade face detector
# --------------------------------------------------

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error: Could not load Haar Cascade.")
    sys.exit(1)


# --------------------------------------------------
# 5. Open webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)


# --------------------------------------------------
# 6. Image counter
# --------------------------------------------------

count = 0

print()
print("Controls:")
print("SPACE -> Capture face")
print("q     -> Quit")
print()


# --------------------------------------------------
# 7. Main webcam loop
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break


    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------
    # Detect faces
    # --------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )


    # --------------------------------------------------
    # Draw rectangles around detected faces
    # --------------------------------------------------

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


    # --------------------------------------------------
    # Display image count
    # --------------------------------------------------

    cv2.putText(
        frame,
        f"Class: {class_name}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Images captured: {count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Show message when no face is detected
    if len(faces) == 0:

        cv2.putText(
            frame,
            "No face detected",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    # --------------------------------------------------
    # Show webcam window
    # --------------------------------------------------

    cv2.imshow(
        "Find My Friend - Image Capture",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------
    # Press SPACE to capture face
    # --------------------------------------------------

    if key == 32:

        if len(faces) == 0:

            print("No face detected. Image not saved.")

        else:

            # Use first detected face
            x, y, w, h = faces[0]


            # ------------------------------------------
            # Add 20% padding
            # ------------------------------------------

            padding_x = int(w * 0.20)
            padding_y = int(h * 0.20)


            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)

            x2 = min(
                frame.shape[1],
                x + w + padding_x
            )

            y2 = min(
                frame.shape[0],
                y + h + padding_y
            )


            # Crop face
            face_crop = frame[
                y1:y2,
                x1:x2
            ]


            # ------------------------------------------
            # Generate filename
            # ------------------------------------------

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            count += 1

            filename = (
                f"{class_name}_"
                f"{timestamp}_"
                f"{count}.jpg"
            )

            filepath = os.path.join(
                output_dir,
                filename
            )


            # ------------------------------------------
            # Save image
            # ------------------------------------------

            cv2.imwrite(
                filepath,
                face_crop
            )

            print(
                f"Saved image {count}: "
                f"{filepath}"
            )


    # --------------------------------------------------
    # Press q to quit
    # --------------------------------------------------

    elif key == ord("q"):
        break


# --------------------------------------------------
# 8. Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print()
print(f"Finished.")
print(f"Total images captured: {count}")
