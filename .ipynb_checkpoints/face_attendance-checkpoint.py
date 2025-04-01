import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime
import csv

# ------------------ Step 1: Load Known Faces and Save Encodings ------------------
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "datasets/attendance.csv"
ENCODINGS_FILE = "datasets/encodings.pkl"

known_face_encodings = []
known_face_names = []

def train_model():
    global known_face_encodings, known_face_names
    for filename in os.listdir(KNOWN_FACES_DIR):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]  # First face detected in the image
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Extract student name from filename

    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print("[INFO] Model trained and encodings saved successfully.")

def load_trained_data():
    global known_face_encodings, known_face_names
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
        print(f"[INFO] Encodings loaded successfully. Found {len(known_face_encodings)} faces.")
    else:
        train_model()

# ------------------ Step 2: Attendance Marking ------------------
def mark_attendance(name, student_id, status):
    with open(ATTENDANCE_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date = datetime.now().strftime("%Y-%m-%d")
        writer.writerow([name, student_id, date, now, status])

# ------------------ Step 3: Classroom Image Recognition ------------------
def recognize_faces_from_image(image_path):
    if not os.path.exists(image_path.strip()):
        print(f"[ERROR] Image not found: {image_path.strip()}")
        return

    image = cv2.imread(image_path.strip())
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detected_students = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        student_id = "-"

        if True in matches:
            matched_indexes = [i for i, match in enumerate(matches) if match]
            for matched_index in matched_indexes:
                if matched_index < len(known_face_names):
                    name = known_face_names[matched_index]
                    student_id = f"ID-{matched_index + 1}"  # Assigning ID dynamically
                    detected_students.append(name)
                    mark_attendance(name, student_id, "Present")

        # Draw Rectangle around detected face
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Mark absent students
    for student in known_face_names:
        if student not in detected_students:
            mark_attendance(student, "-", "Absent")

    cv2.imshow('Classroom Attendance System', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------ Step 4: Main Execution ------------------
if __name__ == "__main__":
    load_trained_data()
    print("[INFO] System Ready.")
    image_path = input("Enter the classroom image path: ").strip()
    recognize_faces_from_image(image_path)