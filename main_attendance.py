import cv2
import face_recognition
import os
import pickle
from datetime import datetime
import csv

# Configuration - ADDED ATTENDANCE_DIR
KNOWN_FACES_DIR = "attendance_system/known_faces"
ATTENDANCE_DIR = "attendance_system/daily_attendance"  # New directory for daily records
ATTENDANCE_FILE = "attendance.csv"   # This is now redundant but kept for compatibility
ENCODINGS_FILE = "face_encodings.pkl"
TOLERANCE = 0.55

os.makedirs(ATTENDANCE_DIR, exist_ok=True)

def capture_image():
    """Capture image from webcam"""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Error: Could not open camera")
        return None
    
    print("\nPress SPACE to capture or ESC to cancel")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Error: Failed to capture image")
            break
            
        cv2.imshow("Camera - Press SPACE to capture", frame)
        key = cv2.waitKey(1)
        
        if key % 256 == 27:  # ESC pressed
            print("Capture cancelled")
            frame = None
            break
        elif key % 256 == 32:  # SPACE pressed
            # Save the captured image
            timestamp = datetime.now().strftime("%Y-%m-%d")
            image_path = os.path.join(ATTENDANCE_DIR, f"captured_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"✅ Image captured and saved to: {image_path}")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    return image_path if frame is not None else None

def get_image_path():
    """Get image path through user input"""
    while True:
        print("\nOptions:")
        print("1. Capture image from camera")
        print("2. Enter path to existing image")
        print("3. Exit")
        
        choice = input("Select option (1/2/3): ").strip()
        
        if choice == '1':
            image_path = capture_image()
            if image_path:
                return image_path
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                return image_path
            print("❌ Error: File not found")
        elif choice == '3':
            return None
        else:
            print("❌ Invalid option")

def mark_attendance(image_path):
    """Enhanced attendance marking with validation"""
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("❌ Error: Model not trained! Run training first.")
        return

    image = face_recognition.load_image_file(image_path)
    input_encodings = face_recognition.face_encodings(image)
    input_locations = face_recognition.face_locations(image)
    
    # Initialize attendance with all students absent
    attendance = {student["id"]: {"name": student["name"], "status": "Absent"}
                for student in data["metadata"]}

    # Store recognition results for display
    recognition_results = []
    
    # Recognition logic
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for encoding, location in zip(input_encodings, input_locations):
        distances = face_recognition.face_distance(data["encodings"], encoding)
        best_match = distances.argmin()
        
        if distances[best_match] <= TOLERANCE:
            student_id = data["metadata"][best_match]["id"]
            student_name = data["metadata"][best_match]["name"]
            attendance[student_id]["status"] = "Present"
            recognition_results.append((location, student_name, student_id))
        else:
            recognition_results.append((location, "Unknown", ""))

    # Save results to daily file
    daily_file = get_daily_filename()
    file_exists = os.path.isfile(daily_file)
    
    with open(daily_file, "a", newline="") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(["Name", "ID", "Status", "Timestamp"])
        
        for student_id, data in attendance.items():
            writer.writerow([
                data["name"],
                student_id,
                data["status"],
                timestamp
            ])

    display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Store used label positions to prevent overlap
    used_label_positions = []

    # Constants for label appearance
    FONT_SCALE = 1.7
    FONT_THICKNESS = 2
    LABEL_PADDING = 15
    LABEL_MARGIN = 10  # Minimum space between labels

    for (top, right, bottom, left), name, student_id in recognition_results:
        # Create label text
        label = f"ID:{student_id}" if student_id else "NA"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_DUPLEX
        (text_width, text_height), _ = cv2.getTextSize(label, font, FONT_SCALE, FONT_THICKNESS)
        
        # Calculate initial label position (below face)
        label_left = left
        label_top = bottom + LABEL_MARGIN
        label_right = label_left + text_width + LABEL_PADDING*2
        label_bottom = label_top + text_height + LABEL_PADDING
        
        # Adjust position if overlaps with existing labels
        for _ in range(3):  # Try 3 different positions
            overlap = False
            for (used_left, used_top, used_right, used_bottom) in used_label_positions:
                if not (label_right < used_left or 
                        label_left > used_right or 
                        label_bottom < used_top or 
                        label_top > used_bottom):
                    overlap = True
                    # Move label down if overlapping
                    label_top += text_height + LABEL_MARGIN
                    label_bottom += text_height + LABEL_MARGIN
                    break
            
            if not overlap:
                break
        
        # Draw face bounding box (thicker)
        cv2.rectangle(display_image, (left, top), (right, bottom), (0, 255, 0), 3)
        
        # Draw label background
        cv2.rectangle(display_image,
                    (label_left, label_top),
                    (label_right, label_bottom),
                    (0, 0, 0), cv2.FILLED)
        
        # Draw label text (with outline for better visibility)
        text_x = label_left + LABEL_PADDING
        text_y = label_bottom - LABEL_PADDING
        
        # Black outline
        cv2.putText(display_image, label, (text_x, text_y),
                font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS + 1)
        # White text
        cv2.putText(display_image, label, (text_x, text_y),
                font, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        
        # Remember this label's position
        used_label_positions.append((label_left, label_top, label_right, label_bottom))

    # Resize for display while maintaining aspect ratio
    display_height = 900
    height, width = display_image.shape[:2]
    if height > display_height:
        ratio = display_height / height
        display_image = cv2.resize(display_image, (int(width * ratio), display_height))

    # Show results
    cv2.imshow("Attendance Results", display_image)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("Attendance Results", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

def get_daily_filename():
    """Generate filename for daily attendance"""  
    today_date = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today_date}.csv")

def validate_filename(filename):
    """Ensure filename follows Name_ID format with numeric ID"""
    try:
        name_id, ext = os.path.splitext(filename)
        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
            return False, "Invalid file extension"
            
        if name_id.count('_') != 1:
            return False, "Filename must contain exactly one underscore"
            
        name, student_id = name_id.split('_')
        if not student_id.isdigit():
            return False, "ID must be numeric"
            
        return True, ""
    except Exception as e:
        return False, str(e)

def train_model():
    """Enhanced training with detailed validation"""
    known_encodings = []
    known_metadata = []
    skipped_files = []

    print("\n🔧 Training Model 🔧")
    print("-------------------")
    
    for filename in os.listdir(KNOWN_FACES_DIR):
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        
        # Step 1: Validate filename format
        valid, reason = validate_filename(filename)
        if not valid:
            skipped_files.append(f"{filename}: {reason}")
            continue
            
        # Step 2: Process image
        try:
            image = face_recognition.load_image_file(filepath)
            # Detect face locations with CNN model (more accurate)
            face_locations = face_recognition.face_locations(image, model="hog")
            if not face_locations:
                skipped_files.append(f"{filename}: No faces detected")
                continue
                
            if len(face_locations) > 1:
                skipped_files.append(f"{filename}: Multiple faces detected")
                continue
                
            # Generate encodings using detected face locations
            encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
            
            # Step 3: Store data
            name, student_id = os.path.splitext(filename)[0].split('_')
            known_encodings.append(encodings[0])
            known_metadata.append({"name": name, "id": student_id})
            print(f"✅ Success: {name} (ID: {student_id})")
            
        except Exception as e:
            skipped_files.append(f"{filename}: {str(e)}")
            continue

    # Save trained data
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "metadata": known_metadata}, f)
        
    # Print training report
    print("\nTraining Report:")
    print(f"Successfully trained: {len(known_metadata)} students")
    if skipped_files:
        print("\nSkipped files:")
        for msg in skipped_files:
            print(f"⚠️ {msg}")

if __name__ == "__main__":
    # Check/retrain model
    if not os.path.exists(ENCODINGS_FILE):
        print("No trained model found. Starting training...")
        train_model()
    else:
        retrain = input("Retrain model? (y/n): ").lower()
        if retrain == 'y':
            train_model()
    
    # Main menu loop
    while True:
        image_path = get_image_path()
        if not image_path:  # User chose to exit
            break
            
        mark_attendance(image_path)
        
        # Ask if user wants to continue
        choice = input("\nProcess another image? (y/n): ").strip().lower()
        if choice != 'y':
            break

    print("\nAttendance marking completed.")