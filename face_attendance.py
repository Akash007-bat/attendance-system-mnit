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

# Create directory if not exists 
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

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

    # Recognition logic
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # ADDED TIMESTAMP
    
    for encoding in input_encodings:
        distances = face_recognition.face_distance(data["encodings"], encoding)
        best_match = distances.argmin()
        
        if distances[best_match] <= TOLERANCE:
            student_id = data["metadata"][best_match]["id"]
            attendance[student_id]["status"] = "Present"

    # Save results to daily file - MODIFIED THIS SECTION
    daily_file = get_daily_filename()
    file_exists = os.path.isfile(daily_file)
    
    with open(daily_file, "a", newline="") as f:  # Changed to append mode
        writer = csv.writer(f)
        
        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(["Name", "ID", "Status", "Timestamp"])
        
        # Write all records with current timestamp
        for student_id, data in attendance.items():
            writer.writerow([
                data["name"],
                student_id,
                data["status"],
                timestamp  # Added timestamp to each record
            ])

    # Display results - keep existing code unchanged
    display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for (top, right, bottom, left) in input_locations:
        cv2.rectangle(display_image, (left, top), (right, bottom), (0, 255, 0), 2)
    display_width = 800  # Adjust as needed
    display_height = 600  # Adjust as needed

    # Resize image
    resized_image = cv2.resize(display_image, (display_width, display_height))

    # Show resized image
    cv2.imshow("Results", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # ... (keep existing main code unchanged) ...
     # Force retrain if no model exists
    if not os.path.exists(ENCODINGS_FILE):
        train_model()
    else:
        retrain = input("Retrain model? (y/n): ").lower()
        if retrain == 'y':
            train_model()
            
    # Process attendance
    image_path = input("Enter classroom image path: ").strip()
    if os.path.exists(image_path):
        mark_attendance(image_path)
    else:
        print("❌ Error: Image not found!")