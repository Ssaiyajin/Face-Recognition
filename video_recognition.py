import face_recognition
import os
import cv2
import pickle
import time
import argparse
from pathlib import Path


def name_to_color(name: str):
    """Generate a color based on the first three letters of the name."""
    return [(ord(c.lower()) - 97) * 8 for c in name[:3]]


def load_known_faces(directory: Path):
    """Load known faces and names from the given directory."""
    known_faces, known_names = [], []

    if not directory.exists():
        print(f"[INFO] Directory '{directory}' does not exist. Creating it...")
        directory.mkdir(parents=True, exist_ok=True)

    for person_dir in directory.iterdir():
        if person_dir.is_dir():
            for file in person_dir.glob("*.pkl"):
                try:
                    with open(file, "rb") as f:
                        encoding = pickle.load(f)
                    known_faces.append(encoding)
                    known_names.append(int(person_dir.name))
                except Exception as e:
                    print(f"[WARN] Could not load {file}: {e}")

    return known_faces, known_names


def save_face_encoding(directory: Path, person_id: str, encoding, frame):
    """Save a new face encoding + snapshot."""
    person_dir = directory / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    with open(person_dir / f"{person_id}-{timestamp}.pkl", "wb") as f:
        pickle.dump(encoding, f)
    cv2.imwrite(str(person_dir / f"{person_id}-{timestamp}.jpg"), frame)


def main():
    parser = argparse.ArgumentParser(description="Dynamic Face Recognition System")
    parser.add_argument("--known_dir", type=str, default="known_faces", help="Directory for known faces")
    parser.add_argument("--tolerance", type=float, default=0.6, help="Tolerance for face matching")
    parser.add_argument("--model", type=str, default="hog", choices=["hog", "cnn"], help="Face detection model")
    parser.add_argument("--camera", type=str, default="0", help="Camera index or video file path")
    parser.add_argument("--frame_thickness", type=int, default=3, help="Bounding box thickness")
    parser.add_argument("--font_thickness", type=int, default=2, help="Label font thickness")
    parser.add_argument("--reload_interval", type=int, default=30, help="Seconds between auto-reload of known faces")

    args = parser.parse_args()

    known_faces_dir = Path(args.known_dir)
    tolerance = args.tolerance
    model = args.model
    frame_thickness = args.frame_thickness
    font_thickness = args.font_thickness
    reload_interval = args.reload_interval

    source = int(args.camera) if args.camera.isdigit() else args.camera

    print(f"\n[CONFIG] Known Faces Dir: {known_faces_dir}")
    print(f"[CONFIG] Model: {model}, Tolerance: {tolerance}")
    print(f"[CONFIG] Using Source: {source}\n")

    video = cv2.VideoCapture(source)
    if not video.isOpened():
        print("[ERROR] Cannot access camera or video file.")
        return

    print("[INFO] Loading known faces...")
    known_faces, known_names = load_known_faces(known_faces_dir)
    next_id = max(map(int, known_names), default=0) + 1
    last_reload = time.time()

    print("[INFO] Starting video stream... Press 'Q' or 'ESC' to quit, 'C' to switch camera.")
    current_cam = source

    while True:
        ret, frame = video.read()
        if not ret:
            print("[WARN] Failed to read frame. Exiting loop...")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_frame, model=model)
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
            match = None

            if True in results:
                match = str(known_names[results.index(True)])
            else:
                match = str(next_id)
                print(f"[NEW] New face detected → ID {match}")
                next_id += 1
                known_names.append(int(match))
                known_faces.append(face_encoding)
                save_face_encoding(known_faces_dir, match, face_encoding, frame)

            top, right, bottom, left = [v * 2 for v in face_location]
            color = name_to_color(match)
            cv2.rectangle(frame, (left, top), (right, bottom), color, frame_thickness)

            label = f"ID: {match}"
            cv2.rectangle(frame, (left, bottom + 20), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 5, bottom + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), font_thickness)

        fps = video.get(cv2.CAP_PROP_FPS) or 30
        cv2.putText(frame, f"Known: {len(known_faces)} | FPS: {int(fps)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Dynamic Face Recognition", frame)

        # 🔹 Handle keypress events safely
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            print("\n[INFO] Exiting gracefully...")
            break
        elif key == ord('c'):
            print("[INFO] Switching camera...")
            video.release()
            current_cam = 1 - int(current_cam)
            video = cv2.VideoCapture(current_cam)

        # 🔁 Auto reload faces every few seconds
        if time.time() - last_reload > reload_interval:
            print("[INFO] Reloading known faces dynamically...")
            known_faces, known_names = load_known_faces(known_faces_dir)
            last_reload = time.time()

        # 🔸 Quit if user closes window
        if cv2.getWindowProperty("Dynamic Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
            print("\n[INFO] Window closed by user.")
            break

    # 🔻 Cleanup safely
    video.release()
    cv2.destroyAllWindows()
    print("[INFO] Cleanup done. Goodbye!")


if __name__ == "__main__":
    main()
