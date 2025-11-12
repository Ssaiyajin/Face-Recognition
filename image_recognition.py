import face_recognition
import os
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, known_faces_dir='known_faces', unknown_faces_dir='unknown_faces',
                 tolerance=0.6, frame_thickness=3, font_thickness=2, model='hog',
                 display_width=800):
        self.known_faces_dir = known_faces_dir
        self.unknown_faces_dir = unknown_faces_dir
        self.tolerance = tolerance
        self.frame_thickness = frame_thickness
        self.font_thickness = font_thickness
        self.model = model
        self.display_width = display_width
        self.known_faces = []
        self.known_names = []

    @staticmethod
    def name_to_color(name):
        """Generate RGB color from name dynamically."""
        hash_value = sum(ord(c) for c in name)
        return [(hash_value * 3) % 256, (hash_value * 7) % 256, (hash_value * 13) % 256]

    def load_known_faces(self):
        if not os.path.isdir(self.known_faces_dir):
            logger.warning(f"Known faces directory '{self.known_faces_dir}' does not exist.")
            return
        
        logger.info(f"Loading known faces from {self.known_faces_dir}...")
        for root, dirs, files in os.walk(self.known_faces_dir):
            for file in files:
                file_path = os.path.join(root, file)
                name = os.path.basename(root)
                
                image = face_recognition.load_image_file(file_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_faces.append(encodings[0])
                    self.known_names.append(name)
        logger.info(f"Loaded {len(self.known_faces)} known faces.")

    def recognize_faces(self):
        if not os.path.isdir(self.unknown_faces_dir):
            logger.warning(f"Unknown faces directory '{self.unknown_faces_dir}' does not exist.")
            return

        logger.info(f"Processing unknown faces from {self.unknown_faces_dir}...")
        for filename in os.listdir(self.unknown_faces_dir):
            file_path = os.path.join(self.unknown_faces_dir, filename)
            logger.info(f"Processing file: {filename}")

            image = face_recognition.load_image_file(file_path)
            locations = face_recognition.face_locations(image, model=self.model)
            encodings = face_recognition.face_encodings(image, locations)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            logger.info(f"Found {len(encodings)} face(s) in {filename}.")

            for face_encoding, face_location in zip(encodings, locations):
                results = face_recognition.compare_faces(self.known_faces, face_encoding, self.tolerance)
                match = None
                if True in results:
                    match = self.known_names[results.index(True)]
                    logger.info(f" - Match: {match}")

                    color = self.name_to_color(match)

                    # Draw rectangle around face
                    top_left = (face_location[3], face_location[0])
                    bottom_right = (face_location[1], face_location[2])
                    cv2.rectangle(image, top_left, bottom_right, color, self.frame_thickness)

                    # Draw text label
                    text_width, text_height = cv2.getTextSize(match, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.font_thickness)[0]
                    label_top_left = (face_location[3], face_location[2])
                    label_bottom_right = (face_location[3] + text_width + 10, face_location[2] + text_height + 5)
                    cv2.rectangle(image, label_top_left, label_bottom_right, color, cv2.FILLED)
                    cv2.putText(image, match, (face_location[3] + 5, face_location[2] + text_height),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), self.font_thickness)

            # Resize image dynamically
            scale = self.display_width / image.shape[1]
            image_resized = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
            cv2.imshow(filename, image_resized)
            cv2.waitKey(0)
            cv2.destroyWindow(filename)


if __name__ == "__main__":
    recognizer = FaceRecognizer(
    known_faces_dir='known_faces',
    unknown_faces_dir='unknown_faces',
    tolerance=0.6,
    model='hog',
    display_width=800
    )

    recognizer.load_known_faces()
    recognizer.recognize_faces()
