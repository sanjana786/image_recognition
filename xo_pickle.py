import pickle
import face_recognition
import os
from PIL import Image

# Load known faces and their images from a directory
known_faces = []
known_images = []

known_faces_dir = "image_subset"
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg"):
        face_image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        face_encodings = face_recognition.face_encodings(face_image)
        
        # Check if a face was found in the image
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]  # Assuming one face per image
            known_faces.append(face_encoding)
            known_images.append(Image.open(os.path.join(known_faces_dir, filename)))

# Create a dictionary to store known data
known_data = {
    "known_faces": known_faces,
    "known_images": known_images
}

# Save the known data to a pickle file
with open("known_data.pkl", "wb") as file:
    pickle.dump(known_data, file)
