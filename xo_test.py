import pickle
import face_recognition

# Load the known data from the pickle file
with open("known_data.pkl", "rb") as file:
    known_data = pickle.load(file)

known_faces = known_data["known_faces"]
known_images = known_data["known_images"]

# Load an unknown face for testing (replace "test.jpg" with the path to your unknown image)
unknown_image = face_recognition.load_image_file("test.jpg")
unknown_face_encodings = face_recognition.face_encodings(unknown_image)

# Check if a face was found in the unknown image
if len(unknown_face_encodings) > 0:
    unknown_face_encoding = unknown_face_encodings[0]
    
    # Compare the unknown face to the known faces
    results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

    if True in results:
        # Match found
        match_index = results.index(True)
        matched_image = known_images[match_index]
        matched_image.show()  # Show the matched image
        print("Successful match found")
    else:
        # No match found
        print("No match found")
else:
    # No face found in the unknown image
    print("No face found in the unknown image")
