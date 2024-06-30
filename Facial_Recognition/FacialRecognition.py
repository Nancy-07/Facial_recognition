import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from FaceDetection import FACE_DETECTION

# Load the face detection model
face_detection_model = load_model("FaceDetectionModel.keras")

# Function to detect and extract the face from an image
def extract_face(image):
    if len(image.shape) != 3:
        raise ValueError("Image does not have 3 dimensions (height, width, channels)")
    face_detection = FACE_DETECTION(image, face_detection_model)
    coordinates = face_detection.get_coords()
    if coordinates and len(coordinates) == 4:
        x1, y1, x2, y2 = coordinates
        face = image[y1:y2, x1:x2]
        return face
    return None

# Function to apply gamma correction to an image
def apply_gamma_correction(image, gamma=1.5):
    image_normalized = image / 255.0
    image_corrected = np.power(image_normalized, gamma)
    return (255 * image_corrected).astype(np.uint8)

# Function to enhance the face by converting to grayscale and equalizing the histogram
def enhance_face(image):
    face = extract_face(image)
    if face is None or face.size == 0:
        return None
    if len(face.shape) != 2:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    equalized_face = cv2.equalizeHist(face)
    enhanced_face = apply_gamma_correction(equalized_face)
    return enhanced_face

# Function to compute Euclidean distance between two vectors
def compute_euclidean_distance(a, b):
    max_length = max(len(a), len(b))
    a = np.pad(a, (0, max_length - len(a)))
    b = np.pad(b, (0, max_length - len(b)))
    return np.linalg.norm(a - b)

# Function to compute SIFT descriptors for two images and find matches
def compute_sift_descriptors(image1, image2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf_matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    descriptors1_pairs = [descriptors1[match.queryIdx] for match in matches]
    descriptors2_pairs = [descriptors2[match.trainIdx] for match in matches]
    return np.array(descriptors1_pairs), np.array(descriptors2_pairs)

# Function to compute the number of close matches between two sets of descriptors
def count_close_matches(descriptors1, descriptors2, threshold=400):
    distances = [compute_euclidean_distance(descriptors1[i], descriptors2[i]) for i in range(len(descriptors1))]
    return sum(1 for dist in distances if dist < threshold)

# Function to compute SIFT descriptors for a single image
def compute_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# Function to compute Bayesian probabilities for similarity scores
def compute_probabilities(similarity_scores, keypoints_list, descriptors_list):
    total_samples = len(keypoints_list)
    prior_probability = 1 / total_samples
    probabilities = []

    for i in range(total_samples):
        if len(descriptors_list[i]) == 0:
            continue  # Skip if there are no descriptors to avoid division by zero
        conditional_probability = similarity_scores[i] / len(descriptors_list[i])
        numerator = prior_probability * conditional_probability
        denominator = sum((prior_probability * similarity_scores[j] / len(descriptors_list[j])) for j in range(total_samples) if len(descriptors_list[j]) != 0)
        if denominator == 0:
            posterior_probability = 0
        else:
            posterior_probability = numerator / denominator
        probabilities.append(posterior_probability)

    return probabilities

# Function to resize a list of images to a target size
def resize_images(images, target_size):
    resized_images = []
    for image in images:
        if len(image.shape) == 2:  
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resized_images.append(cv2.resize(image, target_size))
    return resized_images

# Function to compute the average image from a list of images
def compute_average_image(images):
    image_arrays = [np.array(img, dtype=np.float32) for img in images]
    shape = image_arrays[0].shape
    if not all(array.shape == shape for array in image_arrays):
        raise ValueError("All images must have the same dimensions")

    sum_array = np.zeros(shape, dtype=np.float32)
    for array in image_arrays:
        sum_array += array

    average_array = sum_array / len(image_arrays)
    return np.array(np.round(average_array), dtype=np.uint8)

# Function to recognize a face from a single image using a database of images
def recognize_face(single_image, database_folder, keypoints_list, descriptors_list):
    if len(single_image.shape) == 2:
        single_image = cv2.cvtColor(single_image, cv2.COLOR_GRAY2BGR)
    enhanced_face = enhance_face(single_image)
    if enhanced_face is None:
        return -1
    similarity_scores = []

    for filename in os.listdir(database_folder):
        file_path = os.path.join(database_folder, filename)
        if os.path.isfile(file_path):
            db_image = cv2.imread(file_path)
            if db_image is not None:
                enhanced_db_image = enhance_face(db_image)
                if enhanced_db_image is not None:
                    des1, des2 = compute_sift_descriptors(enhanced_face, enhanced_db_image)
                    similarity_scores.append(count_close_matches(des1, des2))

    probabilities = compute_probabilities(similarity_scores, keypoints_list, descriptors_list)
    return probabilities.index(max(probabilities))

# Initialize video capture from the camera
cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

# Initialize variables
frame_count = 0
captured_frames = []
recognized_index = 0
keypoints_list = []
descriptors_list = []
names = []

# Load the database images and compute SIFT descriptors
database_folder = 'Fotos'
for filename in sorted(os.listdir(database_folder)):  # Ensure consistent order
    file_path = os.path.join(database_folder, filename)
    if os.path.isfile(file_path):
        db_image = cv2.imread(file_path)
        if db_image is not None:
            enhanced_db_image = enhance_face(db_image)
            if enhanced_db_image is not None:
                keypoints_list.append(compute_descriptors(enhanced_db_image))
                descriptors_list.append(compute_descriptors(enhanced_db_image))
                # Extract name from filename without extension
                name, _ = os.path.splitext(filename)
                names.append(name)

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Cannot receive frames. Exiting...")
        break

    # Save frame every 10 frames
    if frame_count % 10 == 0 and frame_count != 0:
        enhanced_faces = [enhance_face(f) for f in captured_frames]
        enhanced_faces = [f for f in enhanced_faces if f is not None]
        if enhanced_faces:
            resized_faces = resize_images(enhanced_faces, (180, 180))
            average_face = compute_average_image(resized_faces)
            recognized_index = recognize_face(average_face, database_folder, keypoints_list, descriptors_list)
        captured_frames = []

    captured_frames.append(frame)
    frame_count += 1

    # Draw rectangle around detected faces and annotate with the recognized name
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if recognized_index >= 0 and recognized_index < len(names):
            cv2.putText(frame, names[recognized_index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (204, 200, 215), 2)
    
    if frame_count % 20 == 0 and frame_count != 0:
        cv2.imwrite(f"Results/Frame_{frame_count}.png", frame)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
