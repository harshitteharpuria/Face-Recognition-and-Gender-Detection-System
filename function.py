import cv2
import tempfile
import os
import numpy as np
import torch
from PIL import Image
from numpy.linalg import norm
from facenet_pytorch import InceptionResnetV1, MTCNN
from deepface import DeepFace
from torchvision import transforms
from io import BytesIO
 
# Load the FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize MTCNN detector
detector = MTCNN(keep_all=True)

# Define transformations for face images
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Resize to match model input size
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

def preprocess_image(image_data):
    try:
        image = Image.open(image_data).convert('RGB')
        return image
    except Exception as e:
        print(f"Error opening image file: {e}")
        return None

def extract_frames(video_stream, num_frames=5):
    try:
        # Write the video stream to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(video_stream.read())
        temp_file.close()

        # Open the video stream using cv2.VideoCapture with the temporary file
        cap = cv2.VideoCapture(temp_file.name)

        if not cap.isOpened():
            raise IOError("Error opening video stream")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // num_frames)

        for count in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        # Clean up the temporary file
        os.remove(temp_file.name)

        return frames
    except Exception as e:
        print(f"Error extracting frames from video stream: {e}")
        return []
def process_image_and_get_embedding(image):
    try:
        # Detect faces using MTCNN
        detected_faces = detector.detect(image)
        if detected_faces[0] is None or len(detected_faces[0]) == 0:
            raise ValueError("No faces detected")

        # Process each detected face
        embeddings = []
        for face in detected_faces[0]:
            x1, y1, x2, y2 = map(int, face)
            face_image = image.crop((x1, y1, x2, y2))  # Crop face from image
            face_image = transform(face_image).unsqueeze(0)  # Apply transformations

            # Generate the embedding using FaceNet
            with torch.no_grad():                                  
                embedding = facenet_model(face_image).numpy()
                embeddings.append(embedding)

        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            raise ValueError("No face embeddings generated")

    except Exception as e:
        print(f"Error processing image and getting embedding: {e}")
        return None

def get_gender(image_data):
    try:
        image = Image.open(image_data).convert('RGB')
        # Convert image to numpy array for DeepFace
        image_np = np.array(image)

        # Ensure the image file is an image, not a video
        analysis = DeepFace.analyze(image_np, actions=['gender'], enforce_detection=False)
        # Use enforce_detection=False to avoid the error if a face is not detected
        gender_probabilities = analysis[0]['gender']
        gender = max(gender_probabilities, key=gender_probabilities.get)
        print(f"gender: {gender}")
        return gender
    except Exception as e:
        print(f"Error detecting gender for image: {e}")
        return None

def get_face_embedding(file_data, is_video=False):
    if is_video:
        frames = extract_frames(file_data, num_frames=5)
        if not frames:
            return None
        embeddings = []
        for frame in frames:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            embedding = process_image_and_get_embedding(image)
            if embedding is not None:
                embeddings.append(embedding)
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return None
    else:
        image = preprocess_image(file_data)
        if image is None:
            return None
        return process_image_and_get_embedding(image)

def verify_face(id_card_embedding, video_frame_embedding, threshold=0.8):
    if id_card_embedding is None or video_frame_embedding is None:
        print("Embeddings are not available for comparison")
        return False

    distance = norm(id_card_embedding - video_frame_embedding)
    print(f"Distance: {distance}")
    is_match = distance < threshold
    if is_match:
        print("Match: True")
    else:
        print("Match: False")
    return distance < threshold