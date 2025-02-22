import json
import boto3
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import argparse
from botocore.exceptions import NoCredentialsError
from urllib.parse import unquote_plus

s3 = boto3.client('s3')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def analyze_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        return [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
    return None

def calculate_angle(A, B, C):
    BA = np.array([A[0] - B[0], A[1] - B[1]])
    BC = np.array([C[0] - B[0], C[1] - B[1]])
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    return np.arccos(cosine_angle) * (180 / np.pi)

def check_posture(pose_data):
    shoulder, hip, knee = pose_data[11], pose_data[23], pose_data[25]  # Left side
    hip_angle = calculate_angle(shoulder, hip, knee)
    return "You're leaning too far forward." if hip_angle < 160 else "Good posture!"

def generate_feedback(image):
    pose_data = analyze_pose(image)
    if pose_data:
        return check_posture(pose_data)
    return "No pose detected."

def lambda_handler(event, context):
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = unquote_plus(event['Records'][0]['s3']['object']['key'])
        
        with tempfile.NamedTemporaryFile() as temp_file:
            s3.download_file(bucket, key, temp_file.name)
            image = cv2.imread(temp_file.name)
            feedback = generate_feedback(image)
        
        return {
            'statusCode': 200,
            'body': json.dumps({'feedback': feedback})
        }
    except NoCredentialsError:
        return {'statusCode': 500, 'body': json.dumps('AWS Credentials not found.')}
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps(str(e))}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()
    
    image = cv2.imread(args.image)
    feedback = generate_feedback(image)
    print("Feedback:", feedback)
    # Open the video file
    video = cv2.VideoCapture('path/to/video.mp4')

    # Check if the video file was successfully opened
    if not video.isOpened():
        print("Error opening video file")
        exit()

    # Read frames from the video
    while True:
        # Read the next frame
        ret, frame = video.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Process the frame and generate feedback
        feedback = generate_feedback(frame)

        # Display the feedback
        print("Feedback:", feedback)

    # Release the video file
    video.release()