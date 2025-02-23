import os
import json
import tempfile
import sys
import cv2
import mediapipe as mp
import numpy as np
import boto3
import botocore.exceptions
from flask_cors import CORS

from flask import Flask, request, jsonify
from anthropic import Anthropic
from dotenv import load_dotenv

##############################################
# 1) Environment, AWS, and PROBLEM_RESOURCES
##############################################

load_dotenv()  # Load .env if present
claude_api_key = os.environ.get("CLAUDE_API_KEY")

# Use environment variables in production. 
# Hard-coding here just to match your example:
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME = os.environ.get("AWS_REGION_NAME")

app = Flask(__name__)
client = Anthropic(api_key=claude_api_key)
CORS(app)


# ---------------------------
# PROBLEM_RESOURCES dictionary
# ---------------------------
PROBLEM_RESOURCES = {
    # 1) Excess strain on lower back or hips
    "Excess strain on lower back or hips": {
        "articles": [
            {
                "title": "How to Prevent and Treat Lower Back Painfrom Running",
                "url": "https://www.runnersworld.com/health-injuries/a20865586/back-pain-and-running/",
                "description": "Practical tips to reduce lower-back strain through posture and core strengthening.",
                "image": "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1304825852.jpg"
            },
            {
                "title": "Hip Pain from Running: Causes and Exercises",
                "url": "https://www.healthline.com/health/hip-pain-running",
                "description": "Learn the common reasons for hip pain in runners and stretches to alleviate it.",
                "image": "https://post.healthline.com/wp-content/uploads/2020/08/Young_Man_Rubbing_Hip_Pain-732x549-thumbnail.jpg"
            }
        ],
        "exercises": [
            {
                "name": "Planks or side planks to strengthen core and stabilize spine.",
                "youtube_link": "https://www.youtube.com/watch?v=0Rl5ZQwmS-o&ab_channel=ReleasePhysicalTherapyWashingtonDC"
            },
            {
                "name": "Wall sits for posture alignment and hip engagement.",
                "youtube_link": "https://www.youtube.com/watch?v=A0i_X3pj3I4&ab_channel=NielsenFitnessPremiumIn-Home%28andVirtual%29PersonalTraining"
            },
            {
                "name": "Hip flexor stretches (kneeling lunge) to relieve tightness.",
                "youtube_link": "https://www.youtube.com/watch?v=UrHcQJCqo4Q&t=198s&ab_channel=ToneandTighten"
            }
        ]
    },

    # 2) Spine alignment issues
    "Can cause back pain or poor running economy": {
        "articles": [
            {
                "title": "5 Tips to Improve Running Posture",
                "url": "https://www.runnersworld.com/training/a20859973/perfect-running-form/",
                "description": "Guidance on keeping your spine neutral for better efficiency and fewer aches.",
                "image": "https://hips.hearstapps.com/hmg-prod/images/mid-adult-female-runner-in-nature-royalty-free-image-1651765778.jpg"
            }
        ],
        "exercises": [
            {
                "name": "Cat-Camel stretches to mobilize the spine.",
                "youtube_link": "https://www.youtube.com/watch?v=Ddz-U9nF_B8&ab_channel=LuStrength%26Therapy"
            },
            {
                "name": "Bird-Dog to strengthen deep core stabilizers.",
                "youtube_link": "https://www.youtube.com/watch?v=xOmNXw8F694&ab_channel=MovementReborn"
            }
        ]
    },

    # 3) Neck strain or upper back tension
    "Neck strain or upper back tension": {
        "articles": [
            {
                "title": "Relieving Neck Tension for Runners",
                "url": "https://www.healthline.com/health/neck-strain",
                "description": "How to maintain a neutral head position and avoid neck tension.",
                "image": "https://post.healthline.com/wp-content/uploads/2020/08/neckstrain_thumb.jpg"
            }
        ],
        "exercises": [
            {
                "name": "Gentle neck mobility drills (rotations, tilts).",
                "youtube_link": "https://www.youtube.com/watch?v=K4dmZ5_n6uU&ab_channel=MarkWildman"
            },
            {
                "name": "Shoulder shrugs and scapular retractions to release upper back tension.",
                "youtube_link": "https://www.youtube.com/watch?v=M2F-QI6h5Yc&ab_channel=SpineCareDecompressionandChiropracticCenter"
            }
        ]
    },

    # 4) Knee Drive
    "May reduce running efficiency": {
        "articles": [
            {
                "title": "Running Efficiency 101",
                "url": "https://www.runnersworld.com/training/a20787239/running-efficiency-tips/",
                "description": "Advice on improving knee drive and stride mechanics for better performance.",
                "image": "https://hips.hearstapps.com/hmg-prod/images/running-efficiency-101.jpg"
            }
        ],
        "exercises": [
            {
                "name": "High-knee drills to reinforce knee lift.",
                "youtube_link": "https://www.youtube.com/watch?v=G9fwHJog4O0&ab_channel=LIVESTRONG.COM"
            },
            {
                "name": "Form strides focusing on a quick, powerful knee drive.",
                "youtube_link": "https://www.youtube.com/watch?v=BjhUfR5AeH8&ab_channel=JPGloria"
            }
        ]
    },

    # 5) Foot Strike
    "Higher impact on knees and shins": {
        "articles": [
            {
                "title": "How to Avoid Knee and Shin Injuries",
                "url": "https://www.healthline.com/health/running-and-knee-injuries",
                "description": "Tips to reduce heel striking and prevent shin splints or knee pain.",
                "image": "https://post.healthline.com/wp-content/uploads/2020/08/638668003-732x549-thumbnail.jpg"
            }
        ],
        "exercises": [
            {
                "name": "Short stride drills to lessen braking forces.",
                "youtube_link": "https://www.youtube.com/watch?v=VjSONvP3kds&ab_channel=RaymerStrength"
            },
            {
                "name": "Toe taps and calf raises for strengthening shins and ankles.",
                "youtube_link": "https://www.youtube.com/watch?v=gRHg6v6-szc&ab_channel=www.sportsinjuryclinic.net"
            }
        ]
    }
}



##############################################
# 2) MediaPipe Setup & Constants
##############################################

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BODY_CONNECTIONS = []
for (start_idx, end_idx) in mp_pose.POSE_CONNECTIONS:
    if start_idx >= 11 and end_idx >= 11:
        BODY_CONNECTIONS.append((start_idx, end_idx))

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

##############################################
# 3) Claude Running Coach Function
##############################################

def call_claude_coach(analysis_categories):
    """
    Sends running form analysis feedback to Claude 3.5 Sonnet 2024-10-22
    and returns personalized running coach advice in JSON.
    """
    if not claude_api_key:
        print("Claude API Key is missing! Set CLAUDE_API_KEY as an environment variable.")
        return {"error": "Claude API Key is missing"}

    # Format categories for the prompt
    formatted_issues = ""
    for category in analysis_categories:
        if category["status"] == "wrong":
            formatted_issues += (
                f"- **{category['title']}**: {category['issue_description']} "
                f"(Potential issues: {category['potential_health_issues']})\n"
            )
        else:
            formatted_issues += f"- **{category['title']}**: No issue detected. Good job!\n"

    # Construct structured prompt
    prompt = f"""
    You are a professional running coach analyzing an athlete’s running form. Act like a tough but supportive coach.
    The following categories were evaluated...

    ### **Running Form Analysis Results**
    {formatted_issues}
    
    Give a paragraph of personalized advice for each category to help the athlete improve their form. Don't include any formatting or bullet points.
    Keep your response to maximum of 75 words.
    """
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        if hasattr(message, "content"):
            try:
                return {"claude_suggestions": message.content[0].text}
            except json.JSONDecodeError:
                return {"error": "Claude did not return valid JSON"}
        else:
            return {"error": "Claude response format issue"}

    except Exception as e:
        print(f"Error calling Claude: {e}")
        return {"error": "Claude API request failed"}

##############################################
# 4) Helper / Utility Functions (Upload, etc.)
##############################################

def upload_to_s3(local_file, bucket, s3_key):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION_NAME
    )
    try: 
        s3.upload_file(local_file, bucket, s3_key)
        print(f"[INFO] Uploaded {local_file} to s3://{bucket}/{s3_key}")
    except botocore.exceptions.ClientError as e:
        error_message = e.response['Error']['Message']
        print(f"Upload failed: {error_message}")
    except FileNotFoundError:
        print(f"[ERROR] The file {local_file} was not found.")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

def convertAVItoMP4(input_avi, output_mp4):
    cap = cv2.VideoCapture(input_avi)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (w,h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Converted {input_avi} -> {output_mp4}")

    if os.path.exists(input_avi):
        os.remove(input_avi)
        print(f"[INFO] Removed {input_avi}")

##############################################
# 5) Skeleton / Heatmap / Overlay Drawing
##############################################

mp_drawing = mp.solutions.drawing_utils

def draw_default_skeleton(landmarks_px, frame, color=(255,255,255), circle_color=(0,255,0), radius=5, thickness=2):
    for (start_idx, end_idx) in BODY_CONNECTIONS:
        pt1 = landmarks_px[start_idx]
        pt2 = landmarks_px[end_idx]
        cv2.line(frame, pt1, pt2, color, thickness)

    for i in range(11,33):
        x, y = landmarks_px[i]
        cv2.circle(frame, (x,y), radius, circle_color, -1)

def blend_color(value, min_val=0.0, max_val=1.0):
    ratio = np.clip((value - min_val)/(max_val - min_val + 1e-9), 0, 1)
    r = int(255 * ratio)
    g = int(255 * (1-ratio))
    b = 0
    return (b,g,r)

def angle_3pt(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = np.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = np.sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba < 1e-9 or mag_bc < 1e-9:
        return 0.0
    cos_angle = dot/(mag_ba*mag_bc)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.degrees(np.arccos(cos_angle))

def angle_to_vertical(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    angle_x = np.degrees(np.arctan2(dy, dx))
    return abs(90 - angle_x)

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_heatmap_stress(landmarks_px):
    lm = {i: (landmarks_px[i][0], landmarks_px[i][1]) for i in range(len(landmarks_px))}
    stress = {}
    for i in range(11,33):
        stress[i] = 0.0

    # Knees
    if (LEFT_HIP in lm) and (LEFT_KNEE in lm) and (LEFT_ANKLE in lm):
        left_knee_angle = angle_3pt(lm[LEFT_HIP], lm[LEFT_KNEE], lm[LEFT_ANKLE])
        if left_knee_angle > 120:
            stress[LEFT_KNEE] = np.clip((left_knee_angle - 120)/60, 0, 1)

    if (RIGHT_HIP in lm) and (RIGHT_KNEE in lm) and (RIGHT_ANKLE in lm):
        right_knee_angle = angle_3pt(lm[RIGHT_HIP], lm[RIGHT_KNEE], lm[RIGHT_ANKLE])
        if right_knee_angle > 120:
            stress[RIGHT_KNEE] = np.clip((right_knee_angle - 120)/60, 0, 1)

    # Hips center
    if (LEFT_HIP in lm) and (RIGHT_HIP in lm):
        hips_center = (
            (lm[LEFT_HIP][0] + lm[RIGHT_HIP][0]) / 2,
            (lm[LEFT_HIP][1] + lm[RIGHT_HIP][1]) / 2
        )
    else:
        hips_center = (0,0)

    # Body height
    body_height = 100
    if (LEFT_SHOULDER in lm) and (RIGHT_SHOULDER in lm):
        shoulders_center = (
            (lm[LEFT_SHOULDER][0] + lm[RIGHT_SHOULDER][0]) / 2,
            (lm[LEFT_SHOULDER][1] + lm[RIGHT_SHOULDER][1]) / 2
        )
        body_height = euclidean_distance(hips_center, shoulders_center)

    # Heel striking
    if LEFT_FOOT_INDEX in lm:
        foot_offset_L = lm[LEFT_FOOT_INDEX][0] - hips_center[0]
        if foot_offset_L > 0.3 * body_height:
            stress[LEFT_ANKLE] = max(stress[LEFT_ANKLE], 1.0)
            stress[LEFT_HEEL]  = max(stress[LEFT_HEEL], 1.0)
            stress[LEFT_FOOT_INDEX] = max(stress[LEFT_FOOT_INDEX], 1.0)

    if RIGHT_FOOT_INDEX in lm:
        foot_offset_R = lm[RIGHT_FOOT_INDEX][0] - hips_center[0]
        if foot_offset_R > 0.3 * body_height:
            stress[RIGHT_ANKLE] = max(stress[RIGHT_ANKLE], 1.0)
            stress[RIGHT_HEEL]  = max(stress[RIGHT_HEEL], 1.0)
            stress[RIGHT_FOOT_INDEX] = max(stress[RIGHT_FOOT_INDEX], 1.0)

    # Trunk lean
    trunk_stress = 0.0
    if (LEFT_SHOULDER in lm) and (RIGHT_SHOULDER in lm):
        angle_vert = angle_to_vertical(shoulders_center, hips_center)
        if angle_vert > 15:
            trunk_stress = np.clip((angle_vert - 15)/30, 0, 1)

    for j in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]:
        if j in stress:
            stress[j] = max(stress[j], trunk_stress)

    return stress

def create_bad_runner_skeleton(video_path, output_avi, white_bg=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_avi, fourcc, fps, (w,h))

    pose = mp_pose.Pose(
        model_complexity=1,
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if white_bg:
            draw_frame = np.full((h, w, 3), 255, dtype=np.uint8)
        else:
            draw_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks_px = []
            for lm in results.pose_landmarks.landmark:
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                landmarks_px.append((x_px, y_px))

            draw_default_skeleton(landmarks_px, draw_frame, (255,255,255), (0,255,0))

        out.write(draw_frame)

    cap.release()
    out.release()

def extract_all_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    all_lms = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            coords = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
        else:
            coords = [(0.0, 0.0)] * 33
        all_lms.append(coords)
    cap.release()
    return all_lms

def center_of_gravity(points_px):
    """
    Compute average (x,y) among a list of points in pixel coords.
    points_px: [(x1,y1), (x2,y2), ...]
    Returns (mean_x, mean_y)
    """
    x_vals = [p[0] for p in points_px]
    y_vals = [p[1] for p in points_px]
    return (np.mean(x_vals), np.mean(y_vals))

def create_bad_runner_with_good_overlay(bad_runner_path, good_runner_path, output_avi):
    good_landmarks = extract_all_landmarks(good_runner_path)
    bad_landmarks  = extract_all_landmarks(bad_runner_path)
    num_good = len(good_landmarks)
    num_bad  = len(bad_landmarks)

    dist_bad_sum = 0.0
    dist_good_sum = 0.0
    valid_count = 0

    max_frames = min(num_good, num_bad)

    for i in range(max_frames):
        bad_norm = bad_landmarks[i]
        good_norm = good_landmarks[i]

        dist_bad_hips  = euclidean_distance(bad_norm[LEFT_HIP],  bad_norm[RIGHT_HIP])
        dist_good_hips = euclidean_distance(good_norm[LEFT_HIP], good_norm[RIGHT_HIP])

        # If either distance is near zero, skip
        if dist_good_hips > 1e-6 and dist_bad_hips > 1e-6:
            dist_bad_sum  += dist_bad_hips
            dist_good_sum += dist_good_hips
            valid_count += 1

    if valid_count > 0 and dist_good_sum > 1e-6:
        scale_factor_global = (dist_bad_sum / valid_count) / (dist_good_sum / valid_count)
    else:
        scale_factor_global = 1.0

    cap_bad = cv2.VideoCapture(bad_runner_path)
    num_bad = int(cap_bad.get(cv2.CAP_PROP_FRAME_COUNT))
    fps  = cap_bad.get(cv2.CAP_PROP_FPS)
    w    = int(cap_bad.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap_bad.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_avi, fourcc, fps, (w,h))

    idx = 0
    while True:
        ret, frame_bad = cap_bad.read()
        if not ret:
            break

        good_idx = int((idx / num_bad) * num_good)
        if good_idx >= num_good:
            good_idx = num_good - 1

        good_lms = good_landmarks[good_idx]
        bad_lms = bad_landmarks[idx]

        bad_points_px = [(int(x*w),  int(y*h)) for (x,y) in bad_lms]
        good_points_px= [(int(x*w),  int(y*h)) for (x,y) in good_lms]

        cog_bad  = center_of_gravity(bad_points_px) 
        cog_good = center_of_gravity(good_points_px)

        final_good = []
        for (gx, gy) in good_points_px:
            # Shift the good runner so that (0,0) is the good CoG
            shifted_x = gx - cog_good[0]
            shifted_y = gy - cog_good[1]

            # Scale around that origin
            scaled_x = shifted_x * scale_factor_global
            scaled_y = shifted_y * scale_factor_global

            # Then shift so that final CoG matches the bad CoG
            final_x = scaled_x + cog_bad[0]
            final_y = scaled_y + cog_bad[1]

            final_good.append((int(final_x), int(final_y)))

        for conn in BODY_CONNECTIONS:
            start_idx, end_idx = conn
            pt1 = bad_points_px[start_idx]
            pt2 = bad_points_px[end_idx]
            cv2.line(frame_bad, pt1, pt2, (0, 0, 255), 2)
        for (bx,by) in bad_points_px:
            cv2.circle(frame_bad, (bx,by), 4, (0,0,255), -1)

        # 2) Good runner (transformed) in Green
        for conn in BODY_CONNECTIONS:
            start_idx, end_idx = conn
            pt1 = final_good[start_idx]
            pt2 = final_good[end_idx]
            cv2.line(frame_bad, pt1, pt2, (0,255,0), 2)
        for (gx,gy) in final_good:
            cv2.circle(frame_bad, (gx,gy), 4, (0,255,0), -1)

        out.write(frame_bad)
        idx += 1

    cap_bad.release()
    out.release()


def create_bad_runner_heatmap(video_path, output_avi, white_bg=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_avi, fourcc, fps, (w,h))

    pose = mp_pose.Pose(
        model_complexity=1,
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if white_bg:
            draw_frame = np.full((h, w, 3), 255, dtype=np.uint8)
        else:
            draw_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks_px = []
            for lm in results.pose_landmarks.landmark:
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                landmarks_px.append((x_px, y_px))

            stress_dict = compute_heatmap_stress(landmarks_px)

            # draw lines
            for (start_idx, end_idx) in BODY_CONNECTIONS:
                s_val = stress_dict[start_idx]
                e_val = stress_dict[end_idx]
                mean_val = (s_val + e_val)/2
                line_color = blend_color(mean_val, 0, 1)
                cv2.line(draw_frame, landmarks_px[start_idx], landmarks_px[end_idx], line_color, 2)

            # draw circles
            for i in range(11,33):
                st = stress_dict[i]
                col = blend_color(st, 0,1)
                cv2.circle(draw_frame, landmarks_px[i], 8, col, -1)

        out.write(draw_frame)

    cap.release()
    out.release()

##############################################
# 6) The Full Running Form Analysis
##############################################

def distance_2d(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def calculate_angle_2d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def vector_angle_with_vertical(a, b):
    a, b = np.array(a), np.array(b)
    vec = b - a
    vertical = np.array([0, 1])
    cosine_angle = np.dot(vec, vertical) / (np.linalg.norm(vec)*np.linalg.norm(vertical))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosine_angle))
    return angle_deg

def moving_average(array, window_size=5):
    if len(array) < window_size:
        return array
    cumsum = np.cumsum(np.insert(array, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

def detect_strikes_using_velocity(y_positions, vel_threshold=0.0005):
    strikes = []
    velocities = np.diff(y_positions)
    for i in range(1, len(velocities)):
        if velocities[i-1] < -vel_threshold and velocities[i] > vel_threshold:
            strikes.append(i)
    return strikes

def detect_local_minima(data, delta=0.005):
    minima_indices = []
    for i in range(1, len(data) - 1):
        if data[i] < data[i - 1] and data[i] < data[i + 1]:
            if abs(data[i] - data[i - 1]) > delta and abs(data[i] - data[i + 1]) > delta:
                minima_indices.append(i)
    return minima_indices

def analyze_running_form(video_path, output_path="output_traced.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    left_side_angles = []
    right_side_angles = []
    left_foot_index_y = []
    left_heel_y = []
    right_foot_index_y = []
    right_heel_y = []

    spine_angles = []
    head_angles = []
    left_knee_drive = []
    right_knee_drive = []
    stride_distances = []
    left_arm_distances = []
    right_arm_distances = []

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            left_shoulder = (lm[11].x, lm[11].y)
            right_shoulder = (lm[12].x, lm[12].y)
            left_hip = (lm[23].x, lm[23].y)
            right_hip = (lm[24].x, lm[24].y)
            left_knee_pt = (lm[25].x, lm[25].y)
            right_knee_pt = (lm[26].x, lm[26].y)
            left_heel_pt = (lm[29].x, lm[29].y)
            right_heel_pt = (lm[30].x, lm[30].y)
            left_foot_idx = (lm[31].x, lm[31].y)
            right_foot_idx = (lm[32].x, lm[32].y)
            nose_pt = (lm[0].x, lm[0].y)
            left_wrist = (lm[15].x, lm[15].y)
            right_wrist = (lm[16].x, lm[16].y)

            mid_shoulder = (
                (left_shoulder[0] + right_shoulder[0]) / 2.0,
                (left_shoulder[1] + right_shoulder[1]) / 2.0
            )
            mid_hip = (
                (left_hip[0] + right_hip[0]) / 2.0,
                (left_hip[1] + right_hip[1]) / 2.0
            )

            # Angles
            angle_left_side  = calculate_angle_2d(left_shoulder, left_hip, left_knee_pt)
            angle_right_side = calculate_angle_2d(right_shoulder, right_hip, right_knee_pt)
            left_side_angles.append(angle_left_side)
            right_side_angles.append(angle_right_side)

            # Feet / Heels
            left_foot_index_y.append(left_foot_idx[1])
            left_heel_y.append(left_heel_pt[1])
            right_foot_index_y.append(right_foot_idx[1])
            right_heel_y.append(right_heel_pt[1])

            # Spine & head
            spine_angle = vector_angle_with_vertical(mid_shoulder, mid_hip)
            spine_angles.append(spine_angle)
            head_angle  = vector_angle_with_vertical(mid_shoulder, nose_pt)
            head_angles.append(head_angle)

            # Knees
            raw_left_knee = left_hip[1] - left_knee_pt[1]
            raw_right_knee = right_hip[1] - right_knee_pt[1]
            left_knee_drive.append(abs(raw_left_knee))
            right_knee_drive.append(abs(raw_right_knee))

            # Stride
            dist_feet = distance_2d(left_foot_idx, right_foot_idx)
            stride_distances.append(dist_feet)

            # Arms
            left_arm_distances.append(abs(left_wrist[0] - left_shoulder[0]))
            right_arm_distances.append(abs(right_wrist[0] - right_shoulder[0]))

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        out_writer.write(frame)

    cap.release()
    out_writer.release()
    pose_estimator.close()

    if duration_sec <= 0:
        duration_sec = 1.0

    # Summaries
    avg_left_angle = np.mean(left_side_angles) if left_side_angles else 0
    avg_right_angle = np.mean(right_side_angles) if right_side_angles else 0

    LEAN_FORWARD_THRESHOLD = 150
    LEAN_BACK_THRESHOLD = 185

    def posture_feedback(angle):
        if angle < LEAN_FORWARD_THRESHOLD:
            return "Leaning forward"
        elif angle > LEAN_BACK_THRESHOLD:
            return "Leaning backward"
        else:
            return "Good posture"

    posture_left = posture_feedback(avg_left_angle)
    posture_right = posture_feedback(avg_right_angle)

    # Foot strike detection
    smoothed_left_idx_y = moving_average(left_foot_index_y, window_size=5)
    smoothed_right_idx_y = moving_average(right_foot_index_y, window_size=5)
    left_vel_strikes = detect_strikes_using_velocity(smoothed_left_idx_y)
    right_vel_strikes = detect_strikes_using_velocity(smoothed_right_idx_y)
    total_foot_strikes = len(left_vel_strikes) + len(right_vel_strikes)

    left_minima = detect_local_minima(smoothed_left_idx_y, delta=0.003)
    right_minima = detect_local_minima(smoothed_right_idx_y, delta=0.003)

    left_heel_strikes = 0
    right_heel_strikes = 0
    for i in left_minima:
        if i < len(left_heel_y):
            if left_heel_y[i] >= left_foot_index_y[i]:
                left_heel_strikes += 1
    for i in right_minima:
        if i < len(right_heel_y):
            if right_heel_y[i] >= right_foot_index_y[i]:
                right_heel_strikes += 1

    total_heel_strikes = left_heel_strikes + right_heel_strikes
    if total_foot_strikes > 0:
        heel_strike_ratio = total_heel_strikes / float(len(left_minima) + len(right_minima))
    else:
        heel_strike_ratio = 0.0

    spm = total_foot_strikes / (duration_sec / 60.0) if total_foot_strikes > 0 else 0.0

    # Spine
    avg_spine_angle = np.mean(spine_angles) if spine_angles else 0
    if avg_spine_angle < 10:
        spine_feedback = "Spine looks upright"
    else:
        spine_feedback = f"Spine angled ~{int(avg_spine_angle)}° from vertical"

    # Head
    avg_head_angle = np.mean(head_angles) if head_angles else 0
    if avg_head_angle < 135:
        head_feedback = "Head is tilted/forward"
    else:
        head_feedback = "Head position looks good"

    # Knee drive
    def knee_drive_feedback(val):
        if val > 0.30:
            return "Very high knee lift"
        elif val > 0.15:
            return "Good knee height"
        else:
            return "Low knee lift"

    avg_left_knee_drive = np.mean(left_knee_drive) if left_knee_drive else 0
    avg_right_knee_drive = np.mean(right_knee_drive) if right_knee_drive else 0
    left_knee_feedback = knee_drive_feedback(avg_left_knee_drive)
    right_knee_feedback = knee_drive_feedback(avg_right_knee_drive)

    # Stride length
    avg_stride_length = np.mean(stride_distances) if stride_distances else 0
    if avg_stride_length < 0.1:
        stride_feedback = f"Short stride (avg ~{avg_stride_length:.2f})"
    elif avg_stride_length > 0.3:
        stride_feedback = f"Long stride (avg ~{avg_stride_length:.2f})"
    else:
        stride_feedback = f"Good stride length (avg ~{avg_stride_length:.2f})"

    # Arm swing
    max_left_arm_amp = np.max(left_arm_distances) if left_arm_distances else 0
    max_right_arm_amp = np.max(right_arm_distances) if right_arm_distances else 0
    excessive_arm_swing = False
    not_enough_arm_swing = False
    arm_swing_comment = ""

    if max_left_arm_amp > 0.1 or max_right_arm_amp > 0.1:
        excessive_arm_swing = True
        arm_swing_comment = "Arms swing amplitude is too great."
    elif max_left_arm_amp < 0.045 and max_right_arm_amp < 0.045:
        not_enough_arm_swing = True
        arm_swing_comment = "Arm swing amplitude is not enough."
    else:
        arm_swing_comment = "Arm swing amplitude looks good."

    # Summarize textual feedback
    feedback_messages = []
    if "forward" in posture_left.lower() or "forward" in posture_right.lower():
        feedback_messages.append("Try to maintain a more upright torso to reduce forward lean.")
    if "backward" in posture_left.lower() or "backward" in posture_right.lower():
        feedback_messages.append("You may be leaning backward. Keep shoulders over hips for efficiency.")
    if "angled" in spine_feedback.lower():
        feedback_messages.append(spine_feedback)
    if "tilted" in head_feedback.lower():
        feedback_messages.append(head_feedback)
    if ("low" in left_knee_feedback.lower() or "very" in left_knee_feedback.lower() or
        "low" in right_knee_feedback.lower() or "very" in right_knee_feedback.lower()):
        feedback_messages.append(f"Left knee: {left_knee_feedback}, Right knee: {right_knee_feedback}")
    if excessive_arm_swing:
        feedback_messages.append("You may be swinging your arms too far on more compact arm drive.")
    if not_enough_arm_swing:
        feedback_messages.append("Increase arm swing amplitude slightly for more power and balance.")

    percent_heel = round(heel_strike_ratio * 100, 1)
    if total_foot_strikes > 0:
        if percent_heel > 70:
            feedback_messages.append(
                f"Predominantly heel-striking (~{percent_heel}%). Consider more midfoot to reduce impact."
            )
        elif percent_heel > 30:
            feedback_messages.append(
                f"Mixed foot strike (~{percent_heel}% heel). Focus on comfort & moderate foot strike."
            )
        else:
            feedback_messages.append(
                f"Mostly forefoot/midfoot (~{percent_heel}% heel)."
            )
    else:
        feedback_messages.append("No foot strikes detected—video too short or detection issue?")

    feedback_messages.append(stride_feedback)
    if not any(msg for msg in feedback_messages if msg != stride_feedback):
        feedback_messages.append("Great form overall! Keep up the good work.")

    # Recommended drills
    recommended_exercises = []
    if any("forward lean" in msg.lower() for msg in feedback_messages):
        recommended_exercises.append("Core exercises (planks) to support upright posture")
        recommended_exercises.append("Wall stands or leaning drills to maintain neutral spine")
    if "spine angled" in " ".join(feedback_messages).lower():
        recommended_exercises.append("Spine mobility & stability exercises (cat-camel, bird-dog, etc.)")
    if "tilted" in " ".join(feedback_messages).lower():
        recommended_exercises.append("Neck stretches and posture awareness drills. Keep eyes forward.")
    if "low knee lift" in " ".join(feedback_messages).lower():
        recommended_exercises.append("Knee drive drills (high-knee marching/running)")
    if any("heel-striking" in msg.lower() for msg in feedback_messages):
        recommended_exercises.append("Short stride drills: land midfoot under center of mass")
        recommended_exercises.append("Light barefoot strides (on a safe surface) to encourage softer foot strike")
    if "short stride" in stride_feedback.lower():
        recommended_exercises.append("Drills focusing on slightly longer push-off and extension")
    elif "long stride" in stride_feedback.lower():
        recommended_exercises.append("Overstriding fix: emphasize landing with foot under hips")
    if not recommended_exercises:
        recommended_exercises = ["No major issues detected. Keep training consistently!"]

    # Build analysis_categories
    analysis_categories = []

    # 1) POSTURE
    posture_item = {
        "title": "Posture",
        "status": "right",
        "issue_description": "No issue",
        "potential_health_issues": "None"
    }
    if ("forward" in posture_left.lower() or "forward" in posture_right.lower() or
        "backward" in posture_left.lower() or "backward" in posture_right.lower()):
        posture_item["status"] = "wrong"
        posture_item["issue_description"] = f"Runner is {posture_left} on left, {posture_right} on right"
        posture_item["potential_health_issues"] = "Excess strain on lower back or hips"

    # If 'wrong', attach resources
    if posture_item["status"] == "wrong":
        issue_key = posture_item["potential_health_issues"]
        if issue_key in PROBLEM_RESOURCES:
            posture_item["articles"] = PROBLEM_RESOURCES[issue_key]["articles"]
            posture_item["exercises"] = PROBLEM_RESOURCES[issue_key]["exercises"]
        else:
            posture_item["articles"] = []
            posture_item["exercises"] = []
    else:
        posture_item["articles"] = []
        posture_item["exercises"] = []
    analysis_categories.append(posture_item)

    # 2) SPINE
    spine_item = {
        "title": "Spine Alignment",
        "status": "right",
        "issue_description": "No issue",
        "potential_health_issues": "None"
    }
    if "angled" in spine_feedback.lower():
        spine_item["status"] = "wrong"
        spine_item["issue_description"] = spine_feedback
        spine_item["potential_health_issues"] = "Can cause back pain or poor running economy"

    if spine_item["status"] == "wrong":
        key = spine_item["potential_health_issues"]
        if key in PROBLEM_RESOURCES:
            spine_item["articles"] = PROBLEM_RESOURCES[key]["articles"]
            spine_item["exercises"] = PROBLEM_RESOURCES[key]["exercises"]
        else:
            spine_item["articles"] = []
            spine_item["exercises"] = []
    else:
        spine_item["articles"] = []
        spine_item["exercises"] = []
    analysis_categories.append(spine_item)

    # 3) HEAD
    head_item = {
        "title": "Head Position",
        "status": "right",
        "issue_description": "No issue",
        "potential_health_issues": "None"
    }
    if "tilted" in head_feedback.lower():
        head_item["status"] = "wrong"
        head_item["issue_description"] = head_feedback
        head_item["potential_health_issues"] = "Neck strain or upper back tension"

    if head_item["status"] == "wrong":
        key = head_item["potential_health_issues"]
        if key in PROBLEM_RESOURCES:
            head_item["articles"] = PROBLEM_RESOURCES[key]["articles"]
            head_item["exercises"] = PROBLEM_RESOURCES[key]["exercises"]
        else:
            head_item["articles"] = []
            head_item["exercises"] = []
    else:
        head_item["articles"] = []
        head_item["exercises"] = []
    analysis_categories.append(head_item)

    # 4) KNEE DRIVE
    knee_item = {
        "title": "Knee Drive",
        "status": "right",
        "issue_description": "No issue",
        "potential_health_issues": "None"
    }

    # For 'low knee lift' in both legs, or 'very high knee lift':
    if ("low knee lift" in left_knee_feedback.lower() and
        "low knee lift" in right_knee_feedback.lower()):
        knee_item["status"] = "wrong"
        knee_item["issue_description"] = f"Left knee: {left_knee_feedback}, Right knee: {right_knee_feedback}"
        knee_item["potential_health_issues"] = "May reduce running efficiency"
    elif ("very high knee lift" in left_knee_feedback.lower() or
        "very high knee lift" in right_knee_feedback.lower()):
        knee_item["status"] = "wrong"
        knee_item["issue_description"] = f"Left knee: {left_knee_feedback}, Right knee: {right_knee_feedback}"
        knee_item["potential_health_issues"] = "Could cause extra energy expenditure"

    if knee_item["status"] == "wrong":
        key = knee_item["potential_health_issues"]
        if key in PROBLEM_RESOURCES:
            knee_item["articles"] = PROBLEM_RESOURCES[key]["articles"]
            knee_item["exercises"] = PROBLEM_RESOURCES[key]["exercises"]
        else:
            knee_item["articles"] = []
            knee_item["exercises"] = []
    else:
        knee_item["articles"] = []
        knee_item["exercises"] = []
    analysis_categories.append(knee_item)

    # 5) FOOT STRIKE
    foot_item = {
        "title": "Foot Strike",
        "status": "right",
        "issue_description": "No issue",
        "potential_health_issues": "None"
    }
    percent_heel = round(heel_strike_ratio * 100, 1)
    if percent_heel > 70:
        foot_item["status"] = "wrong"
        foot_item["issue_description"] = f"Predominantly heel-striking (~{percent_heel}% heel)"
        foot_item["potential_health_issues"] = "Higher impact on knees and shins"
    elif percent_heel > 30:
        foot_item["issue_description"] = f"Mixed foot strike (~{percent_heel}% heel)"
        foot_item["potential_health_issues"] = "Generally okay, watch for any imbalance"
    else:
        foot_item["issue_description"] = f"Mostly forefoot/midfoot (~{percent_heel}% heel)"
        foot_item["potential_health_issues"] = "None"

    if foot_item["status"] == "wrong":
        key = foot_item["potential_health_issues"]
        if key in PROBLEM_RESOURCES:
            foot_item["articles"] = PROBLEM_RESOURCES[key]["articles"]
            foot_item["exercises"] = PROBLEM_RESOURCES[key]["exercises"]
        else:
            foot_item["articles"] = []
            foot_item["exercises"] = []
    else:
        foot_item["articles"] = []
        foot_item["exercises"] = []
    analysis_categories.append(foot_item)

    # 6) STRIDE LENGTH
    stride_item = {
        "title": "Stride Length",
        "status": "right",
        "issue_description": "No issue",
        "potential_health_issues": "None"
    }

    if "short stride" in stride_feedback.lower():
        stride_item["status"] = "wrong"
        stride_item["issue_description"] = stride_feedback
        stride_item["potential_health_issues"] = "Reduced efficiency, possible overuse of calves"
    elif "long stride" in stride_feedback.lower():
        stride_item["status"] = "wrong"
        stride_item["issue_description"] = stride_feedback
        stride_item["potential_health_issues"] = "Overstriding can stress knees & hamstrings"

    if stride_item["status"] == "wrong":
        key = stride_item["potential_health_issues"]
        if key in PROBLEM_RESOURCES:
            stride_item["articles"] = PROBLEM_RESOURCES[key]["articles"]
            stride_item["exercises"] = PROBLEM_RESOURCES[key]["exercises"]
        else:
            stride_item["articles"] = []
            stride_item["exercises"] = []
    else:
        stride_item["articles"] = []
        stride_item["exercises"] = []
    analysis_categories.append(stride_item)

    # 7) ARM SWING
    arm_amp_item = {
        "title": "Arm Swing (Amplitude)",
        "status": "right",
        "issue_description": "No issue",
        "potential_health_issues": "None"
    }

    # If amplitude is too great, or not enough:
    if excessive_arm_swing:
        arm_amp_item["status"] = "wrong"
        arm_amp_item["issue_description"] = "Excessive horizontal arm swing"
        arm_amp_item["potential_health_issues"] = "Shoulder/neck fatigue, wasted energy"
    if not_enough_arm_swing:
        arm_amp_item["status"] = "wrong"
        if arm_amp_item["issue_description"] == "No issue":
            arm_amp_item["issue_description"] = "Insufficient arm swing causing reduced balance, less power in stride"
            arm_amp_item["potential_health_issues"] = "Less balance, less power"
        else:
            arm_amp_item["issue_description"] += "; also insufficient swing"
            if arm_amp_item["potential_health_issues"] == "None":
                arm_amp_item["potential_health_issues"] = "Less balance, less power"
            else:
                arm_amp_item["potential_health_issues"] += ", plus reduced balance/power"

    if arm_amp_item["status"] == "wrong":
        key = arm_amp_item["potential_health_issues"]
        if key in PROBLEM_RESOURCES:
            arm_amp_item["articles"] = PROBLEM_RESOURCES[key]["articles"]
            arm_amp_item["exercises"] = PROBLEM_RESOURCES[key]["exercises"]
        else:
            arm_amp_item["articles"] = []
            arm_amp_item["exercises"] = []
    else:
        arm_amp_item["articles"] = []
        arm_amp_item["exercises"] = []
    analysis_categories.append(arm_amp_item)

    # Final form score
    total_cats = len(analysis_categories)
    wrong_count = sum(1 for cat in analysis_categories if cat["status"] == "wrong")
    raw_score = 100 * (1 - wrong_count / total_cats)
    form_score = max(0, min(100, round(raw_score)))

    results = {
        "duration_sec": round(duration_sec, 2),
        "average_left_side_angle": round(avg_left_angle, 2),
        "average_right_side_angle": round(avg_right_angle, 2),
        "posture_left": posture_left,
        "posture_right": posture_right,
        "foot_strikes_detected": total_foot_strikes,
        "heel_strikes_detected": total_heel_strikes,
        "heel_strike_ratio_percent": percent_heel,
        "estimated_cadence_spm": round(spm, 2),
        "spine_alignment_degs": round(avg_spine_angle, 1),
        "head_position_degs": round(avg_head_angle, 1),
        "spine_feedback": spine_feedback,
        "head_feedback": head_feedback,
        "left_knee_height": round(avg_left_knee_drive, 2),
        "right_knee_height": round(avg_right_knee_drive, 2),
        "left_knee_feedback": left_knee_feedback,
        "right_knee_feedback": right_knee_feedback,
        "stride_feedback": stride_feedback,
        "avg_stride_length": round(avg_stride_length, 3),
        "max_left_arm_amplitude": round(max_left_arm_amp, 3),
        "max_right_arm_amplitude": round(max_right_arm_amp, 3),
        "arm_swing_amplitude_comment": arm_swing_comment,
        "recommended_exercises": recommended_exercises,
        "analysis_categories": analysis_categories,
        "form_score": form_score,
        "traced_video_path": output_path
    }
    return results

##############################################
# 7) Flask Endpoint: /analyze
##############################################

@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    """
    1) Save user-uploaded "bad runner" video
    2) Hard-code "goodForm.mp4" for overlay
    3) Generate 6 .avi videos (skeleton+overlay+heatmap) with original + white bg
    4) Convert .avi -> .mp4
    5) Upload all .mp4 to S3
    6) analyze_running_form, call Claude
    7) Return JSON with everything
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video"]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
        bad_runner_path = temp.name
        video_file.save(bad_runner_path)

    good_runner_path = "goodForm.mp4"  # local file for overlay
    traced_output_path = bad_runner_path.replace(".mp4", "_traced.mp4")

    # 1) Create skeleton, overlay, heatmap (orig background)
    skeleton_avi = bad_runner_path + "_skeleton.avi"
    create_bad_runner_skeleton(bad_runner_path, skeleton_avi, white_bg=False)

    overlay_avi = bad_runner_path + "_overlay.avi"
    create_bad_runner_with_good_overlay(bad_runner_path, good_runner_path, overlay_avi)

    heatmap_avi  = bad_runner_path + "_heatmap.avi"
    create_bad_runner_heatmap(bad_runner_path, heatmap_avi, white_bg=False)

    # 3) Convert each .avi -> .mp4
    skeleton_mp4       = skeleton_avi.replace(".avi", ".mp4")
    overlay_mp4        = overlay_avi.replace(".avi", ".mp4")
    heatmap_mp4        = heatmap_avi.replace(".avi", ".mp4")

    all_avi_mp4_pairs = [
        (skeleton_avi,       skeleton_mp4),
        (overlay_avi,        overlay_mp4),
        (heatmap_avi,        heatmap_mp4)
    ]

    for (avi_file, mp4_file) in all_avi_mp4_pairs:
        convertAVItoMP4(avi_file, mp4_file)

    # 4) Upload all .mp4 to S3
    bucket_name = "formflow-videos"
    s3_links = {}
    final_mp4_list = [
        skeleton_mp4,
        overlay_mp4,
        heatmap_mp4
    ]

    for vid_path in final_mp4_list:
        s3_key = f"initalvids/{os.path.basename(vid_path)}"
        upload_to_s3(vid_path, bucket_name, s3_key)
        s3_links[os.path.basename(vid_path)] = f"s3://{bucket_name}/{s3_key}"

    # 5) Actually analyze running form -> traced video
    results = analyze_running_form(bad_runner_path, traced_output_path)
    if "error" in results:
        return jsonify(results), 400

    # 6) Call Claude
    claude_suggestions = call_claude_coach(results["analysis_categories"])
    results.update(claude_suggestions)

    # Optionally upload the traced video to S3, if it exists
    if os.path.exists(traced_output_path):
        traced_key = f"initalvids/{os.path.basename(traced_output_path)}"
        upload_to_s3(traced_output_path, bucket_name, traced_key)
        s3_links[os.path.basename(traced_output_path)] = f"s3://{bucket_name}/{traced_key}"

    results["generated_videos"] = s3_links

    # Convert list-based categories to dict for final JSON structure
    analysis_categories_dict = {
        cat["title"]: {
            "status": cat["status"],
            "issue_description": cat["issue_description"],
            "potential_health_issues": cat["potential_health_issues"],
            "articles": cat.get("articles", []),
            "exercises": cat.get("exercises", [])
        }
        for cat in results["analysis_categories"]
    }

    # 7) Structure and return final JSON
    response = {
        "statusCode": 200,
        "body": {
            "form_score": results["form_score"],
            "duration": {
                "seconds": results["duration_sec"]
            },
            "angles": {
                "average_left_side": results["average_left_side_angle"],
                "average_right_side": results["average_right_side_angle"]
            },
            "posture": {
                "left": results["posture_left"],
                "right": results["posture_right"]
            },
            "foot_strike": {
                "total_strikes": results["foot_strikes_detected"],
                "heel_strikes": results["heel_strikes_detected"],
                "heel_strike_ratio": results["heel_strike_ratio_percent"]
            },
            "cadence": {
                "steps_per_minute": results["estimated_cadence_spm"]
            },
            "spine_alignment": {
                "degrees": results["spine_alignment_degs"]
            },
            "head_position": {
                "degrees": results["head_position_degs"]
            },
            "knee_drive": {
                "left": {
                    "height": results["left_knee_height"],
                    "feedback": results["left_knee_feedback"]
                },
                "right": {
                    "height": results["right_knee_height"],
                    "feedback": results["right_knee_feedback"]
                }
            },
            "stride": {
                "feedback": results["stride_feedback"],
                "average_length": results["avg_stride_length"]
            },
            "arm_swing": {
                "left_amplitude": results["max_left_arm_amplitude"],
                "right_amplitude": results["max_right_arm_amplitude"],
                "comment": results["arm_swing_amplitude_comment"]
            },
            "recommended_exercises": results["recommended_exercises"],
            "analysis_categories": analysis_categories_dict,
            "traced_video_path": results["traced_video_path"],
            "claude_suggestions": results.get("claude_suggestions"),
            "generated_videos": results["generated_videos"]
        }
    }
    return jsonify(response)


@app.route("/")
def health_check():
    return "Running form analysis + skeleton/heatmap + overlay (6 videos) is up and running!", 200

##############################################
# 8) Main Launch
##############################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
