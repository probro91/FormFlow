#!/usr/bin/env python3

import argparse
import os
import requests
import cv2
import mediapipe as mp
import numpy as np
import sys
import json
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from a .env file (if present)
load_dotenv()

# Retrieve API Key securely from environment variables
claude_api_key = os.environ.get("CLAUDE_API_KEY")
# Replace with your key if testing locally

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Anthropic Client
client = Anthropic(api_key=claude_api_key)


##########################################
# Claude Running Coach Function
##########################################
def call_claude_coach(analysis_categories):
    """
    Sends running form analysis feedback to Claude 3.5 Sonnet 2024-10-22
    and returns personalized running coach advice.
    """
    if not claude_api_key:
        print("Claude API Key is missing! Set CLAUDE_API_KEY as an environment variable.")
        return {"error": "Claude API Key is missing"}

    # Format categories for the prompt
    formatted_issues = ""
    for category in analysis_categories:
        if category["status"] == "wrong":
            formatted_issues += f"- **{category['title']}**: {category['issue_description']} (Potential issues: {category['potential_health_issues']})\n"
        else:
            formatted_issues += f"- **{category['title']}**: No issue detected. Good job!\n"

    # Construct structured prompt
    prompt = f"""
    You are a professional running coach analyzing an athlete’s running form.
    The following categories were evaluated...

    ### **Running Form Analysis Results**
    {formatted_issues}

    ### **IMPORTANT**:
    1. **Output must be in valid JSON**.

    **Example JSON structure**:
    ```json
    {{
    "Head Position": "Keep your gaze forward and relax your neck to avoid excessive strain.",
    "Knee Drive": "Incorporate high-knee drills to improve power and efficiency.",
    "Cadence": "Your cadence is slightly high (~201.9 SPM). Ensure you are not taking excessively short steps."
    }}
    """
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # Here is where you replace the old return statement
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




############################
# Utility / Helper Functions
############################

def distance_2d(a, b):
    """
    Euclidean distance between two normalized 2D points (x, y).
    """
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)

def calculate_angle_2d(a, b, c):
    """
    Calculate the angle (in degrees) formed by points A-B-C (2D).
    Each point is (x, y) in normalized coordinates [0..1].
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def vector_angle_with_vertical(a, b):
    """
    Returns the angle (in degrees) between the vector from A->B
    and a perfect vertical line (pointing down).
    Each point is (x, y) in normalized coordinates.
    """
    a, b = np.array(a), np.array(b)
    vec = b - a
    vertical = np.array([0, 1])
    cosine_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosine_angle))
    return angle_deg

def moving_average(array, window_size=5):
    """Simple moving average to smooth out noisy foot y-coordinates."""
    if len(array) < window_size:
        return array
    cumsum = np.cumsum(np.insert(array, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

def detect_strikes_using_velocity(y_positions, vel_threshold=0.0005):
    """
    Return indices where velocity crosses from negative to positive.
    y_positions is 1D array of foot Y-coordinates.
    """
    strikes = []
    velocities = np.diff(y_positions)
    for i in range(1, len(velocities)):
        if velocities[i-1] < -vel_threshold and velocities[i] > vel_threshold:
            strikes.append(i)
    return strikes

def detect_local_minima(data, delta=0.005):
    """
    Return indices of local minima with threshold delta.
    """
    minima_indices = []
    for i in range(1, len(data) - 1):
        if data[i] < data[i - 1] and data[i] < data[i + 1]:
            if abs(data[i] - data[i - 1]) > delta and abs(data[i] - data[i + 1]) > delta:
                minima_indices.append(i)
    return minima_indices


#############################
# Main Analysis Function
#############################
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

    arm_cross_count_left = 0
    arm_cross_count_right = 0
    frame_count = 0

    spine_angles = []
    head_angles = []
    left_knee_drive = []
    right_knee_drive = []

    stride_distances = []
    left_arm_distances = []
    right_arm_distances = []

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

            angle_left_side  = calculate_angle_2d(left_shoulder, left_hip, left_knee_pt)
            angle_right_side = calculate_angle_2d(right_shoulder, right_hip, right_knee_pt)
            left_side_angles.append(angle_left_side)
            right_side_angles.append(angle_right_side)

            left_foot_index_y.append(left_foot_idx[1])
            left_heel_y.append(left_heel_pt[1])
            right_foot_index_y.append(right_foot_idx[1])
            right_heel_y.append(right_heel_pt[1])

            mid_shoulder_x = mid_shoulder[0]
            if left_wrist[0] > mid_shoulder_x:
                arm_cross_count_left += 1
            if right_wrist[0] < mid_shoulder_x:
                arm_cross_count_right += 1

            spine_angle = vector_angle_with_vertical(mid_shoulder, mid_hip)
            spine_angles.append(spine_angle)

            head_angle = vector_angle_with_vertical(mid_shoulder, nose_pt)
            head_angles.append(head_angle)

            # Use absolute difference so we never get negatives:
            raw_left_knee = left_hip[1] - left_knee_pt[1]
            raw_right_knee = right_hip[1] - right_knee_pt[1]
            left_knee_drive.append(abs(raw_left_knee))
            right_knee_drive.append(abs(raw_right_knee))

            dist_feet = distance_2d(left_foot_idx, right_foot_idx)
            stride_distances.append(dist_feet)

            left_arm_distances.append(abs(left_wrist[0] - left_shoulder[0]))
            right_arm_distances.append(abs(right_wrist[0] - right_shoulder[0]))

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

    avg_spine_angle = np.mean(spine_angles) if spine_angles else 0
    if avg_spine_angle < 10:
        spine_feedback = "Spine looks upright"
    else:
        spine_feedback = f"Spine angled ~{int(avg_spine_angle)}° from vertical"

    # Lower the forward tilt threshold from 25 to 20
    avg_head_angle = np.mean(head_angles) if head_angles else 0
    if avg_head_angle < 135:
        head_feedback = "Head is tilted/forward"
    else:
        head_feedback = "Head position looks good"

    def knee_drive_feedback(val):
        # If val > 0.30 => 'Very high knee lift'
        # If val > 0.15 => 'Good knee height'
        # else => 'Low knee lift'
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

    avg_stride_length = np.mean(stride_distances) if stride_distances else 0
    if avg_stride_length < 0.1:
        stride_feedback = f"Short stride (avg ~{avg_stride_length:.2f})"
    elif avg_stride_length > 0.3:
        stride_feedback = f"Long stride (avg ~{avg_stride_length:.2f})"
    else:
        stride_feedback = f"Good stride length (avg ~{avg_stride_length:.2f})"

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

    analysis_categories = []

    # POSTURE
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
    analysis_categories.append(posture_item)

    # SPINE
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
    analysis_categories.append(spine_item)

    # HEAD
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
    analysis_categories.append(head_item)

    # KNEE DRIVE
    knee_item = {
        "title": "Knee Drive",
        "status": "right",
        "issue_description": "No issue",
        "potential_health_issues": "None"
    }
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
    analysis_categories.append(knee_item)

    # FOOT STRIKE
    foot_item = {
        "title": "Foot Strike",
        "status": "right",
        "issue_description": "No issue",
        "potential_health_issues": "None"
    }
    if percent_heel > 70:
        foot_item["status"] = "wrong"
        foot_item["issue_description"] = f"Predominantly heel-striking (~{percent_heel}% heel)"
        foot_item["potential_health_issues"] = "Higher impact on knees and shins"
    elif percent_heel > 30:
        foot_item["status"] = "right"
        foot_item["issue_description"] = f"Mixed foot strike (~{percent_heel}% heel)"
        foot_item["potential_health_issues"] = "Generally okay, watch for any imbalance"
    else:
        foot_item["status"] = "right"
        foot_item["issue_description"] = f"Mostly forefoot/midfoot (~{percent_heel}% heel)"
        foot_item["potential_health_issues"] = "None"
    analysis_categories.append(foot_item)

    # STRIDE LENGTH
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
    analysis_categories.append(stride_item)

    # ARM SWING (Amplitude)
    arm_amp_item = {
        "title": "Arm Swing (Amplitude)",
        "status": "right",
        "issue_description": "No issue",
        "potential_health_issues": "None"
    }
    if excessive_arm_swing:
        arm_amp_item["status"] = "wrong"
        arm_amp_item["issue_description"] = "Excessive horizontal arm swing"
        arm_amp_item["potential_health_issues"] = "Shoulder/neck fatigue, wasted energy"

    if not_enough_arm_swing:
        # If both happen, last sets "status" again. If you want them separate, add separate category.
        arm_amp_item["status"] = "wrong"
        arm_amp_item["issue_description"] = "Insufficient arm swing causing reduced balance, less power in stride"
        if arm_amp_item["potential_health_issues"] == "None":
            arm_amp_item["potential_health_issues"] = "Less balance, less power"
        else:
            arm_amp_item["potential_health_issues"] += ", plus reduced balance/power"

    analysis_categories.append(arm_amp_item)

    # Calculate a "score" out of 100 based on how many categories are "wrong"
    total_cats = len(analysis_categories)
    wrong_count = sum(1 for cat in analysis_categories if cat["status"] == "wrong")

    # Simple approach: each wrong category deducts a fraction from 100.
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
        "form_score": form_score,  # Our new form score
        "traced_video_path": output_path
    }

    return results

##################
# Main Entry Point
##################
def main():
    parser = argparse.ArgumentParser(
        description="Analyze running form, then get Claude's coach advice."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (e.g. video.mp4).")
    parser.add_argument("--output", type=str, default="output_traced.mp4", help="Path to the output traced video.")
    args = parser.parse_args()

    # 1) Analyze the running form
    results = analyze_running_form(args.video, args.output)

    # 2) Get Claude Coach Suggestions
    claude_suggestions = call_claude_coach(results["analysis_categories"])
    results.update(claude_suggestions)

    # 3) Convert list-based categories to a dict keyed by each category title
    analysis_categories_dict = {
        cat["title"]: {
            "status": cat["status"],
            "issue_description": cat["issue_description"],
            "potential_health_issues": cat["potential_health_issues"]
        }
        for cat in results["analysis_categories"]
    }

    # 3) Structure JSON response properly
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
            "claude_suggestions": results["claude_suggestions"]
        }
    }
    
    # 4) Print structured JSON output
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()
