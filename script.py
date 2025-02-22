import argparse
import os
import requests
import cv2
import mediapipe as mp
import numpy as np
import sys
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
############################


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
        return "Sorry, I couldn’t get additional coach advice from Claude."

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
    The following categories were evaluated, and areas needing
improvement are identified.

    ### **Running Form Analysis Results**
    {formatted_issues}

    ### **Coaching Instructions**
    - If a category is marked as **"wrong"**, provide a **detailed
explanation** on how to improve.
    - If a category is marked as **"right"**, provide **positive
reinforcement**.
    - Keep responses **clear, motivational, and practical**.
    - Provide **one structured sentence per category** with concise advice.

    **Example Output Format:**
    - **Head Position**: Keep your gaze forward and relax your neck to
avoid excessive strain.
    - **Knee Drive**: Incorporate high-knee drills to improve power
and efficiency.
    - **Cadence**: Your cadence is slightly high (~201.9 SPM). Ensure
you are not taking excessively short steps and focus on maintaining
rhythm.

    Provide your response now.
    """

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract the content from `message.content`
        return message.content if hasattr(message, "content") else "Claude returned an unexpected response format."
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return "Sorry, I couldn’t get additional coach advice from Claude."


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
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Avoid numerical issues
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

def detect_local_minima(data, delta=0.005):
    """
    Return indices of local minima in 'data'.
    'delta' is a threshold to avoid minor jitters.
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
    """
    1) Reads video with OpenCV
    2) Uses MediaPipe Pose to:
       - Draw pose landmarks (skeleton) on each frame
       - Track posture, cadence, arm swing crossing
       - Detect foot strike type (heel vs forefoot)
       - Check spine alignment, head position, knee height
       - Estimate stride length
       - Estimate arm swing amplitude
    3) Writes traced frames to 'output_path'
    4) Produces summary feedback (WITHOUT feedback_items)
    """

    # Open the video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 1

    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Mediapipe Pose
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

        # Convert BGR -> RGB for MediaPipe
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
    left_minima = detect_local_minima(smoothed_left_idx_y, delta=0.003)

    smoothed_right_idx_y = moving_average(right_foot_index_y, window_size=5)
    right_minima = detect_local_minima(smoothed_right_idx_y, delta=0.003)

    total_foot_strikes = len(left_minima) + len(right_minima)

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
        heel_strike_ratio = total_heel_strikes / float(total_foot_strikes)
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
    arm_swing_comment = ""
    if max_left_arm_amp > 0.25 or max_right_arm_amp > 0.25:
        excessive_arm_swing = True
        arm_swing_comment = "Arms swing quite wide from the shoulders."
    else:
        arm_swing_comment = "Arm swing amplitude looks reasonable."

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

    if spm < 160:
        feedback_messages.append("Consider increasing cadence slightly (~170–180 SPM).")
    elif spm > 200:
        feedback_messages.append("Cadence might be unusually high. Ensure you're not overstraining.")


    if excessive_arm_swing:
        feedback_messages.append("You may be swinging your arms too widely—focus on more compact arm drive.")

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

    if any("increase cadence" in msg.lower() for msg in feedback_messages):
        recommended_exercises.append("Short interval runs focusing on quicker foot turnover")
        recommended_exercises.append("Metronome training at ~180 BPM")

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
    if ("low knee lift" in left_knee_feedback.lower() or
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

    # CADENCE
    cadence_item = {
        "title": "Cadence",
        "status": "right",
        "issue_description": "No issue",
        "potential_health_issues": "None"
    }
    if spm < 160:
        cadence_item["status"] = "wrong"
        cadence_item["issue_description"] = f"Low cadence (~{spm:.1f} SPM)"
        cadence_item["potential_health_issues"] = "Greater impact per stride, risk of injuries"
    elif spm > 200:
        cadence_item["status"] = "wrong"
        cadence_item["issue_description"] = f"High cadence (~{spm:.1f} SPM)"
        cadence_item["potential_health_issues"] = "Possible overexertion or inefficiency"
    analysis_categories.append(cadence_item)

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
    analysis_categories.append(arm_amp_item)

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
    parser.add_argument("--output", type=str,
default="output_traced.mp4", help="Path to the output traced video.")
    args = parser.parse_args()

    # 1) Analyze
    results = analyze_running_form(args.video, args.output)

    # 2) Get Claude Coach Suggestions
    # We call 'call_claude_coach' with the 'analysis_categories'
    claude_suggestions = call_claude_coach(results["analysis_categories"])
    results["claude_suggestions"] = claude_suggestions

    # 3) Print out analysis
    print("\n--- Running Form Analysis ---\n")
    for k, v in results.items():
        if k == "analysis_categories":
            print("analysis_categories:")
            for cat in v:
                print(f"  - {cat}")
        elif k == "claude_suggestions":
            print("\n--- Claude Coach Advice ---\n")
            print(v)
        else:
            print(f"{k}: {v}")

    print("\nAnalysis Complete!\n")

if __name__ == "__main__":
    main()
