#!/usr/bin/env python3

import argparse
import cv2
import mediapipe as mp
import numpy as np
import sys

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

############################
# Utility / Helper Functions
############################

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
    # A vertical vector might be (0, 1). We'll compare:
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
       - Track posture, cadence, arm swing
       - Detect foot strike type (heel vs forefoot)
       - Checks spine alignment, head position, knee height
    3) Writes traced frames to 'output_path'
    4) Produces summary feedback
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

    # 1) For posture angles
    left_side_angles = []
    right_side_angles = []

    # For foot strike detection
    left_foot_index_y = []
    left_heel_y = []
    right_foot_index_y = []
    right_heel_y = []

    # Arm swing
    arm_cross_count_left = 0
    arm_cross_count_right = 0
    frame_count = 0

    # Additional checks
    # We'll track mid-shoulder and mid-hip for spine alignment
    spine_angles = []
    head_angles = []  # Angle from mid-shoulder to nose vs. vertical
    left_knee_drive = []  # measure knee (y) relative to hip (y)
    right_knee_drive = []

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

            # Landmark indices (Mediapipe Pose):
            # Left side: 11 shoulder, 23 hip, 25 knee, 29 heel, 31 foot_index
            # Right side:12 shoulder,24 hip,26 knee, 30 heel, 32 foot_index
            # Wrists: 15 (L),16 (R)
            # Nose: 0
            # Mid-shoulder approx: average of (11,12)
            # Mid-hip approx: average of (23,24)

            # Basic 2D coords
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

            # Midpoints
            mid_shoulder = (
                (left_shoulder[0] + right_shoulder[0]) / 2.0,
                (left_shoulder[1] + right_shoulder[1]) / 2.0
            )
            mid_hip = (
                (left_hip[0] + right_hip[0]) / 2.0,
                (left_hip[1] + right_hip[1]) / 2.0
            )

            # A) Posture angles (shoulder-hip-knee)
            angle_left_side  = calculate_angle_2d(left_shoulder, left_hip, left_knee_pt)
            angle_right_side = calculate_angle_2d(right_shoulder, right_hip, right_knee_pt)
            left_side_angles.append(angle_left_side)
            right_side_angles.append(angle_right_side)

            # B) Foot Y data for strike detection
            left_foot_index_y.append(left_foot_idx[1])
            left_heel_y.append(left_heel_pt[1])
            right_foot_index_y.append(right_foot_idx[1])
            right_heel_y.append(right_heel_pt[1])

            # C) Arm crossing check
            mid_shoulder_x = mid_shoulder[0]
            if left_wrist[0] > mid_shoulder_x:
                arm_cross_count_left += 1
            if right_wrist[0] < mid_shoulder_x:
                arm_cross_count_right += 1

            # D) Spine alignment
            # Angle between mid_shoulder->mid_hip and vertical
            spine_angle = vector_angle_with_vertical(mid_shoulder, mid_hip)
            spine_angles.append(spine_angle)

            # E) Head position check
            # Angle between mid_shoulder->nose and vertical
            head_angle = vector_angle_with_vertical(mid_shoulder, nose_pt)
            head_angles.append(head_angle)

            # F) Knee height
            # We'll measure difference in Y: hip.y - knee.y (normalized coords)
            # Larger difference => knee is lower, smaller => knee is higher
            # We'll store negative for "knee above hip" which is typically unusual in normal gait
            left_knee_height = (left_hip[1] - left_knee_pt[1])
            right_knee_height = (right_hip[1] - right_knee_pt[1])
            left_knee_drive.append(left_knee_height)
            right_knee_drive.append(right_knee_height)

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Write the traced frame to output
        out_writer.write(frame)

    # Release resources
    cap.release()
    out_writer.release()
    pose_estimator.close()

    ##################################################
    # Post-processing: posture, cadence, arm swing, etc
    ##################################################

    # -- Posture
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

    # -- Cadence (foot strikes)
    smoothed_left_idx_y = moving_average(left_foot_index_y, window_size=5)
    left_minima = detect_local_minima(smoothed_left_idx_y, delta=0.003)

    smoothed_right_idx_y = moving_average(right_foot_index_y, window_size=5)
    right_minima = detect_local_minima(smoothed_right_idx_y, delta=0.003)

    total_foot_strikes = len(left_minima) + len(right_minima)

    # For heel-strike detection
    left_heel_strikes = 0
    right_heel_strikes = 0

    # We'll skip perfect alignment of smoothing offset for brevity
    for i in left_minima:
        if i < len(left_heel_y):
            if left_heel_y[i] >= left_foot_index_y[i]:
                left_heel_strikes += 1
    for i in right_minima:
        if i < len(right_heel_y):
            if right_heel_y[i] >= right_foot_index_y[i]:
                right_heel_strikes += 1

    total_heel_strikes = left_heel_strikes + right_heel_strikes

    if duration_sec > 0:
        spm = total_foot_strikes / (duration_sec / 60.0)
    else:
        spm = 0.0

    # -- Arm swing
    CROSS_THRESHOLD_RATIO = 0.20
    crossing_ratio_left = arm_cross_count_left / frame_count if frame_count > 0 else 0
    crossing_ratio_right = arm_cross_count_right / frame_count if frame_count > 0 else 0

    arm_swing_feedback = []
    if crossing_ratio_left > CROSS_THRESHOLD_RATIO:
        arm_swing_feedback.append("Left arm crosses midline too often")
    if crossing_ratio_right > CROSS_THRESHOLD_RATIO:
        arm_swing_feedback.append("Right arm crosses midline too often")
    if not arm_swing_feedback:
        arm_swing_feedback.append("Arm swing looks okay")

    ###########################
    # Spine, Head, Knee checks
    ###########################

    # Spine: We average the spine angle
    # 0 deg => perfectly vertical, typical range for upright posture maybe < 10 deg
    avg_spine_angle = np.mean(spine_angles) if spine_angles else 0
    if avg_spine_angle < 10:
        spine_feedback = "Spine looks upright"
    else:
        spine_feedback = f"Spine angled ~{int(avg_spine_angle)}° from vertical"

    # Head: We average the head angle
    # If too high or too low, we consider that "off" (this is approximate)
    avg_head_angle = np.mean(head_angles) if head_angles else 0
    # Let's define if angle > 25 => possibly head is forward/down
    # The sign isn't captured in an absolute angle, so we assume any large angle => tilt
    if avg_head_angle > 25:
        head_feedback = "Head is tilted/forward"
    else:
        head_feedback = "Head position looks good"

    # Knee height:
    # We'll measure the average difference hip.y - knee.y
    # If the difference is small => knee is high.
    # Typical difference might be ~0.1 - 0.3, depending on camera angle.
    avg_left_knee_drive = np.mean(left_knee_drive) if left_knee_drive else 0
    avg_right_knee_drive = np.mean(right_knee_drive) if right_knee_drive else 0

    def knee_drive_feedback(val):
        # The smaller the difference, the higher the knee (in normalized coords).
        # We'll define thresholds:
        # val < 0.05 => "very high knee"
        # 0.05 <= val <= 0.15 => "good knee height"
        # >0.15 => "low knee lift"
        if val < 0.05:
            return "Very high knee lift"
        elif val <= 0.15:
            return "Good knee height"
        else:
            return "Low knee lift"

    left_knee_feedback = knee_drive_feedback(avg_left_knee_drive)
    right_knee_feedback = knee_drive_feedback(avg_right_knee_drive)

    ################################
    # Compile overall feedback
    ################################

    feedback_messages = []

    # Basic posture
    if "forward" in posture_left or "forward" in posture_right:
        feedback_messages.append("Try to maintain a more upright torso to reduce forward lean.")
    if "backward" in posture_left or "backward" in posture_right:
        feedback_messages.append("You may be leaning backward. Keep shoulders over hips for efficiency.")

    # Spine
    if "angled" in spine_feedback:
        feedback_messages.append(spine_feedback)

    # Head
    if "tilted" in head_feedback:
        feedback_messages.append(head_feedback)

    # Knee
    # We'll combine left & right to see if there's a consistent issue
    if any(keyword in left_knee_feedback for keyword in ["Low", "Very"]) \
       or any(keyword in right_knee_feedback for keyword in ["Low", "Very"]):
        feedback_messages.append(f"Left knee: {left_knee_feedback}, Right knee: {right_knee_feedback}")

    # Cadence
    if spm < 160:
        feedback_messages.append("Consider increasing cadence slightly (~170–180 SPM).")
    elif spm > 200:
        feedback_messages.append("Cadence might be unusually high. Ensure you're not overstraining.")

    # Arm swing
    if "midline too often" in " ".join(arm_swing_feedback):
        feedback_messages.append("Try to keep arm swing forward-and-back, not crossing your chest.")

    # Heel strike
    heel_strike_ratio = 0.0
    if total_foot_strikes > 0:
        heel_strike_ratio = total_heel_strikes / float(total_foot_strikes)
        percent_heel = round(heel_strike_ratio * 100, 1)

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

    # If no specific issues
    if not feedback_messages:
        feedback_messages.append("Great form! Keep up the good work.")

    ##############################
    # Recommended Exercises
    ##############################
    recommended_exercises = []

    if any("forward lean" in msg for msg in feedback_messages):
        recommended_exercises.append("Core exercises (planks) to support upright posture")
        recommended_exercises.append("Wall stands or leaning drills to maintain neutral spine")

    if "spine angled" in " ".join(feedback_messages):
        recommended_exercises.append("Spine mobility and stability exercises (e.g., cat-camel, bird-dog)")

    if "tilted" in " ".join(feedback_messages):
        recommended_exercises.append("Neck stretches and posture awareness drills. Keep eyes forward.")

    if "Low knee lift" in " ".join(feedback_messages):
        recommended_exercises.append("Knee drive drills (e.g., high-knee marching/running)")

    if any("increase cadence" in msg for msg in feedback_messages):
        recommended_exercises.append("Short interval runs focusing on quicker foot turnover")
        recommended_exercises.append("Metronome training at ~180 BPM")

    if any("heel-striking" in msg for msg in feedback_messages):
        recommended_exercises.append("Short stride drills: land midfoot under center of mass")
        recommended_exercises.append("Light barefoot strides (on a safe surface) to encourage softer foot strike")

    # If no recommended exercises, just say keep training
    if not recommended_exercises:
        recommended_exercises = ["No major issues detected. Keep training consistently!"]

    # Final results
    results = {
        "duration_sec": round(duration_sec, 2),
        "average_left_side_angle": round(avg_left_angle, 2),
        "average_right_side_angle": round(avg_right_angle, 2),
        "posture_left": posture_left,
        "posture_right": posture_right,
        "foot_strikes_detected": total_foot_strikes,
        "heel_strikes_detected": total_heel_strikes,
        "heel_strike_ratio_percent": round(heel_strike_ratio * 100, 2),
        "estimated_cadence_spm": round(spm, 2),
        "arm_swing_feedback": arm_swing_feedback,
        "spine_alignment_degs": round(avg_spine_angle, 1),
        "head_position_degs": round(avg_head_angle, 1),
        "spine_feedback": spine_feedback,
        "head_feedback": head_feedback,
        "left_knee_height": round(avg_left_knee_drive, 2),
        "right_knee_height": round(avg_right_knee_drive, 2),
        "left_knee_feedback": left_knee_feedback,
        "right_knee_feedback": right_knee_feedback,
        "feedback_messages": feedback_messages,
        "recommended_exercises": recommended_exercises,
        "traced_video_path": output_path
    }

    return results

##################
# Main Entry Point
##################

def main():
    parser = argparse.ArgumentParser(description="Analyze running form from a local video, including spine/head/knee checks, and produce a traced output video.")
    parser.add_argument("--video", type=str, required=True, help="Path to the local input video file (e.g. video.mp4).")
    parser.add_argument("--output", type=str, default="output_traced.mp4", help="Path to the output traced video.")
    args = parser.parse_args()

    results = analyze_running_form(args.video, args.output)

    print("\n--- Running Form Analysis ---\n")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("\nAnalysis Complete!\n")

if __name__ == "__main__":
    main()
