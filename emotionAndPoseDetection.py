import cv2
from deepface import DeepFace
import os
from tqdm import tqdm
import mediapipe as mp

def detect_emotions_and_actions(video_path, output_path):
    # Initialize MediaPipe for pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Action detection counters
    action_counts = {
        "waving": 0,
        "writing": 0,
        "using_cellphone": 0,
        "dancing": 0,
        "grimace": 0,
        "walking": 0,
        "raising_arm": 0,
        "greeting": 0,
        "lying_down": 0
    }

    # State tracking to avoid redundant consecutive detections
    previous_actions = {key: False for key in action_counts}

    # Capture video from the specified file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened correctly
    if not cap.isOpened():
        print("Error opening the video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initial values for previous wrist positions
    previous_right_wrist_x = 0.0
    previous_right_wrist_y = 0.0

    # Process each frame in the video
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Emotion detection
        face_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        for face in face_result:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            dominant_emotion = face['dominant_emotion']
            if dominant_emotion:
                if dominant_emotion == "grimace" and not previous_actions["grimace"]:
                    action_counts["grimace"] += 1
                    previous_actions["grimace"] = True
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                previous_actions["grimace"] = False

        # Pose detection for action recognition
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            # Check for waving gesture (movement of wrist above head level)
            is_waving = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y) or \
                        (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
            if is_waving and not previous_actions["waving"]:
                action_counts["waving"] += 1
                previous_actions["waving"] = True
            else:
                previous_actions["waving"] = False

            # Check for raised arm gesture
            is_raising_arm = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y) or \
                             (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
            if is_raising_arm and not previous_actions["raising_arm"]:
                action_counts["raising_arm"] += 1
                previous_actions["raising_arm"] = True
            else:
                previous_actions["raising_arm"] = False

            # Check for walking (movement in leg landmarks over frames)
            left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
            right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
            if abs(left_foot - right_foot) > 0.05 and not previous_actions["walking"]:
                action_counts["walking"] += 1
                previous_actions["walking"] = True
            else:
                previous_actions["walking"] = False

            # Check for waving (upward and downward movement of one hand)
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y

            # Waving if the wrist is significantly above and below the shoulder
            if left_wrist < left_shoulder - 0.1 and abs(left_elbow - left_wrist) > 0.05 and not previous_actions["waving"]:
                action_counts["waving"] += 1
                previous_actions["waving"] = True
            else:
                previous_actions["waving"] = False

            # Check for writing (small repetitive movements of the wrist)
            right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
            right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y

            # Writing detected if there is a subtle repetitive movement
            if abs(right_wrist_x - previous_right_wrist_x) < 0.05 and abs(right_wrist_y - previous_right_wrist_y) < 0.05:
                action_counts["writing"] += 1
                previous_actions["writing"] = True
            else:
                previous_actions["writing"] = False

            # Store current wrist position for comparison in the next frame
            previous_right_wrist_x = right_wrist_x
            previous_right_wrist_y = right_wrist_y
            
            # Check for using a cellphone (hand near the head)
            right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

            if abs(right_hand.x - right_ear.x) < 0.1 and abs(right_hand.y - right_ear.y) < 0.1 and not previous_actions["using_cellphone"]:
                action_counts["using_cellphone"] += 1
                previous_actions["using_cellphone"] = True
            else:
                previous_actions["using_cellphone"] = False

            # Check for dancing (detect broad movement in arms and legs)
            left_arm_movement = abs(left_wrist - left_shoulder)
            right_leg_movement = abs(right_foot - landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y)

            if left_arm_movement > 0.1 and right_leg_movement > 0.1 and not previous_actions["dancing"]:
                action_counts["dancing"] += 1
                previous_actions["dancing"] = True
            else:
                previous_actions["dancing"] = False
                
            # Check for greeting (short upward movement of the hand)
            if left_wrist < left_shoulder - 0.1 and not previous_actions["greeting"]:
                action_counts["greeting"] += 1
                previous_actions["greeting"] = True
            else:
                previous_actions["greeting"] = False

            # Check for lying down (torso and legs at similar y-values)
            shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y

            if abs(shoulder_y - hip_y) < 0.05 and abs(hip_y - knee_y) < 0.05 and not previous_actions["lying_down"]:
                action_counts["lying_down"] += 1
                previous_actions["lying_down"] = True
            else:
                previous_actions["lying_down"] = False
    
        # Write the processed frame to the output video
        out.write(frame)

    # Release video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print action report
    print("Actions detected in video:")
    for action, count in action_counts.items():
        print(f"{action}: {count} times")
        
    # Create and save a text report of the actions detected
    report_path = os.path.join(os.path.dirname(output_path), 'action_report.txt')
    with open(report_path, 'w') as report_file:
        report_file.write("Actions detected in video:\n")
        for action, count in action_counts.items():
            report_file.write(f"{action}: {count} times\n")

    print(f"Action report saved to: {report_path}")

# Set video paths and call the function
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video.mp4')
output_video_path = os.path.join(script_dir, 'output_video.mp4')
detect_emotions_and_actions(input_video_path, output_video_path)
