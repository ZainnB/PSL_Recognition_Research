import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# CONFIG
# ==========================================
DATA_ROOT = r"C:\Users\Dell\Documents\FYP_PSL\data\virkha+zain+ark"
OUTPUT_DIR = "processed_psl_research_zain"
MODEL_PATH = "pose_models/hand_landmarker.task"

MIN_FRAMES = 120
MAX_FRAMES = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# MEDIAPIPE SETUP
# ==========================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

# ==========================================
# VIDEO READER
# ==========================================
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    total = len(frames)

    if total > MAX_FRAMES:
        idx = np.linspace(0, total - 1, MAX_FRAMES).astype(int)
        frames = [frames[i] for i in idx]

    if total < MIN_FRAMES:
        pad_count = MIN_FRAMES - total
        for _ in range(pad_count):
            frames.append(frames[-1])

    return frames

# ==========================================
# PALM SIZE NORMALIZATION
# ==========================================
def palm_normalize(hand_points):
    """
    Normalize by palm size using wrist (0) and middle MCP (9)
    """
    wrist = hand_points[0]
    middle_mcp = hand_points[9]

    palm_size = np.linalg.norm(wrist[:3] - middle_mcp[:3]) + 1e-6
    hand_points[:, :3] /= palm_size

    return hand_points

# ==========================================
# POSE EXTRACTION USING WORLD LANDMARKS
# ==========================================
def extract_pose(frames):
    all_keypoints = []

    for frame in frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image
        )

        result = detector.detect(mp_image)

        left_hand = np.zeros((21, 4))
        right_hand = np.zeros((21, 4))

        if result.hand_world_landmarks:
            for i, hand_landmarks in enumerate(result.hand_world_landmarks):

                label = result.handedness[i][0].category_name
                conf = result.handedness[i][0].score

                points = []
                for lm in hand_landmarks:
                    points.append([lm.x, lm.y, lm.z, conf])

                points = np.array(points)

                # Palm scale normalization
                points = palm_normalize(points)

                if label == "Left":
                    left_hand = points
                else:
                    right_hand = points

        frame_keypoints = np.vstack((left_hand, right_hand))
        all_keypoints.append(frame_keypoints)

    pose = np.array(all_keypoints, dtype=np.float32)  # (T, 42, 4)

    return pose

# ==========================================
# WRIST-CENTER NORMALIZATION
# ==========================================
def wrist_center_normalize(pose):
    # subtract left wrist position
    left_wrist = pose[:, 0:1, :3]
    pose[:, :, :3] -= left_wrist
    return pose

# ==========================================
# VELOCITY COMPUTATION
# ==========================================
def add_velocity(pose):
    velocity = np.zeros_like(pose[:, :, :3])
    velocity[1:] = pose[1:, :, :3] - pose[:-1, :, :3]

    # concatenate: (x,y,z, vx,vy,vz, conf)
    pose_with_vel = np.concatenate(
        [pose[:, :, :3], velocity, pose[:, :, 3:4]],
        axis=2
    )

    return pose_with_vel  # (T, 42, 7)

# ==========================================
# DATASET PROCESSOR
# ==========================================
def process_dataset():
    for label in os.listdir(DATA_ROOT):

        label_path = os.path.join(DATA_ROOT, label)
        if not os.path.isdir(label_path):
            continue

        for video_name in os.listdir(label_path):

            if not video_name.endswith((".mp4")):
                continue

            video_path = os.path.join(label_path, video_name)
            print(f"Processing: {video_path}")

            frames = read_video(video_path)

            pose = extract_pose(frames)
            pose = wrist_center_normalize(pose)
            pose = add_velocity(pose)

            save_name = f"{video_name.split('.')[0]}.npy"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            np.save(save_path, pose)

    print("Research-grade dataset ready.")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    process_dataset()