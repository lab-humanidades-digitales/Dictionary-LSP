import math
from collections import defaultdict
import numpy as np

def read_pose(value):
    kps = value['people'][0]['pose_keypoints_2d']
    x = kps[0::3]
    y = kps[1::3]
    return np.stack((x, y), axis=1)

def calc_pose_flow(prev, next):
    result = np.zeros_like(prev)
    for kpi in range(prev.shape[0]):
        if np.count_nonzero(prev[kpi]) == 0 or np.count_nonzero(next[kpi]) == 0:
            result[kpi, 0] = 0.0
            result[kpi, 1] = 0.0
            continue

        ang = math.atan2(next[kpi, 1] - prev[kpi, 1], next[kpi, 0] - prev[kpi, 0])
        mag = np.linalg.norm(next[kpi] - prev[kpi])

        result[kpi, 0] = ang
        result[kpi, 1] = mag

    return result

def normalize(poses):
    """Normalize each pose in the array to account for camera position. We normalize
    by dividing keypoints by a factor such that the length of the neck becomes 1."""
    for i in range(poses.shape[0]):
        upper_neck = poses[i,11]
        head_top = poses[i, 12]
        neck_length = np.linalg.norm(upper_neck - head_top)
        if math.isclose(np.linalg.norm(upper_neck - head_top), 0):
            print("because np.linalg.norm(upper_neck - head_top) is zero, it will be normalize divided by 220")
            neck_length = 220.0
        poses[i] /= neck_length

        #assert math.isclose(np.linalg.norm(upper_neck - head_top), 1)

    return poses

def impute_missing_keypoints(poses):
    """Replace missing keypoints (on the origin) by values from neighbouring frames."""
    # 1. Collect missing keypoints
    missing_keypoints = defaultdict(list)  # frame index -> keypoint indices that are missing
    for i in range(poses.shape[0]):
        for kpi in range(poses.shape[1]):
            if np.count_nonzero(poses[i, kpi]) == 0:  # Missing keypoint at (0, 0)
                missing_keypoints[i].append(kpi)
    # 2. Impute them
    for i in missing_keypoints.keys():
        missing = missing_keypoints[i]
        for kpi in missing:
            # Possible replacements
            candidates = poses[:, kpi]
            min_dist = np.inf
            replacement = -1
            for f in range(candidates.shape[0]):
                if f != i and np.count_nonzero(candidates[f]) > 0:
                    distance = abs(f - i)
                    if distance < min_dist:
                        min_dist = distance
                        replacement = f
            # Replace
            if replacement > -1:
                poses[i, kpi] = poses[replacement, kpi]
    # 3. We have imputed as many keypoints as possible with the closest non-missing temporal neighbours
    return normalize(poses)


