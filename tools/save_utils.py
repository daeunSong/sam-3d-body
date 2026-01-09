import numpy as np

# ==========================================
# HELPER FUNCTIONS FOR DATA SAVING
# ==========================================

# Define the specific indices to KEEP (Body + Feet + Wrists + Extra)
# We exclude indices 21-40 (Right Fingers) and 42-61 (Left Fingers)
BODY_ONLY_INDICES = np.array(
    list(range(21)) + [41, 62] + list(range(63, 70)), 
    dtype=np.int32
)

def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def forward_cam_from_rot(rot_matrix: np.ndarray):
    """
    Get a camera-frame forward vector from a rotation matrix.
    Assumes Body Local Forward is +Z [0,0,1].
    """
    R = _to_np(rot_matrix).astype(np.float32)
    local_forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    f = R @ local_forward

    return f

def get_human_state(o):
    """
    Extracts the human's position and facing direction strictly in the Robot Frame.
    Robot Frame Definition: +X Forward, +Y Left, +Z Up.
    
    Args:
        o (dict): The output dictionary for a single person.
        
    Returns:
        tuple: (pos_robot, facing_robot)
            pos_robot (np.ndarray): [x, y] in meters. 
                                    x = Forward, y = Left.
            facing_robot (np.ndarray): [dx, dy] normalized direction vector.
                                       dx = Forward component, dy = Left component.
    """
    # 1. Position Extraction
    # Camera: X=Right, Y=Down, Z=Forward
    # Robot:  X=Forward, Y=Left
    t = np.asarray(o["pred_cam_t"], dtype=np.float32).reshape(-1)
    
    robot_x = float(t[2])       # Forward (Cam Z)
    robot_y = float(-t[0])      # Left (Cam -X)
    pos_robot = np.array([robot_x, robot_y])

    # 2. Rotation Extraction
    rot_matrix = None
    if "pred_global_rots" in o:
        rots = o["pred_global_rots"]
        # Priority: Joint 1 (Pelvis/Root)
        if len(rots.shape) == 4: 
            rot_matrix = rots[0][1] # Batch 0, Joint 1
        elif len(rots.shape) == 3: 
            rot_matrix = rots[1]

    # 3. Facing Direction Extraction
    facing_robot = None
    if rot_matrix is not None:
        f_cam = forward_cam_from_rot(rot_matrix)
        
        # Transform Camera Vector (Right, Down, Forward) to Robot Vector (Forward, Left)
        # Cam X (Right) -> Robot -Y (Left)
        # Cam Z (Forward) -> Robot +X (Forward) -> -X (Something related to SAM3D result)
        
        dir_fwd = -float(f_cam[2])   # Robot X component
        dir_left = -float(f_cam[0]) # Robot Y component
        
        norm = (dir_fwd**2 + dir_left**2)**0.5 + 1e-8
        if norm > 0.01:
             facing_robot = np.array([dir_fwd / norm, dir_left / norm])
             
    return pos_robot, facing_robot

def format_keypoints_for_saving(outputs, frame_name):
    """
    Extracts relevant 2D data for all people in a frame using get_human_state.
    Sorts people by distance (Closest First).
    """
    keypoints_list = []
    
    # Sort by depth (Closest first)
    if len(outputs) > 0:
        all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
        
        # argsort(all_depths) sorts ascending: Smallest depth (closest) -> Largest (furthest)
        sorted_indices = np.argsort(all_depths) 
        outputs_sorted = [outputs[i] for i in sorted_indices]
    else:
        outputs_sorted = []

    for pid, o in enumerate(outputs_sorted):
        # 1. Get 2D State using the provided function
        pos_human, facing_human = get_human_state(o)
        
        # Handle case where facing_human might be None (though rare with fallback)
        # Default to a generic "Forward/Up" vector [0, -1] if missing
        if facing_human is None:
             facing_human = np.array([0.0, -1.0], dtype=np.float32)

        # Filter Keypoints
        # Full set of 70 keypoints
        kp_3d_full = _to_np(o.get("pred_keypoints_3d", []))
        kp_3d_body = kp_3d_full[BODY_ONLY_INDICES]
        kp_2d_full = _to_np(o.get("pred_keypoints_2d", []))
        kp_2d_body = kp_2d_full[BODY_ONLY_INDICES]

        # 2. Pack Data
        keypoints_dict = {
            "id": pid,
            "bbox": _to_np(o["bbox"]),
            
            # 2D NAVIGATION DATA
            # pos_human: [Forward, Left] in Meters relative to robot
            "pos_robot": pos_human,       
            # facing_dir: [Map Right, Map Up] Normalized Vector
            "facing_dir": facing_human,     
            
            # POSE DATA (3D Keypoints are often still required for downstream tasks)
            "keypoints_2d": kp_2d_body,
            "keypoints_3d": kp_3d_body,
            "pred_cam_t": _to_np(o.get("pred_cam_t", [])),
        }
        keypoints_list.append(keypoints_dict)

    return keypoints_list



def format_keypoints_for_saving(outputs, frame_name):
    """
    Extracts relevant data for all people in a frame using get_human_state.
    Filters out hand (finger) keypoints, keeping only body/feet/wrists.
    """
    people_list = []
    
    # Sort by depth (closest first)
    if len(outputs) > 0:
        all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
        sorted_indices = np.argsort(all_depths) 
        outputs_sorted = [outputs[i] for i in sorted_indices]
    else:
        outputs_sorted = []

    for pid, o in enumerate(outputs_sorted):
        # 1. Get 2D State
        pos_human, facing_human = get_human_state(o)
        
        if facing_human is None:
             facing_human = np.array([0.0, -1.0], dtype=np.float32)

        # 2. Extract and Filter Keypoints
        # Full set of 70 keypoints
        kp_3d_full = _to_np(o.get("pred_keypoints_3d", []))
        
        # Filter to keep only the 30 body joints
        if len(kp_3d_full) > 0:
            kp_3d_body = kp_3d_full[BODY_ONLY_INDICES]
        else:
            kp_3d_body = []

        # 3. Pack Data
        person_dict = {
            "id": pid,
            "bbox": _to_np(o["bbox"]),
            
            # 2D NAVIGATION DATA
            "pos_robot": pos_human,       
            "facing_dir": facing_human,     
            
            # POSE DATA (Filtered: No Fingers)
            "keypoints_3d": kp_3d_body,
        }
        people_list.append(person_dict)

    return people_list