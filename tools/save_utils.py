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

# def get_human_state(o, T_base_cam):
#     """
#     Extracts the human's position and facing direction strictly in the Robot Frame.
#     Robot Frame Definition: +X Forward, +Y Left, +Z Up.
    
#     Args:
#         o (dict): The output dictionary for a single person.
        
#     Returns:
#         tuple: (pos_robot, facing_robot)
#             pos_robot (np.ndarray): [x, y] in meters. 
#                                     x = Forward, y = Left.
#             facing_robot (np.ndarray): [dx, dy] normalized direction vector.
#                                        dx = Forward component, dy = Left component.
#     """
#     # 1. Position Extraction
#     # Camera: X=Right, Y=Down, Z=Forward
#     # Robot:  X=Forward, Y=Left
#     t = np.asarray(o["pred_cam_t"], dtype=np.float32).reshape(-1)
    
#     robot_x = float(t[2])       # Forward (Cam Z)
#     robot_y = float(-t[0])      # Left (Cam -X)
#     pos_robot = np.array([robot_x, robot_y])

#     # 2. Rotation Extraction
#     rot_matrix = None
#     if "pred_global_rots" in o:
#         rots = o["pred_global_rots"]
#         # Priority: Joint 1 (Pelvis/Root)
#         if len(rots.shape) == 4: 
#             rot_matrix = rots[0][1] # Batch 0, Joint 1
#         elif len(rots.shape) == 3: 
#             rot_matrix = rots[1]

#     # 3. Facing Direction Extraction
#     facing_robot = None
#     if rot_matrix is not None:
#         f_cam = forward_cam_from_rot(rot_matrix)
        
#         # Transform Camera Vector (Right, Down, Forward) to Robot Vector (Forward, Left)
#         # Cam X (Right) -> Robot -Y (Left)
#         # Cam Z (Forward) -> Robot +X (Forward) -> -X (Something related to SAM3D result)
        
#         dir_fwd = -float(f_cam[2])   # Robot X component
#         dir_left = -float(f_cam[0]) # Robot Y component
        
#         norm = (dir_fwd**2 + dir_left**2)**0.5 + 1e-8
#         if norm > 0.01:
#              facing_robot = np.array([dir_fwd / norm, dir_left / norm])
             
#     return pos_robot, facing_robot

def get_human_state(o, T_cam_base):
    """
    Extracts the human's position and facing direction in the Robot Frame
    using a rigid 3D transformation matrix, then projects to 2D.
    
    Args:
        o (dict): The output dictionary for a single person.
        T_base_cam (np.ndarray): 4x4 Transformation matrix from Base to Camera.
                                 (Pose of the Camera in the Robot Base Frame).
        
    Returns:
        tuple: (pos_robot, facing_robot)
            pos_robot (np.ndarray): [x, y] in meters (Robot Frame).
            facing_robot (np.ndarray): [dx, dy] normalized direction vector (Robot Frame).
    """
    if T_cam_base is None:
        T_base_cam = np.array([ # T from Base to Camera
            [ 0,  0,  1,  0],
            [-1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  0,  1]
        ])
    else: 
        T_base_cam = np.linalg.inv(T_cam_base)

    # 1. Position Extraction (3D Transform -> Project to 2D)
    # pred_cam_t is the [x, y, z] translation in Camera Frame
    t_cam = np.asarray(o["pred_cam_t"], dtype=np.float32).reshape(3)
    
    # Create homogeneous coordinate [x, y, z, 1]
    p_cam_homo = np.append(t_cam, 1.0)
    
    # Transform to Base Frame: P_base = T @ P_cam
    p_base = T_base_cam @ p_cam_homo
    
    # Project to XY plane (Robot Frame: X=Forward, Y=Left)
    pos_robot = p_base[:2] 

    # 2. Rotation Extraction
    rot_matrix = None
    if "pred_global_rots" in o:
        rots = o["pred_global_rots"]
        # Priority: Joint 1 (Pelvis/Root)
        if len(rots.shape) == 4: 
            rot_matrix = rots[0][1] # Batch 0, Joint 1
        elif len(rots.shape) == 3: 
            rot_matrix = rots[1]

    # 3. Facing Direction Extraction (3D Transform -> Project to 2D)
    facing_robot = None
    if rot_matrix is not None:
        # Get forward vector in Camera Frame (Assuming Body Local Forward is +Z)
        f_cam = forward_cam_from_rot(rot_matrix) # Returns shape (3,)
        
        # Transform Direction Vector to Base Frame
        # Vectors only need the Rotation part (3x3), not translation
        R_base_cam = T_base_cam[:3, :3]
        f_base = R_base_cam @ f_cam
        
        # Project to XY plane
        dir_x = -f_base[0]
        dir_y = f_base[1]
        
        # Normalize in 2D
        norm = (dir_x**2 + dir_y**2)**0.5 + 1e-8
        if norm > 0.01:
             facing_robot = np.array([dir_x / norm, dir_y / norm])
             
    return pos_robot, facing_robot

def format_keypoints_for_saving(outputs, T_cam_base = None):
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
        pos_human, facing_dir = get_human_state(o, T_cam_base)
        
        # Handle case where facing_human might be None (though rare with fallback)
        # Default to a generic "Forward/Up" vector [0, -1] if missing
        if facing_dir is None:
             facing_dir = np.array([0.0, -1.0], dtype=np.float32)

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
            "position": pos_human,       
            # facing_dir: [Map Right, Map Up] Normalized Vector
            "facing_dir": facing_dir,     
            
            # POSE DATA (3D Keypoints are often still required for downstream tasks)
            "keypoints_2d": kp_2d_body,
            "keypoints_3d": kp_3d_body,
            "pred_cam_t": _to_np(o.get("pred_cam_t", [])),
        }
        keypoints_list.append(keypoints_dict)

    return keypoints_list
