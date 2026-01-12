# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import cv2
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from PIL import Image, ImageDraw, ImageFont

from tools.save_utils import get_human_state

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)

def put_text_rect(
    img_bgr: np.ndarray,
    text: str,
    org_xy: tuple[int, int],
    font_path: str = DEFAULT_FONT_PATH,
    font_size: int = 14,
    color_bgr: tuple[int, int, int] = (0, 0, 0),
):
    """
    Draw TTF text on a BGR(OpenCV) image using PIL.
    """
    try:
        # Convert to RGB for PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            # Fallback to default if path is wrong
            font = ImageFont.load_default()

        # PIL draws text at top-left of org_xy
        draw.text(org_xy, text, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0]))

        # Convert back to BGR for OpenCV
        out_rgb = np.array(pil_img)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(out_bgr)
    except Exception as e:
        # Fallback to OpenCV font if PIL fails hard
        print(f"Font Error: {e}")
        cv2.putText(img_bgr, text, org_xy, cv2.FONT_HERSHEY_PLAIN, 1.0, color_bgr, 1, cv2.LINE_AA)
        return np.ascontiguousarray(img_bgr)

def visualize_sample(img_cv2, outputs, faces):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    rend_img = []
    for pid, person_output in enumerate(outputs):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img1 = visualizer.draw_skeleton(img_keypoints.copy(), keypoints_2d)

        img1 = cv2.rectangle(img1,(int(person_output["bbox"][0]), int(person_output["bbox"][1])),(int(person_output["bbox"][2]), int(person_output["bbox"][3])),(0, 255, 0),2)
        if "lhand_bbox" in person_output:
            img1 = cv2.rectangle(img1,(int(person_output["lhand_bbox"][0]), int(person_output["lhand_bbox"][1])),(int(person_output["lhand_bbox"][2]), int(person_output["lhand_bbox"][3])),(255, 0, 0),2)
        if "rhand_bbox" in person_output:
            img1 = cv2.rectangle(img1,(int(person_output["rhand_bbox"][0]), int(person_output["rhand_bbox"][1])),(int(person_output["rhand_bbox"][2]), int(person_output["rhand_bbox"][3])),(0, 0, 255),2)

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        img2 = (renderer(person_output["pred_vertices"], person_output["pred_cam_t"], img_mesh.copy(), mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1)) * 255)
        white_img = np.ones_like(img_cv2) * 255
        img3 = (renderer(person_output["pred_vertices"], person_output["pred_cam_t"], white_img, mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1), side_view=True) * 255)

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img

def render_topdown_positions(
    outputs_sorted,
    H: int,
    W: int,
    meters_per_px: float = None,   # if None, use fixed x_max_m=20m to set scale
    grid_step_m: float = 1.0,
    bold_step_m: float = 5.0,
    margin_px: int = 40,
    circle_px: int = 10,
    x_max_m: float = 20.5,         # FIXED forward range
    font_path: str = DEFAULT_FONT_PATH,
    T_cam_base: np.ndarray = None,
):
    panel = np.ones((H, W, 3), dtype=np.uint8) * 255
    ox, oy = W // 2, H - 1  # origin

    # -------- choose scale ----------
    if meters_per_px is None:
        usable_h = max(1, H - 2 * margin_px)
        px_per_m = usable_h / float(x_max_m)
    else:
        px_per_m = 1.0 / float(meters_per_px)

    y_max_m = max(0.0, (ox - margin_px) / px_per_m)

    def m_to_px(x_m, y_m):
        u = int(round(ox - y_m * px_per_m))  # +y left => u decreases
        v = int(round(oy - x_m * px_per_m))  # +x forward => v decreases
        return u, v

    # -------- draw grid ----------
    normal = (220, 220, 220)
    bold = (180, 180, 180)
    axis = (0, 0, 0)
    text_color = (120, 120, 120)

    # Draw y-grid (lines of constant y)
    y = -int(y_max_m // grid_step_m) * grid_step_m
    while y <= y_max_m + 1e-6:
        u0, v0 = m_to_px(0.0, y)
        u1, v1 = m_to_px(x_max_m, y)
        is_bold = (abs(y) % bold_step_m) < 1e-6
        cv2.line(panel, (u0, v0), (u1, v1), bold if is_bold else normal, 2 if is_bold else 1)

        if abs(y) > 1e-6 and is_bold:
            panel = put_text_rect(panel, f"{y:.0f}m", (u0 + 5, min(H - 10, v0 - 15)), font_path=font_path, font_size=16, color_bgr=text_color)
        y += grid_step_m

    # Draw x-grid (lines of constant x)
    x = 0.0
    while x <= x_max_m + 1e-6:
        u0, v0 = m_to_px(x, -y_max_m)
        u1, v1 = m_to_px(x,  y_max_m)
        is_bold = (x % bold_step_m) < 1e-6
        cv2.line(panel, (u0, v0), (u1, v1), bold if is_bold else normal, 2 if is_bold else 1)

        if x > 0 and is_bold:
            panel = put_text_rect(panel, f"{x:.0f}m", (max(5, u0 - 50), v0 + 5), font_path=font_path, font_size=16, color_bgr=text_color)
        x += grid_step_m

    # -------- draw axes ----------
    u0, v0 = ox, oy

    # +x axis (forward)
    u1, v1 = m_to_px(min(3.0, x_max_m), 0.0)
    cv2.arrowedLine(panel, (u0, v0), (u1, v1), axis, 2, tipLength=0.2)
    panel = put_text_rect(panel, "+x", (u1 + 10, max(20, v1 + 5)), font_path=font_path, font_size=18, color_bgr=axis)

    # +y axis (left)
    u1, v1 = m_to_px(0.0, min(3.0, y_max_m))
    cv2.arrowedLine(panel, (u0, v0), (u1, v1), axis, 2, tipLength=0.2)
    panel = put_text_rect(panel, "+y", (max(10, u1 + 10), oy - 20), font_path=font_path, font_size=18, color_bgr=axis)

    cv2.circle(panel, (ox, oy), 6, axis, -1)

    # -------- range rings ----------
    for r in [1, 2, 3, 5, 10, 15, 20]:
        r_px = int(round(r * px_per_m))
        if oy - r_px < margin_px:
            break
        cv2.circle(panel, (ox, oy), r_px, (210, 210, 210), 1)
        panel = put_text_rect(panel, f"{r}m", (ox + 6, oy - r_px - 15), font_path=font_path, font_size=14, color_bgr=(140, 140, 140))

    # --- Humans ---    
    for pid, o in enumerate(outputs_sorted):
        # 1. Get Physics Data (Robot Frame)
        pos_robot, facing_robot = get_human_state(o, T_cam_base)
        
        rx, ry = pos_robot[0], pos_robot[1]

        # 2. Filter
        if rx < 0 or rx > x_max_m: continue
        if abs(ry) > y_max_m: continue
        
        # 3. Convert Position to Pixels
        u, v = m_to_px(rx, ry)
        if not (0 <= u < W and 0 <= v < H): continue

        # 4. Draw Arrow (Convert Rotation to Pixels)
        if facing_robot is not None:
            # facing_robot = [fwd, left]
            # Forward (+X) -> Moves Up (-v)
            # Left (+Y)    -> Moves Left (-u)
            
            fwd_comp = facing_robot[0]
            left_comp = facing_robot[1]
            
            arrow_len_px = 30
            
            u_tip = int(u - left_comp * arrow_len_px) # Left means subtract u
            v_tip = int(v - fwd_comp * arrow_len_px)  # Forward means subtract v
            
            cv2.arrowedLine(panel, (u, v), (u_tip, v_tip), (0, 0, 0), 2, tipLength=0.3)

        cv2.circle(panel, (u, v), circle_px, (0, 0, 255), -1)
        cv2.circle(panel, (u, v), circle_px, (0, 0, 0), 2)
        panel = put_text_rect(panel, f"{pid}", (u + circle_px + 3, v - circle_px - 20), font_path, 22, (0, 0, 0))

    return panel


def visualize_sample_together(img_cv2, outputs, faces, T_cam_base):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    if not outputs:
        img_topdown = render_topdown_positions([], img_cv2.shape[0], img_cv2.shape[1], T_cam_base=T_cam_base)
        cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_topdown], axis=1)
        return cur_img

    all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(all_depths)]

    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate([keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1)
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t
    
    renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
    img_mesh = (renderer(all_pred_vertices, fake_pred_cam_t, img_mesh, mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1)) * 255)

    img_topdown = render_topdown_positions(outputs_sorted, img_cv2.shape[0], img_cv2.shape[1], T_cam_base=T_cam_base)

    # # Render side view
    # white_img = np.ones_like(img_cv2) * 255
    # img_mesh_side = (
    #     renderer(
    #         all_pred_vertices,
    #         fake_pred_cam_t,
    #         white_img,
    #         mesh_base_color=LIGHT_BLUE,
    #         scene_bg_color=(1, 1, 1),
    #         side_view=True,
    #     )
    #     * 255
    # )

    # cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_mesh_side], axis=1)
    cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_topdown], axis=1)

    return cur_img