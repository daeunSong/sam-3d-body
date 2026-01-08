# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import cv2
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


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

        img1 = cv2.rectangle(
            img1,
            (int(person_output["bbox"][0]), int(person_output["bbox"][1])),
            (int(person_output["bbox"][2]), int(person_output["bbox"][3])),
            (0, 255, 0),
            2,
        )

        if "lhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["lhand_bbox"][0]),
                    int(person_output["lhand_bbox"][1]),
                ),
                (
                    int(person_output["lhand_bbox"][2]),
                    int(person_output["lhand_bbox"][3]),
                ),
                (255, 0, 0),
                2,
            )

        if "rhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["rhand_bbox"][0]),
                    int(person_output["rhand_bbox"][1]),
                ),
                (
                    int(person_output["rhand_bbox"][2]),
                    int(person_output["rhand_bbox"][3]),
                ),
                (0, 0, 255),
                2,
            )

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        img2 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_mesh.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_cv2) * 255
        img3 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        )

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
):
    panel = np.ones((H, W, 3), dtype=np.uint8) * 255
    ox, oy = W // 2, H - 1  # origin

    # -------- collect positions (robot_x, robot_y) ----------
    positions = []
    for o in outputs_sorted:
        t = np.asarray(o["pred_cam_t"], dtype=np.float32).reshape(-1)
        robot_x = float(t[2])       # forward
        robot_y = float(-t[0])      # left  (flip sign if needed)
        positions.append((robot_x, robot_y))

    # -------- choose scale ----------
    if meters_per_px is None:
        # FIX scale so that x_max_m always fits in height
        usable_h = max(1, H - 2 * margin_px)
        px_per_m = usable_h / float(x_max_m)
    else:
        px_per_m = 1.0 / float(meters_per_px)

    # visible lateral range implied by width at this scale
    y_max_m = max(0.0, (ox - margin_px) / px_per_m)

    # helper: meters -> pixel
    def m_to_px(x_m, y_m):
        u = int(round(ox - y_m * px_per_m))  # +y left => u decreases
        v = int(round(oy - x_m * px_per_m))  # +x forward => v decreases
        return u, v

    # -------- draw grid ----------
    normal = (220, 220, 220)
    bold = (180, 180, 180)
    axis = (0, 0, 0)

    # Draw y-grid (lines of constant y, across x in [0, x_max_m])
    y = -int(y_max_m // grid_step_m) * grid_step_m
    while y <= y_max_m + 1e-6:
        u0, v0 = m_to_px(0.0, y)
        u1, v1 = m_to_px(x_max_m, y)
        is_bold = (abs(y) % bold_step_m) < 1e-6
        cv2.line(panel, (u0, v0), (u1, v1), bold if is_bold else normal, 2 if is_bold else 1)

        if abs(y) > 1e-6 and is_bold:
            cv2.putText(panel, f"{y:.0f}m", (u0 + 5, min(H - 10, v0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        y += grid_step_m

    # Draw x-grid (lines of constant x, across y in [-y_max_m, y_max_m])
    x = 0.0
    while x <= x_max_m + 1e-6:
        u0, v0 = m_to_px(x, -y_max_m)
        u1, v1 = m_to_px(x,  y_max_m)
        is_bold = (x % bold_step_m) < 1e-6
        cv2.line(panel, (u0, v0), (u1, v1), bold if is_bold else normal, 2 if is_bold else 1)

        if x > 0 and is_bold:
            cv2.putText(panel, f"{x:.0f}m", (max(5, u0 - 60), v0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        x += grid_step_m

    # -------- draw axes ----------
    u0, v0 = ox, oy

    # +x axis (forward)
    u1, v1 = m_to_px(min(3.0, x_max_m), 0.0)
    cv2.arrowedLine(panel, (u0, v0), (u1, v1), axis, 2, tipLength=0.2)
    cv2.putText(panel, "+x", (u1 + 10, max(20, v1 + 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis, 2, cv2.LINE_AA)

    # +y axis (left)
    u1, v1 = m_to_px(0.0, min(3.0, y_max_m))
    cv2.arrowedLine(panel, (u0, v0), (u1, v1), axis, 2, tipLength=0.2)
    cv2.putText(panel, "+y", (max(10, u1 + 10), oy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis, 2, cv2.LINE_AA)

    cv2.circle(panel, (ox, oy), 6, axis, -1)

    # -------- range rings ----------
    for r in [1, 2, 3, 5, 10, 15, 20]:
        r_px = int(round(r * px_per_m))
        if oy - r_px < margin_px:
            break
        cv2.circle(panel, (ox, oy), r_px, (210, 210, 210), 1)
        cv2.putText(panel, f"{r}m", (ox + 6, oy - r_px - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

    # -------- humans ----------
    for pid, (robot_x, robot_y) in enumerate(positions):
        # optionally clamp to view window
        if robot_x < 0 or robot_x > x_max_m:
            continue
        if abs(robot_y) > y_max_m:
            continue

        u, v = m_to_px(robot_x, robot_y)
        if not (0 <= u < W and 0 <= v < H):
            continue
        cv2.circle(panel, (u, v), circle_px, (0, 0, 255), -1)
        cv2.circle(panel, (u, v), circle_px, (0, 0, 0), 2)
        cv2.putText(panel, f"{pid}", (u + circle_px + 2, v - circle_px - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # title (no scale text)
    # cv2.putText(panel, "Top-down human positions (robot frame)", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return panel


def visualize_sample_together(img_cv2, outputs, faces):
    # Render everything together
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    if not outputs:
        img_topdown = render_topdown_positions([], img_cv2.shape[0], img_cv2.shape[1])
        cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_topdown], axis=1)
        return cur_img

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    # Then, draw all keypoints.
    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    # Then, put all meshes together as one super mesh
    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # Pull out a fake translation; take the closest two
    fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t
    
    # Render front view
    renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            img_mesh,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        * 255
    )

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

    # NEW: render top-down 2D positions instead of side view
    img_topdown = render_topdown_positions(outputs_sorted, img_cv2.shape[0], img_cv2.shape[1])

    # cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_mesh_side], axis=1)
    cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_topdown], axis=1)

    return cur_img
