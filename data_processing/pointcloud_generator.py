# data_processing/pointcloud_generator.py
import os
import json
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from utils.common_utils import read_depth_from_dat
from utils.camera_utils import unproject_depth_to_hd, find_nearest_hd_camera, interpolate_colors

class PointCloudGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self._load_calibrations()

    def _load_calibrations(self):
        print("Loading calibration files...")
        calib_path = self.cfg.CALIBRATION_DIR
        with open(os.path.join(calib_path, f'kcalibration_{self.cfg.SEQ_NAME}.json'), 'r') as f:
            self.k_calib = json.load(f)
        with open(os.path.join(calib_path, f'calibration_{self.cfg.SEQ_NAME}.json'), 'r') as f:
            self.p_calib = json.load(f)
        with open(os.path.join(calib_path, f'ksynctables_{self.cfg.SEQ_NAME}.json'), 'r') as f:
            self.k_sync = json.load(f)
        with open(os.path.join(calib_path, f'synctables_{self.cfg.SEQ_NAME}.json'), 'r') as f:
            self.p_sync = json.load(f)
        print("Calibrations loaded.")

    def _get_synced_depth_idx(self, k_node_idx, univ_time):
        k_name = f'KINECTNODE{k_node_idx}'
        depth_times = np.array(self.k_sync['kinect']['depth'][k_name]['univ_time'])
        return np.argmin(np.abs(univ_time - depth_times))

    def run(self):
        print("\nStarting Point Cloud Generation...")
        hd_frame_idx = self.cfg.INITIAL_PT_CLOUD_FRAME
        
        univ_time = self.p_sync['hd']['univ_time'][hd_frame_idx + 2]
        print(f"Processing HD Frame Index: {hd_frame_idx}, Universal Time: {univ_time:.3f}")

        all_points_world = []
        all_colors = []

        try:
            processed_frame_idx = list(self.cfg.FRAME_INDICES).index(hd_frame_idx)
        except ValueError:
            print(f"Error: INITIAL_PT_CLOUD_FRAME {hd_frame_idx} not found in FRAME_INDICES.")
            return

        for k_node_idx in tqdm(range(1, 11), desc="Processing Kinect Nodes"):
            d_idx = self._get_synced_depth_idx(k_node_idx, univ_time)

            nearest_hd_cam = find_nearest_hd_camera(k_node_idx, self.k_calib, self.p_calib, self.cfg.TRAIN_CAMS)
            if nearest_hd_cam is None:
                print(f"Warning: No suitable HD camera found for Kinect Node {k_node_idx}")
                continue

            # 3. 데이터 로드 (HD 이미지, 깊이 이미지)
            hd_img_path = os.path.join(self.cfg.D3G_IMS_DIR, f'{nearest_hd_cam}', f'{processed_frame_idx:06d}.jpg')
            depth_file_path = os.path.join(self.cfg.KINECT_DEPTH_DIR, f'KINECTNODE{k_node_idx}', 'depthdata.dat')
            
            hd_img = cv2.imread(hd_img_path)
            if hd_img is None: continue
            hd_img_rgb = cv2.cvtColor(hd_img, cv2.COLOR_BGR2RGB)
            
            depth_img = read_depth_from_dat(depth_file_path, d_idx + 1)
            if depth_img is None: continue
            depth_img_sub = depth_img[::2, ::2]
            # ==============================================================================

            _, points_world, points_2d_hd = unproject_depth_to_hd(
                depth_img_sub, self.k_calib, self.p_calib, k_node_idx, nearest_hd_cam
            )

            colors_interp = interpolate_colors(hd_img_rgb.astype(np.float32) / 255.0, points_2d_hd[:, 0], points_2d_hd[:, 1])

            valid_mask = (depth_img_sub.flatten() != 0) & ~np.isnan(points_world).any(axis=1) & ~np.isnan(colors_interp).any(axis=1)
            if not np.any(valid_mask): continue
                
            all_points_world.append(points_world[valid_mask])
            all_colors.append(colors_interp[valid_mask])

        if not all_points_world:
            print("Error: No points generated. Exiting.")
            return

        final_points = np.vstack(all_points_world)
        final_colors = np.vstack(all_colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_points)
        pcd.colors = o3d.utility.Vector3dVector(final_colors)

        os.makedirs(self.cfg.PLY_OUTPUT_DIR, exist_ok=True)
        ply_path = os.path.join(self.cfg.PLY_OUTPUT_DIR, f'ptcloud_hd{hd_frame_idx:08d}.ply')
        o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)
        print(f"Saved merged point cloud to {ply_path}")
        
        seg_info = np.ones((final_points.shape[0], 1), dtype=np.float32)
        data_to_save = np.hstack((final_points, final_colors, seg_info))
        np.savez(self.cfg.INIT_PT_CLD_NPZ_PATH, data=data_to_save)
        print(f"Saved initial point cloud to: {self.cfg.INIT_PT_CLD_NPZ_PATH}")