# data_processing/trajectory_filter.py
import os
import json
import cv2
import torch
import pickle
import numpy as np
import random
from tqdm import tqdm
from scipy.spatial import cKDTree
from utils.common_utils import transform_points

class TrajectoryFilter_ver2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.T = self.cfg.NUM_FRAMES
        self.cam_idx_list = list(range(31))
        self.ref_cam_list = list(range(29, 31))
        # self.cam_idx = self.cfg.TARGET_CAM_IDX
        
    def _load_data(self, cam_idx):
        self.cam_idx = cam_idx
        print(f"\nLoading data for trajectory filtering (Cam: {self.cam_idx})...")
        with open(self.cfg.TRAIN_META_PATH) as f:
            meta = json.load(f)
            # train_meta.json의 카메라 목록에서 cam_idx가 몇 번째인지 확인
            cam_list = [int(os.path.dirname(f)) for f in meta['fn'][0]]
            meta_cam_idx = cam_list.index(self.cam_idx)

            self.K = np.array(meta['k'][0][meta_cam_idx]).reshape(3, 3)
            self.w2c = np.array(meta['w2c'][0][meta_cam_idx]).reshape(4, 4)
            self.fx, self.fy, self.cx, self.cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        
        # 2. 3D GS 결과 로드
        params = dict(np.load(self.cfg.PARAMS_NPZ_PATH))
        params = {k: torch.tensor(v).float() for k, v in params.items()}
        is_fg = params['seg_colors'][:, 0] > 0.5
        self.gs_tracks_world = np.array([data.cpu().numpy() for data in params['means3D'][:, is_fg, :]])

        # 3. 파일 목록 로드
        self.seg_dir = os.path.join(self.cfg.D3G_SEG_DIR, str(self.cam_idx))
        self.depth_dir = self.cfg.DEPTH_DIR_TEMPLATE.format(cam_idx=self.cam_idx)
        self.rgb_dir = os.path.join(self.cfg.D3G_IMS_DIR, str(self.cam_idx))
        
        self.seg_fnames = sorted(os.listdir(self.seg_dir))
        self.depth_fnames = sorted(os.listdir(self.depth_dir))
        self.rgb_fnames = sorted(os.listdir(self.rgb_dir))

    def _select_initial_queries(self):
        print("Step 1: Selecting initial query points at t=0...")
        seg_t0 = cv2.imread(os.path.join(self.seg_dir, self.seg_fnames[0]), cv2.IMREAD_GRAYSCALE)
        depth_t0 = np.load(os.path.join(self.depth_dir, self.depth_fnames[0]))

        # 마스크 영역에서 쿼리 포인트 샘플링
        mask_indices = np.argwhere(seg_t0 > 128)
        if len(mask_indices) > self.cfg.N_QUERY_POINTS:
            sample_indices = np.random.choice(len(mask_indices), self.cfg.N_QUERY_POINTS, replace=False)
            mask_indices = mask_indices[sample_indices]
        
        # 2D 좌표와 깊이 값으로 3D 포인트 (카메라 좌표계) 생성
        us, vs = mask_indices[:, 1], mask_indices[:, 0]
        zs = depth_t0[vs, us]
        xs = (us - self.cx) * zs / self.fx
        ys = (vs - self.cy) * zs / self.fy
        initial_query_pts_cam = np.stack([xs, ys, zs], axis=-1)

        # t=0의 Gaussian들을 카메라 좌표계로 변환
        gs_pts_cam_t0 = transform_points(self.gs_tracks_world[0], self.w2c)
        
        # KD-Tree를 사용해 가장 가까운 Gaussian 매칭
        kdtree = cKDTree(gs_pts_cam_t0)
        _, matched_indices = kdtree.query(initial_query_pts_cam, k=1)
        
        # 매칭된 Gaussian의 궤적을 선택
        self.tracks_XYZ = self.gs_tracks_world[:, matched_indices, :]
        
        # 매칭된 Gaussian의 t=0 위치로 쿼리 포인트 조정
        matched_gs_cam_t0 = gs_pts_cam_t0[matched_indices]
        u_adj = (self.fx * matched_gs_cam_t0[:, 0] / matched_gs_cam_t0[:, 2]) + self.cx
        v_adj = (self.fy * matched_gs_cam_t0[:, 1] / matched_gs_cam_t0[:, 2]) + self.cy
        self.queries_xyt = np.stack([u_adj, v_adj, np.zeros(len(u_adj))], axis=-1)

    def _project_given_queries(self):
        pts_cam_t0 = transform_points(self.tracks_XYZ[0], self.w2c)
        z_cam = pts_cam_t0[:, 2]
        
        valid_depth_mask = z_cam > 1e-6
        u_proj = np.full(self.tracks_XYZ.shape[1], -1.0)
        v_proj = np.full(self.tracks_XYZ.shape[1], -1.0)

        u_proj[valid_depth_mask] = (self.fx * pts_cam_t0[valid_depth_mask, 0] / z_cam[valid_depth_mask]) + self.cx
        v_proj[valid_depth_mask] = (self.fy * pts_cam_t0[valid_depth_mask, 1] / z_cam[valid_depth_mask]) + self.cy
        
        return np.stack([u_proj, v_proj, np.zeros(len(u_proj))], axis=-1)

    def _compute_visibility_and_2d_tracks(self):
        print("Step 2: Computing 2D tracks and visibility...")
        N = self.queries_xyt.shape[0]
        self.tracks_2D_xyt = np.zeros((self.T, N, 2))
        self.visibility = np.zeros((self.T, N), dtype=bool)
        self.images = []
        
        h, w = self.cfg.TARGET_IMG_SIZE[1], self.cfg.TARGET_IMG_SIZE[0]

        for t in tqdm(range(self.T), desc="Processing frames"):
            depth = np.load(os.path.join(self.depth_dir, self.depth_fnames[t]))
            rgb = cv2.imread(os.path.join(self.rgb_dir, self.rgb_fnames[t]))
            self.images.append(rgb)
            
            # 3D 궤적 포인트를 카메라 좌표계로 변환
            pts_cam = transform_points(self.tracks_XYZ[t], self.w2c)
            z_cam = pts_cam[:, 2]
            
            # 이미지 평면에 투영
            valid_depth_mask = z_cam > 1e-6
            u_proj = np.full(N, -1.0)
            v_proj = np.full(N, -1.0)
            u_proj[valid_depth_mask] = (self.fx * pts_cam[valid_depth_mask, 0] / z_cam[valid_depth_mask]) + self.cx
            v_proj[valid_depth_mask] = (self.fy * pts_cam[valid_depth_mask, 1] / z_cam[valid_depth_mask]) + self.cy
            self.tracks_2D_xyt[t] = np.stack([u_proj, v_proj], axis=-1)
            
            # 가시성 계산
            in_bounds = (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h)
            vis_mask = valid_depth_mask & in_bounds
            
            u_int, v_int = u_proj[vis_mask].astype(int), v_proj[vis_mask].astype(int)
            depth_from_map = depth[v_int, u_int]
            
            depth_diff = np.abs(z_cam[vis_mask] - depth_from_map)
            vis_mask[vis_mask] = depth_diff < self.cfg.VISIBILITY_DELTA
            self.visibility[t] = vis_mask
        

    def _filter_tracks(self):
        print("Step 3: Filtering tracks...")
        N = self.queries_xyt.shape[0]

        # 필터 1: 전경(Foreground)에 충분히 나타나는가?
        min_fg_frames = int(self.T * self.cfg.MIN_VISIBILITY_RATIO)
        fg_counts = np.zeros(N)
        for t in tqdm(range(self.T), desc="  - Checking foreground presence"):
            seg_mask = cv2.imread(os.path.join(self.seg_dir, self.seg_fnames[t]), cv2.IMREAD_GRAYSCALE)
            h, w = seg_mask.shape
            
            visible_at_t = self.visibility[t]
            coords = self.tracks_2D_xyt[t, visible_at_t].astype(int)
            u, v = coords[:, 0], coords[:, 1]
            
            in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            is_in_fg = seg_mask[v[in_bounds], u[in_bounds]] > 128
            
            visible_indices = np.where(visible_at_t)[0]
            fg_indices = visible_indices[in_bounds][is_in_fg]
            fg_counts[fg_indices] += 1
        
        fg_mask = fg_counts >= min_fg_frames
        print(f"    - Found {np.sum(fg_mask)} tracks in foreground for >= {min_fg_frames} frames.")

        # 필터 2: 충분히 움직이는가?
        displacements = self.tracks_2D_xyt[1:] - self.tracks_2D_xyt[:-1]
        distances = np.linalg.norm(displacements, axis=2)
        motion_vis_mask = self.visibility[:-1] & self.visibility[1:]
        path_length = np.sum(distances * motion_vis_mask, axis=0)
        motion_mask = path_length > self.cfg.MIN_MOTION_THRESHOLD_PX
        print(f"    - Found {np.sum(motion_mask)} tracks that moved > {self.cfg.MIN_MOTION_THRESHOLD_PX} pixels.")

        # 최종 필터링
        final_mask = fg_mask & motion_mask
        print(f"Total tracks kept: {np.sum(final_mask)} / {N}")
    
        # choose random 50
        valid_indices = np.where(final_mask)[0]
        # assert len(valid_indices) > self.cfg.N_QUERY_POINTS_RANDOM
        if len(valid_indices) > self.cfg.N_QUERY_POINTS_RANDOM:
            sampled_indices = np.random.choice(valid_indices, size=self.cfg.N_QUERY_POINTS_RANDOM, replace=False)
            new_mask = np.zeros_like(final_mask, dtype=bool)
            new_mask[sampled_indices] = True
            final_mask = new_mask

        # 필터링된 결과 저장
        self.queries_xyt = self.queries_xyt[final_mask]
        self.tracks_XYZ = self.tracks_XYZ[:, final_mask, :]
        self.tracks_2D_xyt = self.tracks_2D_xyt[:, final_mask, :]
        self.visibility = self.visibility[:, final_mask]
        
    def _save_stacked_results(self, REF_CAM_IDX, tracks_XYZ, video_list, coords_list, visibility_list, w2c_list, intrinsics_list):
        output_filepath = os.path.join(self.cfg.TRACK_ANNOTATION_DIR, f'{self.cfg.SEQ_CLASS}/{REF_CAM_IDX}')
        os.makedirs(output_filepath, exist_ok=True)
        output_filename = os.path.join(output_filepath, 'annotation.pkl')
        print(f"\nStacking and saving all data to {output_filename}...")

        output = {
            "video": np.stack(video_list, axis=0), # (V, T, H, W, 3)
            "coords": np.stack(coords_list, axis=0), # (V, N, T, 2)
            "visibility": np.stack(visibility_list, axis=0), # (V, N, T)
            "tracks_XYZ": tracks_XYZ, # (T, N, 3)
            "w2c": np.stack(w2c_list, axis=0), # (V, 4, 4)
            "fx_fy_cx_cy": np.stack(intrinsics_list, axis=0), # (V, 4)
        }

        with open(output_filename, 'wb') as f:
            pickle.dump(output, f)
        print("Save complete.")

    def run(self):
        for REF_CAM_IDX in self.ref_cam_list:
            print(f"--- Step 1: Selecting and Filtering 3D Tracks using Camera {REF_CAM_IDX} ---")
            self._load_data(cam_idx=REF_CAM_IDX)
            self._select_initial_queries()
            self._compute_visibility_and_2d_tracks() 
            self._filter_tracks()

            tracks_XYZ = np.copy(self.tracks_XYZ)
            print(f"\n--- Kept {tracks_XYZ.shape[1]} filtered tracks. Now processing all cameras. ---")

            video_list, coords_list, visibility_list, w2c_list, intrinsics_list = [], [], [], [], []

            for c in self.cam_idx_list:
                self._load_data(cam_idx=c)
                # set query xyz
                self.tracks_XYZ = np.copy(tracks_XYZ)   
                # compute visibility
                self.queries_xyt = self._project_given_queries()
                self._compute_visibility_and_2d_tracks()
                # (T, H, W, 3)
                video_list.append(np.stack(self.images, axis=0))  
                # (T, N, 2) -> (N, T, 2)
                coords_list.append(self.tracks_2D_xyt.transpose(1, 0, 2))
                # (T, N) -> (N, T)
                visibility_list.append(self.visibility.transpose(1, 0))
                # (4, 4)
                w2c_list.append(self.w2c)
                # (4,)
                intrinsics_list.append(np.array([self.fx, self.fy, self.cx, self.cy]))

            self._save_stacked_results(
                REF_CAM_IDX,
                tracks_XYZ,
                video_list,
                coords_list,
                visibility_list,
                w2c_list,
                intrinsics_list
            )