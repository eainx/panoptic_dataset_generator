# data_processing/metadata_generator.py
import os
import json
import numpy as np

class MetadataGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self._load_calibration()
    
    def _load_calibration(self):
        calib_path = os.path.join(self.cfg.CALIBRATION_DIR, f'calibration_{self.cfg.SEQ_NAME}.json')
        with open(calib_path) as f:
            cam_calib_data = json.load(f)
        self.hd_cameras = {int(cam['name'].split('_')[1]): cam for cam in cam_calib_data["cameras"] if cam['type'] == "hd"}

    def _generate_meta(self, camera_ids, save_path):
        """주어진 카메라 ID 목록에 대한 메타데이터 JSON 파일을 생성합니다."""
        k_list, w2c_list = [], []
        width, height = self.cfg.TARGET_IMG_SIZE

        scale_factor = 1.0 / 3.0

        for cam_id in camera_ids:
            cam_data = self.hd_cameras[cam_id]
            
            # Intrinsic matrix (K)
            K = np.array(cam_data['K']).copy()
            K[:2, :] *= scale_factor
            k_list.append(K)

            # Extrinsic matrix (W2C)
            R = np.array(cam_data['R'])
            t = np.array(cam_data['t']).reshape(3, 1) * 0.01 # cm -> m
            
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t.flatten()
            w2c_list.append(w2c)

        # 모든 프레임에 대해 동일한 카메라 파라미터 복제
        k_list_full = [[k.tolist() for k in k_list] for _ in range(self.cfg.NUM_FRAMES)]
        w2c_list_full = [[w.tolist() for w in w2c_list] for _ in range(self.cfg.NUM_FRAMES)]
        
        # 파일 이름 목록 생성
        fn_list = [
            [f"{cam_id}/{frame_idx:06d}.jpg" for cam_id in camera_ids]
            for frame_idx in range(self.cfg.NUM_FRAMES)
        ]

        meta = {
            "w": width, "h": height,
            "k": k_list_full, "w2c": w2c_list_full, "fn": fn_list
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(meta, f, separators=(', ', ':'))
        print(f"Saved meta file to: {save_path}")

    def run(self):
        """학습 및 테스트 메타데이터를 생성합니다."""
        print("\nGenerating metadata JSON files...")
        self._generate_meta(self.cfg.TRAIN_CAMS, self.cfg.TRAIN_META_PATH)