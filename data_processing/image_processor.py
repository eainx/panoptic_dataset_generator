# data_processing/image_processor.py
import os
import json
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self._load_calibration()
        self._init_segmentation_model()

    def _load_calibration(self):
        calib_path = os.path.join(self.cfg.CALIBRATION_DIR, f'calibration_{self.cfg.SEQ_NAME}.json')
        with open(calib_path) as f:
            cam_calib_data = json.load(f)
        self.hd_cameras = [cam for cam in cam_calib_data["cameras"] if cam['type'] == "hd"]
    
    def _init_segmentation_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
        self.model.eval().to(self.device)
        self.preprocess = T.Compose([T.ToTensor()])
        print(f"Segmentation model loaded on {self.device}.")

    def _is_green_frame(self, img, threshold=0.8):
        """이미지가 녹색 스크린 프레임인지 확인합니다."""
        if img is None or img.size == 0: return True
        # 녹색 채널이 강하고 다른 채널은 약한 픽셀의 비율 계산
        green_ratio = ((img[:, :, 1] > 100) & (img[:, :, 0] < 50) & (img[:, :, 2] < 50)).sum()
        return (green_ratio / img.size) > threshold

    def process_and_resize_images(self):
        """
        원본 HD 이미지를 읽어 왜곡 보정, 리사이즈 후 저장합니다.
        녹색 프레임은 이전 프레임으로 대체합니다.
        """
        print("\nProcessing and resizing images...")
        os.makedirs(self.cfg.D3G_IMS_DIR, exist_ok=True)
        
        for cam_id in tqdm(self.cfg.ALL_CAMS, desc="Processing cameras for images"):
            cam_folder = os.path.join(self.cfg.D3G_IMS_DIR, f'{cam_id}')
            os.makedirs(cam_folder, exist_ok=True)

            cam_data = self.hd_cameras[cam_id]
            K = np.array(cam_data['K'])
            dist = np.array(cam_data['distCoef']).reshape(1, -1)
            
            prev_valid_img = None
            
            for i, frame_idx in enumerate(self.cfg.FRAME_INDICES):
                in_path = os.path.join(self.cfg.HD_IMGS_DIR, f'00_{cam_id:02d}', f'00_{cam_id:02d}_{frame_idx:08d}.jpg')
                out_path = os.path.join(cam_folder, f'{i:06d}.jpg')

                if not os.path.exists(in_path):
                    print(f"[Warning] Missing file: {in_path}")
                    continue

                img = cv2.imread(in_path)
                img_undistorted = cv2.undistort(img, K, dist)
                
                if self._is_green_frame(img_undistorted):
                    if prev_valid_img is not None:
                        print(f"[Warning] frame {frame_idx} is green : replaced")
                        img_to_save = prev_valid_img
                    else:
                        print(f"[Warning] First frame is green, cannot replace: {in_path}")
                        img_to_save = img_undistorted # 어쩔 수 없이 녹색 프레임 사용
                else:
                    img_to_save = img_undistorted
                    prev_valid_img = img_undistorted.copy()
                
                resized_img = cv2.resize(img_to_save, self.cfg.TARGET_IMG_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imwrite(out_path, resized_img)

    def generate_segmentation_masks(self):
        """리사이즈된 이미지에 대해 사람과 공 객체에 대한 마스크를 생성합니다."""
        print("\nGenerating segmentation masks...")
        os.makedirs(self.cfg.D3G_SEG_DIR, exist_ok=True)

        for cam_id in tqdm(self.cfg.ALL_CAMS, desc="Processing cameras for masks"):
            in_dir = os.path.join(self.cfg.D3G_IMS_DIR, f'{cam_id}')
            out_dir = os.path.join(self.cfg.D3G_SEG_DIR, f'{cam_id}')
            os.makedirs(out_dir, exist_ok=True)

            img_files = sorted([f for f in os.listdir(in_dir) if f.endswith('.jpg')])

            for img_file in img_files:
                img_path = os.path.join(in_dir, img_file)
                img = Image.open(img_path).convert("RGB")
                
                with torch.no_grad():
                    prediction = self.model([self.preprocess(img).to(self.device)])

                final_mask = np.zeros((img.height, img.width), dtype=np.uint8)
                
                scores = prediction[0]['scores'].cpu().numpy()
                labels = prediction[0]['labels'].cpu().numpy()
                masks = prediction[0]['masks'].cpu().numpy()

                high_conf_indices = np.where(scores > self.cfg.MASK_CONFIDENCE_THRESHOLD)[0]
                for i in high_conf_indices:
                    label = labels[i]
                    if label == self.cfg.PERSON_CLASS_ID or label == self.cfg.BALL_CLASS_ID \
                        or label == self.cfg.BAT_CLASS_ID or label == self.cfg.RACKET_CLASS_ID:
                        mask = (masks[i, 0] > 0.5).astype(np.uint8) * 255
                        final_mask = np.maximum(final_mask, mask)

                out_path = os.path.join(out_dir, os.path.splitext(img_file)[0] + '.png')
                cv2.imwrite(out_path, final_mask)
    
    def run(self):
        self.process_and_resize_images()
        self.generate_segmentation_masks()