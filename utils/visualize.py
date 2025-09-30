# data_processing/visualizer.py

import os
import cv2
import pickle
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

class Visualizer:
    def __init__(self, config_obj):
        self.cfg = config_obj
        self.ref_cam_list = list(range(29, 31))
        self.cam_idx_list = list(range(31))

    def _load_stacked_file(self):
        self.input_pkl_path = os.path.join(
            self.cfg.TRACK_ANNOTATION_DIR,
            f'{self.cfg.SEQ_CLASS}/{self.ref_cam_id}/annotation.pkl'
        )
        print(f"Loading stacked data: {self.input_pkl_path}")
        if not os.path.exists(self.input_pkl_path):
            raise FileNotFoundError(f"Stacked pkl file not found: {self.input_pkl_path}")

        with open(self.input_pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 전체 데이터를 클래스 변수에 저장
        self.all_videos = data['video']
        self.all_coords = data['coords']
        self.all_visibility = data['visibility']
        print("✅ Stacked data loading completed.")

    @staticmethod
    def _draw_circle(rgb_image, coord, radius, color):
        draw = ImageDraw.Draw(rgb_image)
        draw.ellipse([
            (coord[0] - radius, coord[1] - radius),
            (coord[0] + radius, coord[1] + radius)]
        , fill=color, outline=color)
        return rgb_image

    @staticmethod
    def _draw_line(rgb_image, coord1, coord2, color, linewidth):
        draw = ImageDraw.Draw(rgb_image)
        draw.line((coord1[0], coord1[1], coord2[0], coord2[1]), fill=color, width=linewidth)
        return rgb_image

    def _prepare_visualization(self):
        first_img_np = self.images[0]
        h, w, _ = first_img_np.shape
        self.h, self.w = h, w

        initial_y_coords = self.tracks_2d[0, :, 1]
        sorted_indices = np.argsort(initial_y_coords)
        rainbow_palette = plt.cm.hsv(np.linspace(0, 1, self.N_QUERY))
        self.colors = np.zeros_like(rainbow_palette)
        self.colors[sorted_indices] = rainbow_palette

        self.fig, self.ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.axis('off')
        self.im_display = self.ax.imshow(np.zeros((h, w, 3), dtype=np.uint8))

    def _animate(self, t):
        img_np_bgr = self.images[t]
        pil_image = Image.fromarray(cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB))

        for i in range(self.N_QUERY):
            start_frame = max(1, t - 9)
            for tau in range(start_frame, t + 1):
                if self.visibility[tau - 1, i] and self.visibility[tau, i]:
                    pt1 = tuple(self.tracks_2d[tau - 1, i].astype(int))
                    pt2 = tuple(self.tracks_2d[tau, i].astype(int))
                    color_rgb = tuple(int(c * 255) for c in self.colors[i][:3])
                    pil_image = self._draw_line(pil_image, pt1, pt2, color_rgb, linewidth=2)
        
        for i in range(self.N_QUERY):
            if self.visibility[t, i]:
                coord = tuple(self.tracks_2d[t, i].astype(int))
                color_rgb = tuple(int(c * 255) for c in self.colors[i][:3])
                pil_image = self._draw_circle(pil_image, coord, radius=3, color=color_rgb)

        self.im_display.set_data(np.array(pil_image))
        self.pbar.update(1)
        return (self.im_display,)

    def run(self):
        for ref_cam_id in self.ref_cam_list:
            self.ref_cam_id = ref_cam_id
            self._load_stacked_file()

            for cam_data_idx, cam_id in enumerate(self.cam_idx_list):
                
                self.images = self.all_videos[cam_data_idx]
                self.tracks_2d = self.all_coords[cam_data_idx].transpose(1, 0, 2)
                self.visibility = self.all_visibility[cam_data_idx].transpose(1, 0)

                self.T = self.images.shape[0]
                self.N_QUERY = self.tracks_2d.shape[1]
                self.output_video_path = os.path.join(
                    self.cfg.TRACK_ANNOTATION_DIR,
                    f'{self.cfg.SEQ_CLASS}/{self.ref_cam_id}/{cam_id}.mp4'
                )
                
                self._prepare_visualization()
                self.pbar = tqdm(total=self.T, desc=f"Rendering for Cam {cam_id}")

                ani = FuncAnimation(self.fig, self._animate, frames=self.T, interval=50, blit=True)
                ani.save(self.output_video_path, writer='ffmpeg', fps=30)
                
                self.pbar.close()
                plt.close(self.fig)
            
        print("\nAll videos saved successfully!")