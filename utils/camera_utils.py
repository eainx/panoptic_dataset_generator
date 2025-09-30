# utils/camera_utils.py
import numpy as np
import cv2
from scipy.ndimage import map_coordinates
from .common_utils import transform_points

def unproject_depth_to_hd(depth_im, k_calib, p_calib, k_node_idx, hd_cam_idx):
    h, w = depth_im.shape
    
    # -- 1. 파라미터 로드 --
    k_sensor_data = k_calib["sensors"][k_node_idx - 1]
    K_depth = np.array(k_sensor_data['K_depth'])
    
    p_kinect_color_cams = [c for c in p_calib["cameras"] if c['type'] == "kinect-color"]
    p_kinect_color_data = p_kinect_color_cams[k_node_idx - 1]

    p_hd_cams = [c for c in p_calib["cameras"] if c['type'] == "hd"]
    p_hd_data = p_hd_cams[hd_cam_idx]
    K_hd = np.array(p_hd_data['K'])

    # -- 2. 리사이즈된 이미지에 맞게 카메라의 K 행렬 스케일링 --
    K_depth_scaled = K_depth.copy()
    K_depth_scaled[0, :] *= 0.5 
    K_depth_scaled[1, :] *= 0.5

    K_hd_scaled = K_hd.copy()
    K_hd_scaled[0, :] *= (1.0 / 3.0)
    K_hd_scaled[1, :] *= (1.0 / 3.0)

    # -- 3. 좌표계 변환 행렬 계산 (이전과 동일) --
    T_world_to_hd = np.eye(4)
    T_world_to_hd[:3, :3] = np.array(p_hd_data['R'])
    T_world_to_hd[:3, 3] = np.array(p_hd_data['t']).flatten() * 0.01

    T_world_to_kinect_color = np.eye(4)
    T_world_to_kinect_color[:3, :3] = np.array(p_kinect_color_data['R'])
    T_world_to_kinect_color[:3, 3] = np.array(p_kinect_color_data['t']).flatten() * 0.01

    M_color_to_sensor = np.array(k_sensor_data['M_color'])
    M_depth_to_sensor = np.array(k_sensor_data['M_depth'])
    T_depth_to_color = np.linalg.inv(M_color_to_sensor) @ M_depth_to_sensor

    T_kinect_color_to_world = np.linalg.inv(T_world_to_kinect_color)
    T_kinect_depth_to_world = T_kinect_color_to_world @ T_depth_to_color

    # -- 4. 깊이 이미지 Unprojection --
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.vstack((x.flatten(), y.flatten())).T.astype(np.float32)
    undistorted_pixels = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K_depth_scaled, None).reshape(-1, 2)
    
    z_cam = depth_im.flatten() * 0.001
    x_cam = undistorted_pixels[:, 0] * z_cam
    y_cam = undistorted_pixels[:, 1] * z_cam
    points3d_depth_cam = np.vstack((x_cam, y_cam, z_cam)).T

    # -- 5. World 변환 및 HD 카메라로 투영 --
    points3d_panoptic_world = transform_points(points3d_depth_cam, T_kinect_depth_to_world)
    points3d_hd_cam = transform_points(points3d_panoptic_world, T_world_to_hd)

    # 스케일링된 K_hd_scaled 행렬로 투영
    points2d_hd = (K_hd_scaled @ points3d_hd_cam.T).T
    points2d_hd = points2d_hd[:, :2] / points2d_hd[:, 2:3]
    
    return points3d_depth_cam, points3d_panoptic_world, points2d_hd[:, :2]

# def unproject_depth_to_hd(depth_im, k_calib, p_calib, k_node_idx, hd_cam_idx):
#     """
#     Kinect 깊이 이미지를 3D 포인트 클라우드로 변환하고, 지정된 HD 카메라 뷰로 투영합니다.

#     Args:
#         depth_im (np.ndarray): 깊이 이미지 (H, W), 단위: mm
#         k_calib (dict): Kinect 보정 데이터 (kcalibration.json)
#         p_calib (dict): Panoptic 보정 데이터 (calibration.json)
#         k_node_idx (int): Kinect 노드 인덱스 (1-based)
#         hd_cam_idx (int): HD 카메라 인덱스

#     Returns:
#         tuple: (points3d_depth_cam, points3d_panoptic_world, points2d_hd)
#     """
#     h, w = depth_im.shape
    
#     # -- 1. 파라미터 로드 --
#     k_sensor_data = k_calib["sensors"][k_node_idx - 1]
#     K_depth = np.array(k_sensor_data['K_depth'])
    
#     p_kinect_color_cams = [c for c in p_calib["cameras"] if c['type'] == "kinect-color"]
#     p_kinect_color_data = p_kinect_color_cams[k_node_idx - 1]

#     p_hd_cams = [c for c in p_calib["cameras"] if c['type'] == "hd"]
#     p_hd_data = p_hd_cams[hd_cam_idx]
#     K_hd = np.array(p_hd_data['K'])

#     # -- 2. 좌표계 변환 행렬 계산 --
#     # Panoptic World -> HD Camera
#     T_world_to_hd = np.eye(4)
#     T_world_to_hd[:3, :3] = np.array(p_hd_data['R'])
#     T_world_to_hd[:3, 3] = np.array(p_hd_data['t']).flatten() * 0.01  # cm -> m

#     # Panoptic World -> Kinect Color Camera
#     T_world_to_kinect_color = np.eye(4)
#     T_world_to_kinect_color[:3, :3] = np.array(p_kinect_color_data['R'])
#     T_world_to_kinect_color[:3, 3] = np.array(p_kinect_color_data['t']).flatten() * 0.01

#     # Depth Cam -> Color Cam (in Kinect local space)
#     M_color_to_sensor = np.array(k_sensor_data['M_color'])
#     M_depth_to_sensor = np.array(k_sensor_data['M_depth'])
#     T_depth_to_color = np.linalg.inv(M_color_to_sensor) @ M_depth_to_sensor

#     # 최종 변환: Kinect Depth -> Panoptic World
#     T_kinect_color_to_world = np.linalg.inv(T_world_to_kinect_color)
#     T_kinect_depth_to_world = T_kinect_color_to_world @ T_depth_to_color

#     # -- 3. 깊이 이미지 Unprojection --
#     x, y = np.meshgrid(np.arange(w), np.arange(h))
#     pixels = np.vstack((x.flatten(), y.flatten())).T.astype(np.float32)
#     undistorted_pixels = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K_depth, None).reshape(-1, 2)
    
#     z_cam = depth_im.flatten() * 0.001  # mm -> m
#     x_cam = undistorted_pixels[:, 0] * z_cam
#     y_cam = undistorted_pixels[:, 1] * z_cam
#     points3d_depth_cam = np.vstack((x_cam, y_cam, z_cam)).T

#     # -- 4. World 좌표계로 변환 및 HD 카메라로 투영 --
#     points3d_panoptic_world = transform_points(points3d_depth_cam, T_kinect_depth_to_world)
#     points3d_hd_cam = transform_points(points3d_panoptic_world, T_world_to_hd)

#     # HD 카메라로 투영
#     points2d_hd = (K_hd @ points3d_hd_cam.T).T
#     points2d_hd = points2d_hd[:, :2] / points2d_hd[:, 2:3]
    
#     return points3d_depth_cam, points3d_panoptic_world, points2d_hd[:, :2]

def find_nearest_hd_camera(k_node_idx, k_calib, p_calib, available_hd_cams):
    # Kinect Depth 카메라의 World 좌표계 변환 행렬 계산
    p_kinect_color_cams = [c for c in p_calib["cameras"] if c['type'] == "kinect-color"]
    p_kinect_color_data = p_kinect_color_cams[k_node_idx - 1]
    
    T_world_to_kinect_color = np.eye(4)
    T_world_to_kinect_color[:3, :3] = np.array(p_kinect_color_data['R'])
    T_world_to_kinect_color[:3, 3] = np.array(p_kinect_color_data['t']).flatten() * 0.01
    T_kinect_color_to_world = np.linalg.inv(T_world_to_kinect_color)

    k_sensor_data = k_calib["sensors"][k_node_idx - 1]
    M_color_to_sensor = np.array(k_sensor_data['M_color'])
    M_depth_to_sensor = np.array(k_sensor_data['M_depth'])
    T_depth_to_color_local = np.linalg.inv(M_color_to_sensor) @ M_depth_to_sensor
    
    T_kinect_depth_to_world = T_kinect_color_to_world @ T_depth_to_color_local
    kinect_origin_world = T_kinect_depth_to_world[:3, 3]

    # 각 HD 카메라와의 거리 계산
    distances = []
    hd_cams_data = [cam for cam in p_calib['cameras'] if cam['type'] == 'hd']
    for cam_data in hd_cams_data:
        cam_id = int(cam_data['name'].split('_')[1])
        if cam_id in available_hd_cams:
            R = np.array(cam_data['R'])
            t = np.array(cam_data['t']).flatten() * 0.01
            hd_origin_world = -R.T @ t
            dist = np.linalg.norm(kinect_origin_world - hd_origin_world)
            distances.append((cam_id, dist))

    distances.sort(key=lambda x: x[1])
        
    return distances[0][0] if distances else None
    
def interpolate_colors(image, coords_x, coords_y):
    h, w = image.shape[:2]
    coords = np.array([coords_y.flatten(), coords_x.flatten()])
    
    mask = (coords[0] >= 0) & (coords[0] < h) & (coords[1] >= 0) & (coords[1] < w)
    
    output_shape = coords_x.shape
    n_channels = image.shape[2]
    
    interpolated_values = np.full((len(coords_x.flatten()), n_channels), np.nan)
    valid_coords = coords[:, mask]

    for c in range(n_channels):
        channel_values = map_coordinates(image[:, :, c], valid_coords, order=1, mode='constant', cval=np.nan)
        interpolated_values[mask, c] = channel_values
        
    return interpolated_values.reshape(output_shape + (n_channels,))
