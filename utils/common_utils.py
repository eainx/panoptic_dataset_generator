# utils/common_utils.py
import numpy as np

def read_depth_from_dat(file_path, frame_idx, width=512, height=424):
    """
    .dat 파일에서 특정 프레임의 깊이 데이터를 읽습니다.

    Args:
        file_path (str): .dat 파일 경로
        frame_idx (int): 읽어올 프레임의 1-based 인덱스
        width (int): 깊이 이미지 너비
        height (int): 깊이 이미지 높이

    Returns:
        np.ndarray: (H, W) 형태의 깊이 이미지 (단위: mm), 실패 시 None
    """
    bytes_per_pixel = 2  # uint16
    frame_size_in_bytes = width * height * bytes_per_pixel
    
    try:
        with open(file_path, 'rb') as f:
            f.seek(frame_size_in_bytes * (frame_idx - 1))
            data = np.fromfile(f, dtype=np.uint16, count=width * height)
            if data.size < width * height:
                print(f"Warning: Could not read full depth frame {frame_idx} from {file_path}")
                return None
    except FileNotFoundError:
        print(f"Error: Depth file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading depth file: {e}")
        return None

    depth_im = data.reshape((height, width)).astype(np.float64)
    return np.fliplr(depth_im) # 수평 뒤집기

def transform_points(points, transform_matrix):
    assert points.shape[1] == 3, "Input points should be (N, 3)"
    points_h = np.hstack((points, np.ones((points.shape[0], 1)))) # (N, 4)
    points_transformed_h = points_h @ transform_matrix.T
    # Normalize by the homogeneous coordinatex
    return points_transformed_h[:, :3]
