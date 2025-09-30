import os

class BaseConfig:
    # -- 입력 경로 --
    PANOPTIC_TOOLBOX_ROOT = "/media/ssd3/panoptic-toolbox"
    DYNAMIC_3D_GAUSSIANS_ROOT = "/media/ssd3/Dynamic3DGaussians/data"
    OUTPUT_ROOT = "/media/ssd4/sh/output"

    # -- 포인트 클라우드 생성 파라미터 --
    REMOVE_FLOOR = True
    FLOOR_HEIGHT_THRESHOLD_CM = 0.5
    REMOVE_WALLS = True

    # -- 궤적 필터링 파라미터 --
    N_QUERY_POINTS = 500
    N_QUERY_POINTS_RANDOM = 50
    VISIBILITY_DELTA = 0.1
    MIN_VISIBILITY_RATIO = 0.75
    MIN_MOTION_THRESHOLD_PX = 10.0

    # -- 마스크 생성 파라미터 --
    MASK_CONFIDENCE_THRESHOLD = 0.7
    PERSON_CLASS_ID = 1
    BALL_CLASS_ID = 37
    BAT_CLASS_ID = 39
    RACKET_CLASS_ID = 43
    
    # ==============================================================================

    def __init__(self):
        required_vars = ['SEQ_NAME', 'SEQ_CLASS', 'EXP_NUM', 'FRAME_INDICES']
        if not all(hasattr(self, var) for var in required_vars):
            raise NotImplementedError("Child config class must define SEQ_NAME, SEQ_CLASS, EXP_NUM, and FRAME_INDICES.")

        self.NUM_FRAMES = len(self.FRAME_INDICES)
        
        # -- 카메라 설정 --
        # No test cam setting
        self.ALL_CAMS = range(31)
        self.TRAIN_CAMS = [i for i in self.ALL_CAMS]
        self.TARGET_IMG_SIZE = (640, 360)  # (너비, 높이)

        # Panoptic Toolbox 경로
        self.CALIBRATION_DIR = os.path.join(self.PANOPTIC_TOOLBOX_ROOT, self.SEQ_NAME)
        self.HD_IMGS_DIR = os.path.join(self.CALIBRATION_DIR, 'hdImgs')
        self.KINECT_DEPTH_DIR = os.path.join(self.CALIBRATION_DIR, 'kinect_shared_depth')
        
        # Dynamic3DGaussians 데이터 경로
        d3g_data_dir = os.path.join(self.DYNAMIC_3D_GAUSSIANS_ROOT, self.SEQ_CLASS)
        self.D3G_DATA_DIR = d3g_data_dir
        self.D3G_IMS_DIR = os.path.join(d3g_data_dir, 'ims')
        self.D3G_SEG_DIR = os.path.join(d3g_data_dir, 'seg')
        os.makedirs(self.D3G_DATA_DIR, exist_ok=True)
        
        # 실험 결과 경로
        exp_output_dir = os.path.join(self.OUTPUT_ROOT, f'exp{self.EXP_NUM}', self.SEQ_CLASS)
        self.PARAMS_NPZ_PATH = os.path.join(exp_output_dir, 'params.npz')
        self.DEPTH_DIR_TEMPLATE = os.path.join(self.OUTPUT_ROOT, f'exp{self.EXP_NUM}', self.SEQ_CLASS, '{cam_idx}', 'depth')
        
        # 최종 출력 경로
        self.PLY_OUTPUT_DIR = os.path.join(self.CALIBRATION_DIR, 'kinoptic_ptclouds_py')
        self.INIT_PT_CLD_NPZ_PATH = os.path.join(d3g_data_dir, 'init_pt_cld.npz')
        self.TRAIN_META_PATH = os.path.join(d3g_data_dir, 'train_meta.json')
        self.TEST_META_PATH = os.path.join(d3g_data_dir, 'test_meta.json')
        self.TRACK_ANNOTATION_DIR = f'/media/ssd3/dataset_panoptic/track_annotation/{self.EXP_NUM}'
        os.makedirs(self.TRACK_ANNOTATION_DIR, exist_ok=True)

# ==============================================================================
# ==                       실험별 설정 클래스 정의                            ==
# ==============================================================================

class Dance2Config_seq1(BaseConfig):
    SEQ_NAME = "170307_dance2"
    SEQ_CLASS = "dance2_seq1"
    EXP_NUM = 1
    FRAME_INDICES = range(5400, 5550)
    INITIAL_PT_CLOUD_FRAME = 5400
    EASY_CAMS = [0, 11, 12, 13, 20, 24, 25, 28]
    MEDIUM_CAMS = [2, 5, 6, 7, 9, 22, 26, 29]
    HARD_CAMS = [1, 3, 4, 8, 14, 16, 18, 23]

class Pose1Config_seq1(BaseConfig):
    SEQ_NAME = "171204_pose1"
    SEQ_CLASS = "pose1_seq1"
    EXP_NUM = 1
    FRAME_INDICES = range(340, 490)
    INITIAL_PT_CLOUD_FRAME = 340
    EASY_CAMS = [0, 11, 13, 17, 21, 24, 25, 28]
    MEDIUM_CAMS = [2, 5, 6, 8, 12, 20, 26, 27]
    HARD_CAMS = [1, 3, 4, 7, 8, 14, 15, 30]

class Pose1Config_seq2(BaseConfig):
    SEQ_NAME = "171204_pose1"
    SEQ_CLASS = "pose1_seq2"
    EXP_NUM = 1
    FRAME_INDICES = range(811, 961)
    INITIAL_PT_CLOUD_FRAME = 811
    EASY_CAMS = [0, 11, 13, 17, 21, 24, 25, 28]
    MEDIUM_CAMS = [2, 5, 6, 8, 12, 20, 26, 27]
    HARD_CAMS = [1, 3, 4, 7, 8, 14, 15, 30]
    
class Pose1Config_seq3(BaseConfig):
    SEQ_NAME = "171204_pose1"
    SEQ_CLASS = "pose1_seq3"
    EXP_NUM = 1
    FRAME_INDICES = range(1710, 1860)
    INITIAL_PT_CLOUD_FRAME = 1710
    EASY_CAMS = [0, 11, 13, 17, 21, 24, 25, 28]
    MEDIUM_CAMS = [2, 5, 6, 8, 12, 20, 26, 27]
    HARD_CAMS = [1, 3, 4, 7, 8, 14, 15, 30]

class Pose1Config_seq4(BaseConfig):
    SEQ_NAME = "171204_pose1"
    SEQ_CLASS = "pose1_seq4"
    EXP_NUM = 1
    FRAME_INDICES = range(6235, 6385)
    INITIAL_PT_CLOUD_FRAME = 6235
    EASY_CAMS = [0, 11, 13, 17, 21, 24, 25, 28]
    MEDIUM_CAMS = [2, 5, 6, 8, 12, 20, 26, 27]
    HARD_CAMS = [1, 3, 4, 7, 8, 14, 15, 30]

class Pose1Config_seq5(BaseConfig):
    SEQ_NAME = "171204_pose1"
    SEQ_CLASS = "pose1_seq5"
    EXP_NUM = 1
    FRAME_INDICES = range(11045, 11195)
    INITIAL_PT_CLOUD_FRAME = 11045
    EASY_CAMS = [0, 11, 13, 17, 21, 24, 25, 28]
    MEDIUM_CAMS = [2, 5, 6, 8, 12, 20, 26, 27]
    HARD_CAMS = [1, 3, 4, 7, 8, 14, 15, 30]

class Pose1Config_seq6(BaseConfig):
    SEQ_NAME = "171204_pose1"
    SEQ_CLASS = "pose1_seq6"
    EXP_NUM = 1
    FRAME_INDICES = range(19380, 19530)
    INITIAL_PT_CLOUD_FRAME = 19380
    EASY_CAMS = [0, 11, 13, 17, 21, 24, 25, 28]
    MEDIUM_CAMS = [2, 5, 6, 8, 12, 20, 26, 27]
    HARD_CAMS = [1, 3, 4, 7, 8, 14, 15, 30]

class Sports1Config_seq1(BaseConfig):
    SEQ_NAME = "161029_sports1"
    SEQ_CLASS = "sports1_seq1"
    EXP_NUM = 1
    FRAME_INDICES = range(380, 530)
    INITIAL_PT_CLOUD_FRAME = 380
    EASY_CAMS = [0, 8, 11, 13, 17, 21, 24, 25]
    MEDIUM_CAMS = [10, 12, 14, 15, 16, 19, 20, 27]
    HARD_CAMS = [1, 2, 4, 5, 9, 18, 22, 23]

class Sports1Config_seq2(BaseConfig):
    SEQ_NAME = "161029_sports1"
    SEQ_CLASS = "sports1_seq2"
    EXP_NUM = 1
    FRAME_INDICES = range(535, 685)
    INITIAL_PT_CLOUD_FRAME = 535
    MIN_VISIBILITY_RATIO = 0.90

class Sports1Config_seq3(BaseConfig):
    SEQ_NAME = "161029_sports1"
    SEQ_CLASS = "sports1_seq3"
    EXP_NUM = 1
    FRAME_INDICES = range(830, 980)
    INITIAL_PT_CLOUD_FRAME = 830

class Sports1Config_seq4(BaseConfig):
    SEQ_NAME = "161029_sports1"
    SEQ_CLASS = "sports1_seq4"
    EXP_NUM = 1
    FRAME_INDICES = range(1300, 1450)
    INITIAL_PT_CLOUD_FRAME = 1300

class Sports1Config_seq7(BaseConfig):
    SEQ_NAME = "161029_sports1"
    SEQ_CLASS = "sports1_seq7"
    EXP_NUM = 1
    FRAME_INDICES = range(2186, 2336)
    INITIAL_PT_CLOUD_FRAME = 2186

class Sports1Config_seq8(BaseConfig):
    SEQ_NAME = "161029_sports1"
    SEQ_CLASS = "sports1_seq8"
    EXP_NUM = 1
    FRAME_INDICES = range(2420, 2570)
    INITIAL_PT_CLOUD_FRAME = 2420

EXPERIMENT_LIST = [
    # Pose1Config_seq5, Pose1Config_seq6
    # Sports1Config_seq7, Sports1Config_seq8
    Sports1Config_seq1, Sports1Config_seq2, Pose1Config_seq1, Pose1Config_seq2
    # Pose1Config_seq3, Pose1Config_seq4
]