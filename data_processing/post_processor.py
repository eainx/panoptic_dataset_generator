# data_processing/trajectory_filter.py
import os
import pickle

class PostProcessor8:
    def __init__(self, cfg):
        self.cfg = cfg
        self.T = self.cfg.NUM_FRAMES
        self.easy_cams_list = self.cfg.EASY_CAMS
        self.medium_cams_list = self.cfg.MEDIUM_CAMS
        self.hard_cams_list = self.cfg.HARD_CAMS
        self.levels_list = [
            ('easy', self.easy_cams_list),
            ('medium', self.medium_cams_list),
            ('hard', self.hard_cams_list)
        ]

    def _load_results(self):
        output_filepath = os.path.join(self.cfg.TRACK_ANNOTATION_DIR, f'{self.cfg.SEQ_CLASS}')
        output_filename = os.path.join(output_filepath, 'annotation.pkl')
        if not os.path.exists(output_filename):
            print(f"Error: Source file not found at {output_filename}")
            return
    
        with open(output_filename, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def _save_split_data(self, level, data_dict):
        output_filepath = os.path.join(self.cfg.TRACK_ANNOTATION_DIR, f'{self.cfg.SEQ_CLASS}', level)
        os.makedirs(output_filepath, exist_ok=True)
        output_filename = os.path.join(output_filepath, 'annotation.pkl')
        print(f"\nSaving split data to {output_filename}...")
        with open(output_filename, 'wb') as f:
            pickle.dump(data_dict, f)
        print("Save complete.")

    def run(self):
        data = self._load_results()
        cams = list(range(31))
        
        if len(cams) != data['video'].shape[0]:
            print(f"Warning: Number of cameras in config ({len(cams)}) does not match data dimension ({data['video'].shape[0]})")


        for level, cam_list_for_group in self.levels_list:
            print(f"\n--- Splitting data for '{level.upper()}' group ---")

            slice_indices = [i for i, cam_idx in enumerate(cams) if cam_idx in cam_list_for_group]
            print(cam_list_for_group)
            print(slice_indices)

            if not slice_indices:
                print(f"No cameras from '{level}' group found in the ordered list. Skipping.")
                continue
            
            output_data = {
                "video": data['video'][slice_indices],
                "coords": data['coords'][slice_indices],
                "visibility": data['visibility'][slice_indices],
                "tracks_XYZ": data['tracks_XYZ'],
                "w2c": data['w2c'][slice_indices],
                "fx_fy_cx_cy": data['fx_fy_cx_cy'][slice_indices],
            }
            
            self._save_split_data(level, output_data)
        
        print("\n--- All processes end. ---")
