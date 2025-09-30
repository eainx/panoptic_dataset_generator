# main.py
import argparse
import config as cfg
from config import EXPERIMENT_LIST
from data_processing.pointcloud_generator import PointCloudGenerator
from data_processing.image_processor import ImageProcessor
from data_processing.metadata_generator import MetadataGenerator
from data_processing.trajectory_filter_ver2 import TrajectoryFilter_ver2
from data_processing.trajectory_filter import TrajectoryFilter
from data_processing.post_processor import PostProcessor8
from utils.visualize import Visualizer

def run_experiment(config_class, step):
    config_instance = config_class()

    print("="*80)
    print(f"  Starting Processing for Sequence: {config_instance.SEQ_NAME} {config_instance.SEQ_CLASS}")
    print("="*80)

    if step in ['image', 'all']:
        ImageProcessor(config_instance).run()

    if step in ['pointcloud', 'all']:
        PointCloudGenerator(config_instance).run()

    if step in ['meta', 'all']:
        MetadataGenerator(config_instance).run()

    if step in ['filter']:
        TrajectoryFilter(config_instance).run()

    if step in ['filterv2']:
        TrajectoryFilter_ver2(config_instance).run()
    
    if step in ['visualize']:
        Visualizer(config_instance).run()
    
    if step in ['postprocess']:
        PostProcessor8(config_instance).run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Launcher")
    parser.add_argument('--exp_id', type=int, help=f"ID of the experiment to run (0 to {len(EXPERIMENT_LIST)-1})")
    parser.add_argument('--step', type=str, default='all', help="Run a specific step ('pointcloud', 'image', 'meta', 'filter', 'all')")
    parser.add_argument('--all', action='store_true', help="Run all experiments")

    args = parser.parse_args()

    if args.all:
        print(f"Running all {len(EXPERIMENT_LIST)} experiments...")
        for config_cls in EXPERIMENT_LIST:
            run_experiment(config_cls, args.step)
            
    elif args.exp_id is not None:
        if 0 <= args.exp_id < len(EXPERIMENT_LIST):
            selected_config_class = EXPERIMENT_LIST[args.exp_id]
            run_experiment(selected_config_class, args.step)
        else:
            print(f"Error: Experiment ID {args.exp_id} is out of range.")
    else:
        print("Please specify an experiment with --exp_id <ID> or run all with --all.")
        print("Available experiments:")
        for i, cfg_cls in enumerate(EXPERIMENT_LIST):
            print(f"  ID {i}: {cfg_cls.SEQ_NAME}")