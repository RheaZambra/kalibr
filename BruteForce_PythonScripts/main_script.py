import os
import sys
import argparse
#from smartphone_data import SmartphoneData
from smartphone_data_flushing import SmartphoneData

def main():
    parser = argparse.ArgumentParser(description="Process smartphone data.")
    parser.add_argument('directory_name', type=str, help='Directory name containing the data')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--imu_smoothing', type=int, default=1, help='IMU smoothing factor')
    parser.add_argument('--media', type=int, default=1, help='Video (0) or Photos (1)')
    parser.add_argument('--downsample_factor', type=int, default=1, help='Downsample factor')
    parser.add_argument('--cam_imu_delT', type=float, default=0.0, help='Camera-IMU time delta')
    parser.add_argument('--smartphone_type', type=int, default=1, help='Not Standardised (0) or Standardised (1)')
    parser.add_argument('--action', nargs='+', type=str, default=['all'], help='List of actions to perform: all, split_images, rename_files, IMU_processing, IMU_smoothing, IMU_downsampling, monoint2rosbag')

    args = parser.parse_args()

    directory_path = os.path.expanduser(args.directory_name)
    smartphone_data = SmartphoneData(directory_path, args.fps, args.imu_smoothing, args.media, args.cam_imu_delT, args.smartphone_type, args.downsample_factor)

    actions = args.action

    if 'split_images' in actions or 'all' in actions:
        smartphone_data.split_images()
    if 'rename_files' in actions or 'all' in actions:
        smartphone_data.rename_files()
        if 'all' in actions:
            input("Please check /rgb for photo quality and press enter to resume actions.")
    if 'IMU_processing' in actions or 'all' in actions:
        smartphone_data.IMU_processing()
    if 'IMU_smoothing' in actions or 'all' in actions:
        smartphone_data.IMU_smoothing()
    # if 'IMU_downsampling' in actions or 'all' in actions:
    #     smartphone_data.IMU_downsampling()
    if 'monoint2rosbag' in actions or 'all' in actions:
        smartphone_data.monoint2rosbag()

if __name__ == "__main__":
    main()
