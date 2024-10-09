import os
import sys
import csv
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pandas import DataFrame as df
import PIL.Image
import rospy
import rosbag
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class SmartphoneData:
    def __init__(self, directory_name, fps, imu_smoothing, media, cam_imu_delT, smartphone_type, downsample_factor):
        self.directory_name = directory_name
        self.fps = fps
        self.imu_smoothing = imu_smoothing
        self.media = media
        self.cam_imu_delT = cam_imu_delT*1E9
        self.smartphone_type = smartphone_type
        self.downsample_factor = downsample_factor
        self.start_time = self.get_start_time()
        self.accelerometer_data = os.path.join(directory_name, "Accelerometer.csv")
        self.gravity_data = os.path.join(directory_name, "Gravity.csv")
        self.gyroscope_data = os.path.join(directory_name, "Gyroscope.csv")

    def get_start_time(self):
        camera_directory = os.path.join(self.directory_name, "Camera")
        if self.media == 1:
            filenames = os.listdir(camera_directory)
            image_filenames = [f for f in filenames if f.endswith('.jpg')]
            timestamps = [float(os.path.splitext(filename)[0]) for filename in image_filenames]
            start_time = min(timestamps)
        else:
            video_files = [f for f in os.listdir(camera_directory) if f.endswith('.mp4')]
            if not video_files:
                raise FileNotFoundError("No video files found in the Camera directory.")
            video_file = video_files[0]
            start_time = os.path.splitext(video_file)[0]
            try:
                return int(start_time)
            except ValueError:
                raise ValueError(f"Cannot extract start time from video file name: {video_file}")

    def split_images(self):
        rgb_directory = os.path.join(self.directory_name, "rgb")
        if os.path.exists(rgb_directory):
            shutil.rmtree(rgb_directory)
        os.makedirs(rgb_directory)
        if self.media == 1:
            camera_directory = os.path.join(self.directory_name, "Camera")
            image_filenames = [f for f in os.listdir(camera_directory) if f.endswith('.jpg')]
            for filename in image_filenames:
                img_path = os.path.join(camera_directory, filename)
                new_img_path = os.path.join(rgb_directory, filename)
                shutil.copy(img_path, new_img_path)
        else:
            video_file = os.path.join(self.directory_name, "Camera", f"{self.start_time}.mp4")
            # ffmpeg_command = f"ffmpeg -i {video_file} -vf 'crop=1080:1080:0:420,fps={self.fps}' {os.path.join(self.directory_name, 'rgb')}/%d.jpg"
            ffmpeg_command = f"ffmpeg -i {video_file} -vf 'fps={self.fps}' {os.path.join(self.directory_name, 'rgb')}/%d.jpg"
            os.system(ffmpeg_command)

    def rename_files(self):
        if self.media == 1:
            return

        else:
            rgb_directory = os.path.join(self.directory_name, "rgb")
            sorted_filenames = sorted(os.listdir(rgb_directory), key=self.filename_key)
            
            for index, filename in enumerate(sorted_filenames):
                interval = (1.0 / self.fps) * 10**3
                pose_timestamp = int(self.start_time + index * interval)
                new_filename = os.path.join(rgb_directory, "{}.jpg".format(pose_timestamp))
                os.rename(os.path.join(rgb_directory, filename), new_filename)

    @staticmethod
    def filename_key(filename):
        name, extension = os.path.splitext(filename)
        try:
            return float(name)
        except ValueError:
            return float('inf')

    def IMU_processing(self):
        try:
            accelerometer_df = pd.read_csv(self.accelerometer_data)
            gyroscope_df = pd.read_csv(self.gyroscope_data)
            gravity_df = pd.read_csv(self.gravity_data)

            accel_grav_merged_df = pd.merge(accelerometer_df, gravity_df, on='time', suffixes=('_accel', '_gravity'))

            processed_accelerometer_df = accel_grav_merged_df.copy()
            processed_accelerometer_df['x_accel'] += accel_grav_merged_df['x_gravity']
            processed_accelerometer_df['y_accel'] += accel_grav_merged_df['y_gravity']
            processed_accelerometer_df['z_accel'] += accel_grav_merged_df['z_gravity']
            processed_accelerometer_df.drop(columns=['x_gravity', 'y_gravity', 'z_gravity'])

            processed_IMU_df = pd.merge(processed_accelerometer_df, gyroscope_df, on='time')
            processed_IMU_df[['x_accel', 'x']] *= -1
            processed_IMU_df[['time']] -= self.cam_imu_delT

            # If smartphone_type is 0, multiply processed IMU values by -1
            if self.smartphone_type == 0:
                processed_IMU_df[['x_accel', 'y_accel', 'z_accel', 'x', 'y', 'z']] *= -1

            processed_IMU_df[['time']] /= 1000000
            column_order = ['time', 'y', 'x', 'z', 'y_accel', 'x_accel', 'z_accel']
            processed_IMU_df = processed_IMU_df.reindex(columns=column_order)

            processed_IMU_file = os.path.join(self.directory_name, "imu.csv")
            column_rename_map = {
                'time': 'timestamp',
                'x': 'wy [rad/s]',
                'y': 'wx [rad/s]',
                'z': 'wz [rad/s]',
                'x_accel': 'ay [m/s2]',
                'y_accel': 'ax [m/s2]',
                'z_accel': 'az [m/s2]'
            }
            processed_IMU_df.rename(columns=column_rename_map, inplace=True)

            # Save the DataFrame to a CSV file
            processed_IMU_df.to_csv(processed_IMU_file, index=False)
            self.processed_IMU_data = processed_IMU_df

            print("IMU data processed and saved to:", processed_IMU_file)
        except Exception as e:
            print("Error: cannot process IMU data", e)

    @staticmethod
    def moving_average(data, sample_size):
        # Calculate the bounds for handling edges
        lower_bound = int(-1 * math.ceil((sample_size - 1) / 2))
        upper_bound = int(math.floor((sample_size - 1) / 2))
        
        # Compute the moving average using np.convolve
        avg_list = np.convolve(data, np.ones(sample_size) / sample_size, mode='valid')
        
        # If sample_size is not 1, handle the edges
        if sample_size != 1:
            # Handle the beginning edge using a window size of 5
            beginning = [np.mean(data[max(0, i - 24):i + 25]) for i in range(upper_bound)]
            
            # Handle the ending edge using a window size of 5
            ending = [np.mean(data[i - 24:min(len(data), i + 25)]) for i in range(len(data) + lower_bound, len(data))]
            
            # Concatenate the beginning, average list, and ending to get the full average list
            avg_list = np.concatenate((beginning, avg_list, ending))
        
        return avg_list

    @staticmethod
    def plot_imu_data(smoothed_data, raw_data=None):
        # Calculate time elapsed for the x-axis
        smoothed_timestamps = smoothed_data['timestamp']
        time_elapsed_smooth = smoothed_timestamps - smoothed_timestamps[0]
        
        if raw_data is not None:
            raw_timestamps = raw_data['timestamp']
            time_elapsed_raw = raw_timestamps - raw_timestamps[0]
            ax_raw = raw_data['ax [m/s2]']
            ay_raw = raw_data['ay [m/s2]']
            az_raw = raw_data['az [m/s2]']
            wx_raw = raw_data['wx [rad/s]']
            wy_raw = raw_data['wy [rad/s]']
            wz_raw = raw_data['wz [rad/s]']

        ax_smooth = smoothed_data['ax [m/s2]']
        ay_smooth = smoothed_data['ay [m/s2]']
        az_smooth = smoothed_data['az [m/s2]']
        wx_smooth = smoothed_data['wx [rad/s]']
        wy_smooth = smoothed_data['wy [rad/s]']
        wz_smooth = smoothed_data['wz [rad/s]']

        # Plot specific force (accelerometer readings)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        if raw_data is not None:
            plt.plot(time_elapsed_raw, ax_raw, label='Raw ax', alpha=0.5)
        plt.plot(time_elapsed_smooth, ax_smooth, label='Smoothed ax', alpha=0.8)
        plt.legend()
        plt.title('Accelerometer Readings over Time')
        
        plt.subplot(3, 1, 2)
        if raw_data is not None:
            plt.plot(time_elapsed_raw, ay_raw, label='Raw ay', alpha=0.5)
        plt.plot(time_elapsed_smooth, ay_smooth, label='Smoothed ay', alpha=0.8)
        plt.ylabel('Acceleration (m/s^2)')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        if raw_data is not None:
            plt.plot(time_elapsed_raw, az_raw, label='Raw az', alpha=0.5)
        plt.plot(time_elapsed_smooth, az_smooth, label='Smoothed az', alpha=0.8)
        plt.legend()
        plt.xlabel('Time Elapsed (ms)')
        plt.tight_layout()
        
        # Plot angular velocity (gyroscope readings)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        if raw_data is not None:
            plt.plot(time_elapsed_raw, wx_raw, label='Raw wx', alpha=0.5)
        plt.plot(time_elapsed_smooth, wx_smooth, label='Smoothed wx', alpha=0.8)
        plt.legend()
        plt.title('Gyroscope Readings over Time')
        
        plt.subplot(3, 1, 2)
        if raw_data is not None:
            plt.plot(time_elapsed_raw, wy_raw, label='Raw wy', alpha=0.5)
        plt.plot(time_elapsed_smooth, wy_smooth, label='Smoothed wy', alpha=0.8)
        plt.ylabel('Angular Velocity (rad/s)')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        if raw_data is not None:
            plt.plot(time_elapsed_raw, wz_raw, label='Raw wz', alpha=0.5)
        plt.plot(time_elapsed_smooth, wz_smooth, label='Smoothed wz', alpha=0.8)
        plt.legend()
        plt.xlabel('Time Elapsed (ms)')
        plt.tight_layout()
        
        plt.show()

    def IMU_smoothing(self):
        accelerometer = pd.read_csv(self.accelerometer_data)
        gyroscope = pd.read_csv(self.gyroscope_data)
        accel_gyro_merged = pd.merge(accelerometer, gyroscope, on='time', suffixes=('_a', '_w'))

        smoothed_imu_df = pd.DataFrame({
            'timestamp': accel_gyro_merged['time'],
            'ax': self.moving_average(accel_gyro_merged['x_a'], self.imu_smoothing),
            'ay': self.moving_average(accel_gyro_merged['y_a'], self.imu_smoothing),
            'az': self.moving_average(accel_gyro_merged['z_a'], self.imu_smoothing),
            'wx': self.moving_average(accel_gyro_merged['x_w'], self.imu_smoothing),
            'wy': self.moving_average(accel_gyro_merged['y_w'], self.imu_smoothing),
            'wz': self.moving_average(accel_gyro_merged['z_w'], self.imu_smoothing)
        })

        # smoothed_imu_df['wx'] = 0 #####
        # # smoothed_imu_df['wy'] = 0.0005 #####
        # smoothed_imu_df['wz'] = 0 #####
        # # smoothed_imu_df['ax'] = -1 #####
        # smoothed_imu_df['ay'] = 0 #####
        # smoothed_imu_df['az'] = 0 #####

        gravity_df = pd.read_csv(self.gravity_data)
        smoothed_imu_df['ax'] += gravity_df['x']
        smoothed_imu_df['ay'] += gravity_df['y']
        smoothed_imu_df['az'] += gravity_df['z']
        # smoothed_imu_df['ay'] += 9.81
        # #average_ay = smoothed_imu_df['ax'].mean()
        # #smoothed_imu_df['ax'] = average_ay

        smoothed_imu_df['ax'] *= -1
        smoothed_imu_df['wx'] *= -1
        smoothed_imu_df['timestamp'] -= self.cam_imu_delT

        # If smartphone_type is 0, multiply processed IMU values by -1
        if self.smartphone_type == 0:
            smoothed_imu_df['ax'] *= -1 
            smoothed_imu_df['ay'] *= -1 
            smoothed_imu_df['az'] *= -1 
            smoothed_imu_df['wx'] *= -1 
            smoothed_imu_df['wy'] *= -1 
            smoothed_imu_df['wz'] *= -1

        smoothed_imu_df['timestamp'] /= 1000000
        column_order = ['timestamp', 'wy', 'wx', 'wz', 'ay', 'ax', 'az']
        smoothed_imu_df = smoothed_imu_df.reindex(columns=column_order)

        smoothed_IMU_file = os.path.join(self.directory_name, "imu_smoothed.csv")
        column_rename_map = {
            'timestamp': 'timestamp',
            'wx': 'wy [rad/s]',
            'wy': 'wx [rad/s]',
            'wz': 'wz [rad/s]',
            'ax': 'ay [m/s2]',
            'ay': 'ax [m/s2]',
            'az': 'az [m/s2]'
        }
        smoothed_imu_df.rename(columns=column_rename_map, inplace=True)

        smoothed_imu_df.to_csv(smoothed_IMU_file, index=False)
        self.smoothed_IMU_data = smoothed_imu_df

        self.plot_imu_data(smoothed_imu_df, self.processed_IMU_data)
        self.plot_imu_data(smoothed_imu_df)

        # Save processed IMU data to a new CSV file
        smoothed_IMU_file = os.path.join(self.directory_name, "imu_smoothed.csv")
        smoothed_imu_df.rename(columns={
            'timestamp': 'timestamp',
            'wx': 'wy [rad/s]',
            'wy': 'wx [rad/s]',
            'wz': 'wz [rad/s]',
            'ax': 'ay [m/s2]',
            'ay': 'ax [m/s2]',
            'az': 'az [m/s2]'
        }).to_csv(smoothed_IMU_file, index=False)

        print("IMU data smoothed and saved to:", smoothed_IMU_file)
        self.smoothed_IMU_data = smoothed_imu_df

    # def IMU_downsampling(self):
    #     downsampled_IMU_file = os.path.join(self.directory_name, "imu_downsampled.csv")
    #     downsampled_rows = []
        
    #     for i in range(0, len(self.smoothed_IMU_data), self.downsample_factor):
    #         downsampled_row = {
    #             'timestamp': self.smoothed_IMU_data['timestamp'].iloc[i],
    #             'wx': self.smoothed_IMU_data['wy'].iloc[i],
    #             'wy': self.smoothed_IMU_data['wx'].iloc[i],
    #             'wz': self.smoothed_IMU_data['wz'].iloc[i],
    #             'ax': self.smoothed_IMU_data['ay'].iloc[i],
    #             'ay': self.smoothed_IMU_data['ax'].iloc[i],
    #             'az': self.smoothed_IMU_data['az'].iloc[i]
    #         }
    #         downsampled_rows.append(downsampled_row)

    #     downsampled_data = pd.DataFrame(downsampled_rows)
    #     downsampled_data.to_csv(downsampled_IMU_file, index=False)
    #     print("IMU data downsampled and saved to:", downsampled_IMU_file)
    #     self.downsampled_IMU_data = downsampled_data

    def find_ROS_start_time(self):
        imgs_directory = os.path.join(self.directory_name, "rgb")
        img_files = [f for f in os.listdir(imgs_directory) if f.endswith('.jpg')]
        
        min_value = float('inf')
        max_value = float('-inf')
        min_file = None
        max_file = None

        for img_file in img_files:
            try:
                float_value = float(os.path.splitext(img_file)[0])
                if float_value < min_value:
                    min_value = float_value
                    min_file = img_file
                if float_value > max_value:
                    max_value = float_value
                    max_file = img_file
            except ValueError:
                continue

        if min_file:
            print(f"The image file with the earliest timestamp is: {min_value}")
        else:
            print("No valid image files found.")

        if max_file:
            print(f"The image file with the latest timestamp is: {max_value}")
        else:
            print("No valid image files found.")

        min_imu = self.processed_IMU_data['timestamp'].iloc[0]
        max_imu = self.processed_IMU_data['timestamp'].iloc[-1]

        print(f"The earliest timestamped IMU signal is: {min_imu}")
        print(f"The latest timestamped IMU signal is: {max_imu}")

        if min_imu > min_value:
            min_value = min_imu

        print(f"The ROS start time is: {min_value}")
        print(f"The ROS end time is: {max_value}")

        self.ROS_start_time = min_value
        self.ROS_end_time = max_value

    def monoint2rosbag(self):
        self.find_ROS_start_time()
        image_dir = os.path.join(self.directory_name, 'rgb')
        image_files = sorted(os.listdir(image_dir), key=lambda x: float(os.path.splitext(x)[0]))
        imu_file = os.path.join(self.directory_name, 'imu.csv')
        smoothed_imu_file = os.path.join(self.directory_name, 'imu_smoothed.csv')

        imu_msgs = []
        seq = 0
        for i in range(0, len(self.processed_IMU_data), 1):
            if float(self.processed_IMU_data['timestamp'].iloc[i]) < self.ROS_start_time or float(self.processed_IMU_data['timestamp'].iloc[i]) > self.ROS_end_time:
                continue

            imu_msg = Imu()
            imu_msg.header.stamp = rospy.Time.from_sec(float(self.processed_IMU_data['timestamp'].iloc[i]) / 1e3)
            imu_msg.header.seq = seq
            seq += 1
            imu_msg.linear_acceleration.x = self.processed_IMU_data['ax [m/s2]'].iloc[i]
            imu_msg.linear_acceleration.y = self.processed_IMU_data['ay [m/s2]'].iloc[i]
            imu_msg.linear_acceleration.z = self.processed_IMU_data['az [m/s2]'].iloc[i]
            imu_msg.angular_velocity.x = self.processed_IMU_data['wx [rad/s]'].iloc[i]
            imu_msg.angular_velocity.y = self.processed_IMU_data['wy [rad/s]'].iloc[i]
            imu_msg.angular_velocity.z = self.processed_IMU_data['wz [rad/s]'].iloc[i]

        # Set default orientation and high covariance to indicate no reliable orientation data
            imu_msg.orientation.x = 0.0
            imu_msg.orientation.y = 0.0
            imu_msg.orientation.z = 0.0
            imu_msg.orientation.w = 1.0
            imu_msg.orientation_covariance[0] = 99999.9
            imu_msg.orientation_covariance[4] = 99999.9
            imu_msg.orientation_covariance[8] = 99999.9

            imu_msgs.append(imu_msg)

        smoothed_imu_msgs = []
        seq = 0
        for i in range(0, len(self.smoothed_IMU_data), 1):
            if float(self.smoothed_IMU_data['timestamp'].iloc[i]) < self.ROS_start_time:
                continue
            imu_msg = Imu()
            imu_msg.header.stamp = rospy.Time.from_sec(float(self.smoothed_IMU_data['timestamp'].iloc[i]) / 1e3)
            imu_msg.header.seq = seq
            seq += 1
            imu_msg.linear_acceleration.x = self.smoothed_IMU_data['ax [m/s2]'].iloc[i]
            imu_msg.linear_acceleration.y = self.smoothed_IMU_data['ay [m/s2]'].iloc[i]
            imu_msg.linear_acceleration.z = self.smoothed_IMU_data['az [m/s2]'].iloc[i]
            imu_msg.angular_velocity.x = self.smoothed_IMU_data['wx [rad/s]'].iloc[i]
            imu_msg.angular_velocity.y = self.smoothed_IMU_data['wy [rad/s]'].iloc[i]
            imu_msg.angular_velocity.z = self.smoothed_IMU_data['wz [rad/s]'].iloc[i]

        # Set default orientation and high covariance to indicate no reliable orientation data
            imu_msg.orientation.x = 0.0
            imu_msg.orientation.y = 0.0
            imu_msg.orientation.z = 0.0
            imu_msg.orientation.w = 1.0
            imu_msg.orientation_covariance[0] = 99999.9
            imu_msg.orientation_covariance[4] = 99999.9
            imu_msg.orientation_covariance[8] = 99999.9

            smoothed_imu_msgs.append(imu_msg)

        image_msgs = []
        bridge = CvBridge()
        batch_size = 500  # Process images in batches of 500

        IMU_bag_name = self.directory_name + '.bag'
        bag = rosbag.Bag(IMU_bag_name, 'w')
        print("Creating IMU ROSbag ...")

        # Process and write IMU data
        for imu_msg in imu_msgs:
            bag.write('/imu', imu_msg, imu_msg.header.stamp)

        # Process images in batches
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
            image_msg.header.stamp = rospy.Time.from_sec(float(image_file.split('.')[0]) / 1e3)
            image_msgs.append(image_msg)

            # Once a batch is collected, write it to the bag
            if (idx + 1) % batch_size == 0:
                for img_msg in image_msgs:
                    bag.write('/camera/image_raw', img_msg, img_msg.header.stamp)
                print(f"Processed and wrote batch of {batch_size} images. Flushing to disk.")
                bag.flush()  # Flush to ensure data is written to disk
                image_msgs.clear()  # Clear list after writing

        # Process remaining images
        if image_msgs:
            for img_msg in image_msgs:
                bag.write('/camera/image_raw', img_msg, img_msg.header.stamp)
            print(f"Processed and wrote remaining {len(image_msgs)} images. Flushing to disk.")
            bag.flush()
            image_msgs.clear()

        bag.close()
        print("ROS bag file created:", IMU_bag_name)

        # Create smoothed IMU ROSbag
        smoothed_IMU_bag_name = self.directory_name + '_smoothed' + str(self.imu_smoothing) + '.bag'
        bag = rosbag.Bag(smoothed_IMU_bag_name, 'w')
        print("Creating smoothed IMU ROSbag ...")

        smoothed_imu_msgs = []
        # Process and write smoothed IMU data
        for imu_msg in smoothed_imu_msgs:
            bag.write('/imu', imu_msg, imu_msg.header.stamp)

        # Process and write image data again for smoothed IMU ROSbag
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
            image_msg.header.stamp = rospy.Time.from_sec(float(image_file.split('.')[0]) / 1e3)
            image_msgs.append(image_msg)

            if (idx + 1) % batch_size == 0:
                for img_msg in image_msgs:
                    bag.write('/camera/image_raw', img_msg, img_msg.header.stamp)
                print(f"Processed and wrote batch of {batch_size} images for smoothed bag. Flushing to disk.")
                bag.flush()
                image_msgs.clear()

        # Process remaining images for smoothed IMU ROSbag
        if image_msgs:
            for img_msg in image_msgs:
                bag.write('/camera/image_raw', img_msg, img_msg.header.stamp)
            print(f"Processed and wrote remaining {len(image_msgs)} images for smoothed bag. Flushing to disk.")
            bag.flush()

        bag.close()
        print("ROS bag file created:", smoothed_IMU_bag_name)