import argparse
import os
import subprocess

def write_bash_script(parameter, param_min, param_max, interval, reprojection_sigma, cam_chain, bag_name, output_folder):
    # Generate the bash script content
    bash_script_content = f"#!/bin/bash\n\n"
    bagname_trunc = os.path.splitext(bag_name)[0]
    
    # Loop over the parameter range and generate commands
    current_value = param_min
    run_number = 1
    while current_value <= param_max:
        # Command to modify imu.yaml with the current parameter value
        modify_command = f"sed -i 's/{parameter}: .*/{parameter}: {current_value}/' imu.yaml\n"
        # Kalibr calibration command
        calibration_command = (
            f"rosrun kalibr kalibr_calibrate_imu_camera "
            f"--imu-models calibrated "
            f"--reprojection-sigma {reprojection_sigma} "
            f"--target checkerboard.yaml "
            f"--imu imu.yaml "
            f"--cams {cam_chain} "
            f"--bag {bag_name}\n"
        )
        # Add to the script
        bash_script_content += modify_command
        bash_script_content += calibration_command
        bash_script_content += f"mv {bagname_trunc}-report-imucam.pdf {output_folder}/{bagname_trunc}camIMU{run_number}.pdf\n\n"

        # Increment the current value and run number
        current_value += interval
        run_number += 1

    # Write the bash script to a file
    script_path = os.path.join(output_folder, "run_kalibr.sh")
    with open(script_path, 'w') as f:
        f.write(bash_script_content)
    
    # Make the bash script executable
    os.chmod(script_path, 0o755)
    
    # Execute the bash script
    subprocess.run([script_path], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and run a bash script for Kalibr IMU-Camera calibration.")
    parser.add_argument("--parameter", required=True, help="Parameter to adjust in imu.yaml")
    parser.add_argument("--min", type=float, required=True, help="Minimum value for the parameter")
    parser.add_argument("--max", type=float, required=True, help="Maximum value for the parameter")
    parser.add_argument("--interval", type=float, required=True, help="Interval between parameter values")
    parser.add_argument("--reprojection-sigma", type=float, required=True, help="Reprojection sigma value")
    parser.add_argument("--cams", required=True, help="Camera chain yaml file")
    parser.add_argument("--bag", required=True, help="Rosbag file")
    parser.add_argument("--output", required=True, help="Output folder for the calibration reports")
    
    args = parser.parse_args()

    # Ensure the output folder exists
    os.makedirs(args.output, exist_ok=True)
    
    write_bash_script(
        parameter=args.parameter,
        param_min=args.min,
        param_max=args.max,
        interval=args.interval,
        reprojection_sigma=args.reprojection_sigma,
        cam_chain=args.cams,
        bag_name=args.bag,
        output_folder=args.output
    )

