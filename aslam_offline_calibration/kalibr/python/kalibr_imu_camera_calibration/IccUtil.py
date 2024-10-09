from __future__ import print_function #handle print in 2.x python
from sm import PlotCollection
from . import IccPlots as plots
import sm
import numpy as np
import pylab as pl
import sys
import subprocess
import yaml
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mpl_toolkits.mplot3d.axes3d as p3
import io
try:
    # Python 2
    from cStringIO import StringIO
except ImportError:
    # Python 3
    from io import StringIO
import matplotlib.patches as patches

# make numpy print prettier
np.set_printoptions(suppress=True)


def plotTrajectory(cself, fno=1, clearFigure=True, title=""):
    f = pl.figure(fno)
    if clearFigure:
        f.clf()
    f.suptitle(title)

    size = 0.05
    a3d = f.add_subplot(111, projection='3d')

    # get times we will evaulate at (fixed frequency)
    imu = cself.ImuList[0]
    bodyspline = cself.poseDv.spline()
    times_imu = np.array([im.stamp.toSec() + imu.timeOffset for im in imu.imuData \
                      if im.stamp.toSec() + imu.timeOffset > bodyspline.t_min() \
                      and im.stamp.toSec() + imu.timeOffset < bodyspline.t_max() ])
    times = np.arange(np.min(times_imu), np.max(times_imu), 1.0/10.0)
    
    #plot each pose
    traj_max = np.array([-9999.0, -9999.0, -9999.0])
    traj_min = np.array([9999.0, 9999.0, 9999.0])
    T_last = None
    for time in times:
        position =  bodyspline.position(time)
        orientation = sm.r2quat(bodyspline.orientation(time))
        T = sm.Transformation(orientation, position)
        sm.plotCoordinateFrame(a3d, T.T(), size=size)
        # record min max
        traj_max = np.maximum(traj_max, position)
        traj_min = np.minimum(traj_min, position)
        # compute relative change between
        if T_last != None:
            pos1 = T_last.t()
            pos2 = T.t()
            a3d.plot3D([pos1[0], pos2[0]],[pos1[1], pos2[1]],[pos1[2], pos2[2]],'k-', linewidth=1)
        T_last = T

    #TODO: should also plot the target board here (might need to transform into imu0 grav?)

    a3d.auto_scale_xyz([traj_min[0]-size, traj_max[0]+size], [traj_min[1]-size, traj_max[1]+size], [traj_min[2]-size, traj_max[2]+size])


def printErrorStatistics(cself, dest=sys.stdout):
    # Reprojection errors
    print("Normalized Residuals\n----------------------------", file=dest)
    for cidx, cam in enumerate(cself.CameraChain.camList):
        if len(cam.allReprojectionErrors)>0:
            e2 = np.array([ np.sqrt(rerr.evaluateError()) for reprojectionErrors in cam.allReprojectionErrors for rerr in reprojectionErrors])
            print("Reprojection error (cam{0}):     mean {1}, median {2}, std: {3}".format(cidx, np.mean(e2), np.median(e2), np.std(e2) ), file=dest)
        else:
            print("Reprojection error (cam{0}):     no corners".format(cidx), file=dest)
    
    for iidx, imu in enumerate(cself.ImuList):
        # Gyro errors
        e2 = np.array([ np.sqrt(e.evaluateError()) for e in imu.gyroErrors ])
        print("Gyroscope error (imu{0}):        mean {1}, median {2}, std: {3}".format(iidx, np.mean(e2), np.median(e2), np.std(e2)), file=dest)
        # Accelerometer errors
        e2 = np.array([ np.sqrt(e.evaluateError()) for e in imu.accelErrors ])
        print("Accelerometer error (imu{0}):    mean {1}, median {2}, std: {3}".format(iidx, np.mean(e2), np.median(e2), np.std(e2)), file=dest)

    print("", file=dest)
    print("Residuals\n----------------------------", file=dest)
    for cidx, cam in enumerate(cself.CameraChain.camList):
        if len(cam.allReprojectionErrors)>0:
            e2 = np.array([ np.linalg.norm(rerr.error()) for reprojectionErrors in cam.allReprojectionErrors for rerr in reprojectionErrors])
            print("Reprojection error (cam{0}) [px]:     mean {1}, median {2}, std: {3}".format(cidx, np.mean(e2), np.median(e2), np.std(e2) ), file=dest)
        else:
            print("Reprojection error (cam{0}) [px]:     no corners".format(cidx), file=dest)
    
    for iidx, imu in enumerate(cself.ImuList):
        # Gyro errors
        e2 = np.array([ np.linalg.norm(e.error()) for e in imu.gyroErrors ])
        print("Gyroscope error (imu{0}) [rad/s]:     mean {1}, median {2}, std: {3}".format(iidx, np.mean(e2), np.median(e2), np.std(e2)), file=dest)
        # Accelerometer errors
        e2 = np.array([ np.linalg.norm(e.error()) for e in imu.accelErrors ])
        print("Accelerometer error (imu{0}) [m/s^2]: mean {1}, median {2}, std: {3}".format(iidx, np.mean(e2), np.median(e2), np.std(e2)), file=dest)

def printGravity(cself):
    print("")
    print("Gravity vector: (in target coordinates): [m/s^2]")
    print(cself.gravityDv.toEuclidean())

def printResults(cself, withCov=False):
    nCams = len(cself.CameraChain.camList)
    for camNr in range(0,nCams):
        T_cam_b = cself.CameraChain.getResultTrafoImuToCam(camNr)

        print("")
        print("Transformation T_cam{0}_imu0 (imu0 to cam{0}, T_ci): ".format(camNr))
        if withCov and camNr==0:
            print("    quaternion: ", T_cam_b.q(), " +- ", cself.std_trafo_ic[0:3])
            print("    translation: ", T_cam_b.t(), " +- ", cself.std_trafo_ic[3:])
        print(T_cam_b.T())
        
        if not cself.noTimeCalibration:
            print("")
            print("cam{0} to imu0 time: [s] (t_imu = t_cam + shift)".format(camNr))
            print(cself.CameraChain.getResultTimeShift(camNr), end=' ')
            
            if withCov:
                print(" +- ", cself.std_times[camNr])
            else:
                print("")

    print("")
    for (imuNr, imu) in enumerate(cself.ImuList):
        print("IMU{0}:\n".format(imuNr), "----------------------------")
        imu.getImuConfig().printDetails()
            
def printBaselines(self):
    #print all baselines in the camera chain
    if nCams > 1:
        for camNr in range(0,nCams-1):
            T, baseline = cself.CameraChain.getResultBaseline(camNr, camNr+1)
            
            if cself.CameraChain.camList[camNr+1].T_extrinsic_fixed:
                isFixed = "(fixed to external data)"
            else:
                isFixed = ""
            
            print("")
            print("Baseline (cam{0} to cam{1}): [m] {2}".format(camNr, camNr+1, isFixed))
            print(T.T())
            print(baseline, "[m]")
    

def plotVector(ax, T, color='r', label='Translation Vector'):
    origin = (0,0,0)
    direction = T[:, 3]  # Last column as vector
    ax.quiver(origin[0], origin[1], origin[2], 
              direction[0], direction[1], direction[2], 
              color=color, label=label)
    ax.set_xlim([-0.128, 0.032]) # ADJUST ACCORDING TO PHONE DIMENSIONS
    ax.set_ylim([-0.010, 0.010]) # ADJUST ACCORDING TO PHONE DIMENSIONS
    ax.set_zlim([-0.004, 0.004]) # ADJUST ACCORDING TO PHONE DIMENSIONS
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

def generateReport(cself, filename="report.pdf", showOnScreen=True, reprojection_sigma=None, camera_config=None):
    figs = list()
    plotter = PlotCollection.PlotCollection("Calibration report")
    
    # Get results as text
    result_lines = getResultText(cself)
    
    for cidx, cam in enumerate(cself.CameraChain.camList):
        # Create a figure with three subplots: one for text (left), one for reprojection error plot (upper right), and one for the 3D vector plot (bottom right)
        fig = plt.figure(figsize=(17, 22))
        gs = fig.add_gridspec(2, 2, width_ratios=[0.3, 0.7], height_ratios=[0.3, 0.7]) 

        ax_text = fig.add_subplot(gs[:, 0])
        ax_plot = fig.add_subplot(gs[1, 1])
        ax_vector = fig.add_subplot(gs[0, 1], projection='3d')
        fig.suptitle(f"Camera {cidx} Calibration Report", fontsize=12)
        ax_text.axis('off')
        
        text_content = "\n".join(result_lines[:4])  # Transformation parameters
        text_content += "\n\n"
        text_content += "\n".join(result_lines[4:])  # Camera and IMU parameters
        
        ax_text.text(0, 1, text_content, fontsize=10, va='top', ha='left', wrap=True, transform=ax_text.transAxes)
        title = f"Cam {cidx}: Reprojection Errors"
        plots.plotReprojectionScatter(cself, cidx, ax=ax_plot, title=title)
        
        T = cself.CameraChain.getResultTrafoImuToCam(cidx).T()  # Get the transformation matrix
        plotVector(ax_vector, T)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the margins to fit the plots
        
        figs.append(fig)
    
    # Save all figures to a single PDF
    pdf = PdfPages(filename)
    for fig in figs:
        pdf.savefig(fig)
    pdf.close()
    
    # if showOnScreen:
    #     plotter.show()


def exportPoses(cself, filename="poses_imu0.csv"):
    
    # Append our header, and select times at IMU rate
    f = open(filename, 'w')
    print("#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []", file=f)
    imu = cself.ImuList[0]
    bodyspline = cself.poseDv.spline()
    times = np.array([im.stamp.toSec() + imu.timeOffset for im in imu.imuData \
                      if im.stamp.toSec() + imu.timeOffset > bodyspline.t_min() \
                      and im.stamp.toSec() + imu.timeOffset < bodyspline.t_max() ])

    # Times are in nanoseconds -> convert to seconds
    # Use the ETH groundtruth csv format [t,q,p,v,bg,ba]
    for time in times:
        position =  bodyspline.position(time)
        orientation = sm.r2quat(bodyspline.orientation(time))
        print("{:.0f},".format(1e9 * time) + ",".join(map("{:.6f}".format, position)) \
               + "," + ",".join(map("{:.6f}".format, orientation)) , file=f)

def saveResultTxt(cself, filename='cam_imu_result.txt'):
    f = open(filename, 'w')
    getResultText(cself, stream=f)

def getResultText(cself):
    import io
    stream = io.StringIO()
    
    # Calibration results
    nCams = len(cself.CameraChain.camList)
    result_lines = []
    
    for camNr in range(nCams):
        T = cself.CameraChain.getResultTrafoImuToCam(camNr)
        result_lines.append(f"T_ic:  (cam{camNr} to imu0): \n{T.inverse().T()}\n")
        result_lines.append(f"timeshift cam{camNr} to imu0 [s] (t_imu = t_cam + shift): {cself.CameraChain.getResultTimeShift(camNr)}\n")

    for camNr, cam in enumerate(cself.CameraChain.camList):
        result_lines.append(f"cam{camNr}\n-----\n")
        cam_config = io.StringIO()
        cam.camConfig.printDetails(cam_config)
        result_lines.append(cam_config.getvalue() + "\n")

        target_config = io.StringIO()
        cam.targetConfig.printDetails(target_config)
        result_lines.append(target_config.getvalue() + "\n")

    result_lines.append("IMU configuration: \n")
    
    for imuNr, imu in enumerate(cself.ImuList):
        imu_config = io.StringIO()  # Create a StringIO object to capture the output
        imu.getImuConfig().printDetails(imu_config)  # Print details to the StringIO object
        imu_config_content = imu_config.getvalue()

        # Filter the content to get only the desired sections
        capture = False
        for line in imu_config_content.splitlines():
            if "Accelerometer:" in line or "Gyroscope:" in line:
                capture = True
            if capture:
                result_lines.append(line)
    
    return result_lines