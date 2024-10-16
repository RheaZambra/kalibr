import numpy as np
import pylab as pl


def plotIMURates(cself, iidx, fno=1, clearFigure=True, noShow=False):   
    #timestamps we have me
    imu = cself.ImuList[iidx]
    bodyspline = cself.poseDv.spline()   
    timestamps = np.array([im.stamp.toSec() + imu.timeOffset for im in imu.imuData \
                      if im.stamp.toSec() + imu.timeOffset > bodyspline.t_min() \
                      and im.stamp.toSec() + imu.timeOffset < bodyspline.t_max() ])
    

    scale = 1000.0
    unit = "ms"
    z_thresh = 1.2

    #calculate the relative rate between readings
    times = []
    rates = []
    for idx in range(1,len(timestamps)):
        times.append(timestamps[idx] - timestamps[0])
        rates.append(scale * (timestamps[idx] - timestamps[idx-1]))
    rate_avg = np.average(rates)
    rate_std = np.std(rates)
    
    #loop through and so z-test to find outliers
    #https://en.wikipedia.org/wiki/Z-test
    sizes = []
    colors = []
    for idx in range(0,len(rates)):
        rate = rates[idx]
        #if (abs(rate - rate_avg)/rate_std) > z_thresh:
        #    sizes.append(5)
        #    colors.append("r")
        #else:
        sizes.append(1)
        colors.append("b")
        # rates[idx] = abs(rate - rate_avg)
        rates[idx] = rate

    #plot it    
    f = pl.figure(fno)
    if clearFigure:
        f.clf()
    f.suptitle("imu{0}: sample inertial rate".format(iidx))
    pl.scatter(times, rates, s=sizes, c=colors, marker="x")
    pl.text(0.1, 0.9, 'avg dt ('+unit+') = {:.2f} +- {:.4f}'.format(rate_avg, rate_std), fontsize=12, transform=f.gca().transAxes)
    pl.grid('on')
    pl.xlabel("time (s)")
    pl.ylabel("sample rate ("+unit+")")
    f.gca().set_xlim((min(times), max(times)))
    f.gca().set_ylim((0.0, max(rates)))

def plotGyroError(cself, iidx, fno=1, clearFigure=True, noShow=False):
    errors = np.array([np.dot(re.error(), re.error()) for re in  cself.ImuList[iidx].gyroErrors])
   
    f = pl.figure(fno)
    if clearFigure:
        f.clf()
    f.suptitle("imu{0}: angular velocities error".format(iidx))
    
    pl.subplot(2, 1, 1)
    pl.plot(errors)
    pl.xlabel('error index')
    pl.ylabel('error (rad/sec) squared')
    pl.grid('on')
    
    #only plot till 5*sigma  (output formatting...)
    sigma=np.std(errors)
    errors = errors[ errors < 5*sigma ]
    
    pl.subplot(2, 1, 2)
    pl.hist(errors, len(errors)/100)
    pl.xlabel('error ($rad/s$) squared')
    pl.ylabel('error index')
    pl.grid('on')

def plotGyroErrorPerAxis(cself, iidx, fno=1, clearFigure=True, noShow=False):
    errors = np.array([re.error() for re in  cself.ImuList[iidx].gyroErrors])
   
    f = pl.figure(fno)
    if clearFigure:
        f.clf()
    f.suptitle("imu{0}: angular velocities error".format(iidx))
    
    for i in range(3):
        pl.subplot(3, 1, i+1)
        pl.plot(errors[:,i])
        pl.xlabel('error index')
        pl.ylabel('error ($rad/s$)')
        pl.grid('on')
        sigma = cself.ImuList[iidx].getImuConfig().getGyroStatistics()[0]
        pl.plot(np.array([0., errors.shape[0]]), sigma * 3.* np.ones(2), 'r--')
        pl.plot(np.array([0., errors.shape[0]]), -sigma * 3.* np.ones(2), 'r--')
        pl.xlim([0., errors.shape[0]])

def plotAccelError(cself, iidx, fno=1, clearFigure=True, noShow=False):
    errors = np.array([np.dot(re.error(), re.error()) for re in cself.ImuList[iidx].accelErrors])
   
    f = pl.figure(fno)
    if clearFigure:
        f.clf()
    f.suptitle("imu{0}: acceleration error".format(iidx))
        
    pl.subplot(2, 1, 1)
    pl.plot(errors)
    pl.xlabel('error index')
    pl.ylabel('(m/sec*sec) squared')
    pl.grid('on')
    
    #only plot till 5*sigma  (output formatting...)
    sigma=np.std(errors)
    errors = errors[ errors < 5*sigma ]
    
    pl.subplot(2, 1, 2)
    pl.hist(errors, len(errors)/100)
    pl.xlabel('($m/s^2$) squared')
    pl.ylabel('Error Number')
    pl.grid('on')

def plotAccelErrorPerAxis(cself, iidx, fno=1, clearFigure=True, noShow=False):
    errors = np.array([re.error() for re in  cself.ImuList[iidx].accelErrors])
   
    f = pl.figure(fno)
    if clearFigure:
        f.clf()
    f.suptitle("imu{0}: acceleration error".format(iidx))
    
    for i in range(3):
        pl.subplot(3, 1, i+1)
        pl.plot(errors[:,i])
        pl.xlabel('error index')
        pl.ylabel('error ($m/s^2$)')
        pl.grid('on')
        sigma = cself.ImuList[iidx].getImuConfig().getAccelerometerStatistics()[0]
        pl.plot(np.array([0, errors.shape[0]]), sigma * 3.* np.ones(2), 'r--')
        pl.plot(np.array([0, errors.shape[0]]), -sigma * 3.* np.ones(2), 'r--')
        pl.xlim([0., errors.shape[0]])

def plotAccelBias(cself, imu_idx, fno=1, clearFigure=True, noShow=False):
    imu = cself.ImuList[imu_idx]
    bias = imu.accelBiasDv.spline()
    times = np.array([im.stamp.toSec() for im in imu.imuData if im.stamp.toSec() > bias.t_min() \
                      and im.stamp.toSec() < bias.t_max() ])
    acc_bias_spline = np.array([bias.evalD(t,0) for t in times]).T
    times = times - times[0]     #remove time offset

    plotVectorOverTime(times, acc_bias_spline, 
                       title="imu{0}: estimated accelerometer bias (imu frame)".format(imu_idx), 
                       ylabel="bias ($m/s^2$)", 
                       fno=fno, clearFigure=clearFigure, noShow=noShow)

    sigma_rw = cself.ImuList[imu_idx].getImuConfig().getAccelerometerStatistics()[1]
    bounds = 3. * sigma_rw * np.sqrt(times)
    for i in range(3):
        pl.subplot(3, 1, i+1)
        pl.plot(times, acc_bias_spline[i,0] + bounds, 'r--')
        pl.plot(times, acc_bias_spline[i,0] - bounds, 'r--')

def plotAngularVelocityBias(cself, imu_idx, fno=1, clearFigure=True, noShow=False):
    imu = cself.ImuList[imu_idx]
    bias = imu.gyroBiasDv.spline()
    times = np.array([im.stamp.toSec() for im in imu.imuData if im.stamp.toSec() > bias.t_min() \
                      and im.stamp.toSec() < bias.t_max() ])
    gyro_bias_spline = np.array([bias.evalD(t,0) for t in times]).T
    times = times - times[0]     #remove time offset
    
    plotVectorOverTime(times, gyro_bias_spline, 
                       title="imu{0}: estimated gyro bias (imu frame)".format(imu_idx), 
                       ylabel="bias ($rad/s$)", 
                       fno=fno, clearFigure=clearFigure, noShow=noShow)

    sigma_rw = cself.ImuList[imu_idx].getImuConfig().getGyroStatistics()[1]
    bounds = 3. * sigma_rw * np.sqrt(times)
    for i in range(3):
        pl.subplot(3, 1, i+1)
        pl.plot(times, gyro_bias_spline[i,0] + bounds, 'r--')
        pl.plot(times, gyro_bias_spline[i,0] - bounds, 'r--')

#plots angular velocity of the body fixed spline versus all imu measurements
def plotAngularVelocities(cself, iidx, fno=1, clearFigure=True, noShow=False):
    #predicted (over the time of the imu)
    imu = cself.ImuList[iidx]
    bodyspline = cself.poseDv.spline()   
    times = np.array([im.stamp.toSec() + imu.timeOffset for im in imu.imuData \
                      if im.stamp.toSec() + imu.timeOffset > bodyspline.t_min() \
                      and im.stamp.toSec() + imu.timeOffset < bodyspline.t_max() ])
    predictedAng_body =  np.array([err.getPredictedMeasurement() for err in imu.gyroErrors]).T
    
    #transform the measurements to the body frame
    #not neccessray for imu0 as it is aligned with the spline
    measuredAng_body =  np.array([err.getMeasurement() for err in imu.gyroErrors]).T

    #remove time offset
    times = times - times[0]
    
    #plot the predicted measurements
    plotVectorOverTime(times, predictedAng_body, 
                       title="Comparison of predicted and measured angular velocities (body frame)", 
                       ylabel="ang. velocity ($rad/s$)", 
                       label="est. bodyspline",
                       fno=fno, clearFigure=clearFigure, noShow=noShow, lw=3)
    
    #plot measurements
    for r in range(0,3):
        ax=pl.subplot(3, 1, r+1)
        pl.plot(times, measuredAng_body[r,:], 'x', lw=1, label="imu{0}".format(iidx))
        pl.legend()

def plotAccelerations(cself, iidx, fno=1, clearFigure=True, noShow=False):   
    #predicted 
    imu = cself.ImuList[iidx]
    bodyspline = cself.poseDv.spline()   
    times = np.array([im.stamp.toSec() + imu.timeOffset for im in imu.imuData \
                      if im.stamp.toSec() + imu.timeOffset > bodyspline.t_min() \
                      and im.stamp.toSec() + imu.timeOffset < bodyspline.t_max() ])
    predicetedAccel_body =  np.array([err.getPredictedMeasurement() for err in imu.accelErrors]).T
    
    #transform accelerations from imu to body frame (on fixed body and geometry was estimated...)
    #works for imu0 as it is aligned with the spline
    #TODO(schneith): implement the fixed-body acceleration transformation 
    measuredAccel_imu =  np.array([err.getMeasurement() for err in imu.accelErrors]).T
    measuredAccel_body = measuredAccel_imu
    
    #remove time offset
    times = times - times[0] 
    
    #plot the predicted measurements
    plotVectorOverTime(times, predicetedAccel_body, 
                       title="Comparison of predicted and measured specific force (imu0 frame)", 
                       ylabel="specific force ($m/s^2$)", 
                       label="est. bodyspline",
                       fno=fno, clearFigure=clearFigure, noShow=noShow, lw=3)
    
    #plot the measurements
    for r in range(0,3):
        ax=pl.subplot(3, 1, r+1)
        pl.plot(times, measuredAccel_body[r,:], 'x', lw=1, label="imu{0}".format(iidx))
        pl.legend()

def plotVectorOverTime(times, values, title="", ylabel="", label="", fno=1, clearFigure=True, noShow=False, lw=3):
    f = pl.figure(fno)
    if clearFigure:
        f.clf()
    f.suptitle(title)
    for r in range(0,3):
        pl.subplot(3, 1, r+1)
        pl.plot(times, values[r,:], 'b-', lw=lw, label=label)
        pl.grid('on')
        pl.xlabel("time (s)")
        pl.ylabel(ylabel)
        if label != "":
            pl.legend()

def plotReprojectionScatter(cself, cam_id, ax=None, title="", noShow=False):
    cam = cself.CameraChain.camList[cam_id]
    
    # Create a new figure and axis if not provided
    if ax is None:
        f, ax = pl.subplots()
        f.suptitle(title)
    else:
        ax.clear()
        ax.set_title(title)

    numImages = len(cam.allReprojectionErrors)
    values = np.arange(numImages) / np.double(numImages)
    cmap = pl.cm.jet(values, alpha=0.5)

    # Reprojection errors scatter plot
    for image_id, rerrs_image in enumerate(cam.allReprojectionErrors):
        color = cmap[image_id, :]
        rerrs = np.array([rerr.error() for rerr in rerrs_image])  
        ax.plot(rerrs[:, 0], rerrs[:, 1], 'x', lw=3, mew=3, color=color)

    # Add uncertainty bound
    uncertainty_bound = pl.Circle((0, 0), 3. * cam.cornerUncertainty, color='k', linestyle='dashed',
                                  fill=False, lw=2, zorder=len(cam.allReprojectionErrors))
    ax.add_artist(uncertainty_bound)

    ax.axis('equal')
    ax.grid('on')
    ax.set_xlabel('error x ($pix$)')
    ax.set_ylabel('error y ($pix$)')
    SM = pl.cm.ScalarMappable(pl.cm.colors.Normalize(0.0, numImages), pl.cm.jet)
    SM.set_array(np.arange(numImages))
    cb = pl.colorbar(SM, ax=ax)
    cb.set_label('image index')

    if not noShow:
        pl.show(block=False)
        pl.pause(2)
        pl.close('all')


class CameraPlot:
    def __init__(self, fig,  targetPoints, camSize):
        self.initialized = False
        #get the data
        self.targetPoints = targetPoints
        self.camSize = camSize
        self.fig = fig
        #setup the figure
        self.setupFigure()
        self.plot3Dgrid()
        #initialize camerea
        T = np.eye(4,4)
        self.plot3DCamera(T)
        
    def setupFigure(self):
        #interactive mode
        pl.ion()
        #hack to enforce axis equal (matplotlib doesn't support that)
        #self.ax.set_aspect('equal')
        MAX = 1
        for direction in (-1, 1):
            for point in np.diag(direction * MAX * np.array([1,1,1])):
                self.ax.plot([point[0]], [point[1]], [point[2]], 'w')
        self.fig.show()
        
    def plot3Dgrid(self):
        #draw target corners        
        for i in range(0, len(self.targetPoints) ):
            self.ax.scatter(self.targetPoints[i,0], self.targetPoints[i,1], self.targetPoints[i,2],color="g",s=1)
        
        self.ax.plot([0,self.targetPoints[-1,0]],[0,0],[0,0], color="r")
        self.ax.plot([0,0],[0,self.targetPoints[-1,1]],[0,0], color="g")
        self.ax.plot([0,0],[0,0],[0,self.targetPoints[-1,0]], color="b")
 
    def plot3DCamera(self, T):
        #transform affine
        ori = T * np.matrix([[0],[0],[0],[1]])
        v1 =  T * np.matrix([[self.camSize],[0],[0],[1]])
        v2 =  T * np.matrix([[0],[self.camSize],[0],[1]])
        v3 =  T * np.matrix([[0],[0],[self.camSize],[1]])
        
        #initialize objects
        if not self.initialized:
            self.cam_x = self.ax.plot(np.squeeze([ori[0], v1[0]]), np.squeeze([ori[1], v1[1]]), np.squeeze([ori[2], v1[2]]), color="r")
            self.cam_y = self.ax.plot(np.squeeze([ori[0], v2[0]]), np.squeeze([ori[1], v2[1]]), np.squeeze([ori[2], v2[2]]), color="g")
            self.cam_z = self.ax.plot(np.squeeze([ori[0], v3[0]]), np.squeeze([ori[1], v3[1]]), np.squeeze([ori[2], v3[2]]), color="b")
            self.initialized = True
            
        else:
            xy=np.squeeze([ori[0:2], v1[0:2]]).transpose()
            z=np.squeeze([ori[2], v1[2]]).transpose()
            self.cam_x[0].set_data(xy)
            self.cam_x[0].set_3d_properties(z)
            xy=np.squeeze([ori[0:2], v2[0:2]]).transpose()
            z=np.squeeze([ori[2], v2[2]]).transpose()
            self.cam_y[0].set_data(xy)
            self.cam_y[0].set_3d_properties(z)
            xy=np.squeeze([ori[0:2], v3[0:2]]).transpose()
            z=np.squeeze([ori[2], v3[2]]).transpose()
            self.cam_z[0].set_data(xy)
            self.cam_z[0].set_3d_properties(z)
            pl.pause(0.00001)
