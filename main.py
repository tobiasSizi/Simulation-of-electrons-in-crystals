import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.signal
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D


#IMPORTANT
#the following units are used:
#Angström, fs, eV, Ampere, kg, etc...
a = 6.24150913 # my personal constant
h = 0.6582119569 #hbar in units of eV*fs
e = 1/1000/a #charge of electron in units of fs*eV
c_0 = 2997.92458 #speed of light in units of Ang/fs


def gaussian(t, width):
    # a basic gaussian function
    return np.exp(-t**2/(2*width**2))

def createTransient(E_0, omega, alpha, pol, width, phase=0, N=1, omega_chirp=0):
    """Returns a tuple of three functions with parameter time: (E_x(t), E_y(t), B_z(t));
    E_0: Amplitude of the Transient;
    omega: angular frequency
    pol: Polarisation (0: E perpendicular to surface, pi/2 parallel to surface)
    width: Width of the Transient
    phase: CEP
    N: refractive index
    omega_chirp: chirp freq."""
    @np.vectorize
    def inner_Ex(t):
        if pol == np.pi/2: # fixing numeric zero issues
            return 0
        return E_0*gaussian(t, width)*np.cos(omega*t + omega_chirp*t**2 + phase)*np.sin(alpha)*np.cos(pol)
    def inner_Ey(t):
        return -E_0*gaussian(t, width)*np.cos(omega*t + omega_chirp*t**2 + phase)*np.sin(pol)
    def inner_Bz(t):
        return -E_0*N/c_0*gaussian(t, width)*np.cos(omega*t + omega_chirp*t**2 + phase)*np.cos(alpha)*np.sin(pol)
    return inner_Ex, inner_Ey, inner_Bz

def createBandTightBinding(amplitude, lattice_const):
    """Returns function e(kx, ky):
    amplitude: Amplitude of the band
    lattice_const: REAL SPACE lattice const."""
    def inner(kx, ky):
        return - amplitude*(np.cos(lattice_const*kx) + np.cos(lattice_const*ky) - 2)
    return inner

def createVelocityTightBinding(amplitude, lattice_const):
    """Returns functions vx(kx, ky), vy(kx, ky):
    amplitude: Amplitude of the band
    lattice_const: REAL SPACE lattice const."""
    factor = amplitude*lattice_const/h
    def vx(kx, ky):
        return factor*np.sin(lattice_const*kx)
    def vy(kx, ky):
        return factor*np.sin(lattice_const*ky)
    return vx, vy

def createBandGraphene(amplitude, lattice_const):
    def inner(kx,ky):
        return amplitude*np.sqrt(  1 + 4* np.cos(np.sqrt(3)*kx*lattice_const/2)* np.cos(ky*lattice_const/2) + 4* np.cos(ky*lattice_const/2)**2 )
    return inner    

def createVelocityGraphene(amplitude, lattice_const):
    e = createBandGraphene(amplitude, lattice_const)
    def inner_vx(kx, ky):
        return - lattice_const*amplitude**2/h/e(kx,ky)*( 2* np.cos(lattice_const*ky/2) + np.cos(np.sqrt(3)*lattice_const*kx/2) )*np.sin(lattice_const*ky/2)
    def inner_vy(kx, ky):
        return - lattice_const*amplitude**2/h/e(kx,ky)*np.sqrt(3)*np.cos(lattice_const*ky/2)*np.sin(np.sqrt(3)*lattice_const*kx/2)
    return inner_vx, inner_vy
    
def createBandDiracCone(amplitude):
    def inner(kx, ky):
        return np.sqrt(kx**2 + ky**2)
    return inner

def createVelocityDiracCone(amplitude):
    def inner_vx(kx, ky):
        return kx/np.sqrt(kx**2 + ky**2)/h
    def inner_vy(kx, ky):
        return ky/np.sqrt(kx**2 + ky**2)/h
    return inner_vx, inner_vy

def fourierTransform(tArray, realValues):
    """Returns: fftFreqs, fftVals"""
    y = realValues
    n = len(tArray)
    freqs = np.fft.fftfreq(n)
    mask = freqs > 0
    fft_theo = 2*np.abs(np.fft.fft(y)/n)
    return freqs[mask], fft_theo[mask] 

def cutoff_log_fit(freq, alpha1, alpha2, beta, cutoff):
    """log of a basic cutoff function to fit onto a spectrum"""
    if freq <= cutoff:
        return alpha1*freq + beta
    else:
        return alpha2*freq + cutoff*(alpha1-alpha2) + beta

cutoff_log_fit = np.vectorize(cutoff_log_fit, otypes=[float]) # vectorize it

def cutoff_fit(freq, alpha1, alpha2, beta, cutoff):
    """basic cutoff function to fit onto a spectrum"""
    if freq <= cutoff:
        return np.exp(alpha1*freq + beta)
    else:
        return np.exp(alpha2*freq + cutoff*(alpha1-alpha2) + beta)

cutoff_fit = np.vectorize(cutoff_fit, otypes=[float]) # vectorize this one as well


class Transient:
    def __init__(self, E_0 = 25*a, omega = 0.05*np.pi, alpha = 86/180*np.pi, pol = np.pi/2, width = 100/2.35482, phase = 0, refrac = 10, omega_chirp=0):
        self.E_0 = E_0
        self.omega = omega
        self.alpha = alpha
        self.pol = pol
        self.width = width
        self.phase = phase
        self.refrac = refrac
        self.omega_chirp = omega_chirp
        self.E_x, self.E_y, self.B_z = createTransient(self.E_0, self.omega, self.alpha, self.pol, self.width, self.phase, self.refrac, self.omega_chirp)

    def changeParam(self, **kwargs):
        for elem in kwargs:
            self.__dict__[elem] = kwargs[elem]
        self.E_x, self.E_y, self.B_z = createTransient(self.E_0, self.omega, self.alpha, self.pol, self.width, self.phase, self.refrac)

    def plotTrans(self, t_init=-4*10**2, t_final=4*10**2, t_delta=.01):
        t_eval = np.arange(t_init, t_final, t_delta)
        fig_1 = plt.figure(1, figsize = (10, 10))
        ex_chart = fig_1.add_subplot(311)
        ex_chart.set_title("E_x(t)")
        ey_chart = fig_1.add_subplot(312)
        ey_chart.set_title("E_y(t)")
        bz_chart = fig_1.add_subplot(313)
        bz_chart.set_title("B_z(t)")
        ex_chart.plot(t_eval, self.E_x(t_eval))
        ex_chart.set_ylim(self.E_x(t_eval).min()-0.01, self.E_x(t_eval).max()+0.01)
        ey_chart.plot(t_eval, self.E_y(t_eval))
        ey_chart.set_ylim(self.E_y(t_eval).min()-0.01, self.E_y(t_eval).max()+0.01)
        bz_chart.plot(t_eval, self.B_z(t_eval))
        bz_chart.set_ylim(self.B_z(t_eval).min()-0.01, self.B_z(t_eval).max()+0.01)
        plt.show()

    def plotFFT(self, t_init=-4*10**2, t_final=4*10**2, t_delta=.01, scale="log"):
        t_eval = np.arange(t_init, t_final, t_delta)
        fig_1 = plt.figure(1, figsize = (10, 10))
        ex_chart = fig_1.add_subplot(311)
        ex_chart.set_yscale(scale)
        ex_chart.set_title("FFT of E_x(t)")
        ey_chart = fig_1.add_subplot(312)
        ey_chart.set_yscale(scale)
        ey_chart.set_title("FFT of E_y(t)")
        bz_chart = fig_1.add_subplot(313)
        bz_chart.set_yscale(scale)
        bz_chart.set_title("FFT of B_z(t)")
        ex_chart.plot(*fourierTransform(t_eval, self.E_x(t_eval)))
        ey_chart.plot(*fourierTransform(t_eval, self.E_y(t_eval)))
        bz_chart.plot(*fourierTransform(t_eval, self.B_z(t_eval)))
        plt.show()
    

class BandTightBinding:
    def __init__(self, lattice_const, amplitude):
        self.lattice_const = lattice_const
        self.amplitude = amplitude
        self.vx, self.vy = createVelocityTightBinding(self.amplitude, self.lattice_const)
        self.band = createBandTightBinding(amplitude, lattice_const)
    
    def plotVel(self, size, numStep):
        X, Y = np.linspace(-size, size, numStep), np.linspace(-size, size, numStep)
        X, Y = np.meshgrid(X, Y)
        U, V = self.vx(X, Y), self.vy(X, Y)
        plt.quiver(X,Y,U,V)
        plt.show()


class BandGraphene:
    def __init__(self, amplitude, lattice_const):
        self.amplitude = amplitude
        self.lattice_const = lattice_const
        self.band = createBandGraphene(amplitude, lattice_const)
        self.vx, self.vy = createVelocityGraphene(amplitude, lattice_const)

    def plotBand(self, xrange=(-1,1), yrange=(-1,1), stepsize=0.01, contourmap=True, wireframe=False):
        x = np.arange(*xrange, stepsize)
        y = np.arange(*yrange, stepsize)
        xx, yy = np.meshgrid(x, y, sparse=True)
        if contourmap:
            z = self.band(xx, yy)
            plt.contourf(x,y,z)
            plt.show()
        if wireframe:
            z2 = self.band(xx, yy)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', title = "BandStruc")
            ax.plot_wireframe(xx, yy, z2)
            plt.show()


class BandDiracCone:
    def __init__(self, amplitude):
        self.amplitude = amplitude
        self.band = createBandDiracCone(amplitude)
        self.vx, self.vy = createVelocityDiracCone(amplitude)

    def plotBand(self, xrange=(-1,1), yrange=(-1,1), stepsize=0.01, contourmap=True, wireframe=False):
        x = np.arange(*xrange, stepsize)
        y = np.arange(*yrange, stepsize)
        xx, yy = np.meshgrid(x, y, sparse=True)
        if contourmap:
            z = self.band(xx, yy)
            plt.contourf(x,y,z)
            plt.show()
        if wireframe:
            z2 = self.band(xx, yy)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', title = "BandStruc")
            ax.plot_wireframe(xx, yy, z2)
            plt.show()


class Solution:
    def __init__(self, bandstructure, transient, t_init=-4*10**2, t_final=4*10**2, t_delta=.01, kx_0 = 0, ky_0 = 0): #the bandstructure and the transient are objects
        self.bandstructure = bandstructure
        self.transient = transient
        self.t_init = t_init
        self.t_final = t_final
        self.t_delta = t_delta
        self.y0 = [0, 0, kx_0, ky_0]

    def solve(self):
        Vlc_x = self.bandstructure.vx
        Vlc_y = self.bandstructure.vy
        E_x, E_y, B_z = self.transient.E_x, self.transient.E_y, self.transient.B_z
        def f(t, y): #y is array: y = [r_x, r_y, k_x, k_y].
            dr_x = Vlc_x(y[2], y[3])
            dr_y = Vlc_y(y[2], y[3])
            dk_x = (-e * E_x(t) - e * Vlc_y(y[2], y[3]) * B_z(t)) / h
            dk_y = (-e * E_y(t) + e * Vlc_x(y[2], y[3]) * B_z(t)) / h
            return np.array([dr_x, dr_y, dk_x, dk_y])
        
        t_span = (self.t_init, self.t_final)
        t_eval = np.arange(self.t_init, self.t_final, self.t_delta)

        self.raw_sol = scipy.integrate.solve_ivp(fun = f, t_span = t_span, y0 = self.y0, t_eval = t_eval)
        self.raw_sol_mirror = scipy.integrate.solve_ivp(fun = f, t_span = t_span, y0 = [self.y0[0], self.y0[1], -self.y0[2], -self.y0[3]], t_eval = t_eval)
        return self.raw_sol, self.raw_sol_mirror

    def plotReziTraj(self):
        plt.plot(self.raw_sol.y[2], self.raw_sol.y[3])
        plt.plot(self.raw_sol_mirror.y[2], self.raw_sol_mirror.y[3])
        plt.show()

    def plotRealTraj(self):
        plt.plot(self.raw_sol.y[2], self.raw_sol.y[3])
        plt.plot(self.raw_sol_mirror.y[2], self.raw_sol_mirror.y[3])
        plt.show()

    def process(self):
        self.vx = self.bandstructure.vx(self.raw_sol.y[2], self.raw_sol.y[3])
        self.vy = self.bandstructure.vy(self.raw_sol.y[2], self.raw_sol.y[3])
        self.vx_mirror = self.bandstructure.vx(self.raw_sol_mirror.y[2], self.raw_sol_mirror.y[3])
        self.vy_mirror  = self.bandstructure.vy(self.raw_sol_mirror.y[2], self.raw_sol_mirror.y[3])

        self.ax = np.gradient(self.vx)/self.t_delta
        self.ay = np.gradient(self.vy)/self.t_delta
        self.ax_mirror = np.gradient(self.vx_mirror)/self.t_delta
        self.ay_mirror = np.gradient(self.vy_mirror)/self.t_delta

        self.fftx = fourierTransform(self.raw_sol.t, self.ax + self.ax_mirror)
        self.ffty = fourierTransform(self.raw_sol.t, self.ay + self.ay_mirror)
    
    def plotVel(self): #WARNING: DOES NOT INCLUDE MIRROR PARTICLE
        plt.plot(self.raw_sol.t, self.vx)
        plt.plot(self.raw_sol.t, self.vy)
        plt.show()
    
    def plotAcc(self): #WARNING: DOES NOT INCLUDE MIRROR PARTICLE
        plt.plot(self.raw_sol.t, self.ax)
        plt.plot(self.raw_sol.t, self.ay)
        plt.show()
    
    def plotFFt(self, scale="log"):
        figg = plt.figure(1, figsize=(22,7))
        chrt_x = figg.add_subplot(211)
        chrt_y = figg.add_subplot(212)
        chrt_x.plot(*self.fftx)
        chrt_y.plot(*self.ffty)
        chrt_x.set_title("fftx")
        chrt_y.set_title("ffty")
        chrt_x.set_yscale(scale)
        chrt_y.set_yscale(scale)
        plt.show()

    def processRadPower(self, delta): #NOT FINISHED
        gauss = gaussian(np.arange(-4*delta, 4*delta, self.t_delta),delta)
        self.radX = np.convolve(self.ax + self.ax_mirror,gauss, mode="same")**2
        self.radY = np.convolve(self.ay + self.ay_mirror,gauss, mode="same")**2

    def processFFTMaxima(self, num_delta=20, shorten=0, plotX=False, plotY=False):
        self.ind_max_X = scipy.signal.find_peaks(self.fftx[1][:-shorten], distance=num_delta)[0]
        self.log_fftX_max = np.log(self.fftx[1][self.ind_max_X])
        if plotX:
            plt.scatter(self.fftx[0][self.ind_max_X], self.log_fftX_max) # scatter plot to estimate start parameters
            plt.show()

        self.ind_max_Y = scipy.signal.find_peaks(self.ffty[1][:-shorten], distance=num_delta)[0]
        self.log_fftY_max = np.log(self.ffty[1][self.ind_max_Y])
        if plotY:
            plt.scatter(self.ffty[0][self.ind_max_Y], self.log_fftY_max) # scatter plot to estimate start parameters
            plt.show()

    def processCutoffFit(self, p0_X=None, p0_Y = None, plotX=False, plotY=False):
        self.poptX, _ = scipy.optimize.curve_fit(
            f = cutoff_log_fit, 
            xdata = self.fftx[0][self.ind_max_X], 
            ydata = self.log_fftX_max, 
            p0 = p0_X)
        if plotX:
            plt.scatter(self.fftx[0][self.ind_max_X], self.log_fftX_max) # scatter plot to estimate start parameters
            plt.plot(self.fftx[0][self.ind_max_X], cutoff_log_fit(self.fftx[0][self.ind_max_X], *self.poptX), color="red")
            plt.show()

        self.poptY, _ = scipy.optimize.curve_fit(
            f = cutoff_log_fit, 
            xdata = self.ffty[0][self.ind_max_Y], 
            ydata = self.log_fftY_max,
            p0 = p0_Y)
        if plotY:
            plt.scatter(self.ffty[0][self.ind_max_Y], self.log_fftY_max) # scatter plot to estimate start parameters
            plt.plot(self.ffty[0][self.ind_max_Y], cutoff_log_fit(self.ffty[0][self.ind_max_Y], *self.poptY), color="red")
            plt.show()

        return self.poptX, self.poptY


