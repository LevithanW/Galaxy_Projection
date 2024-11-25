import os
import webbrowser
from io import BytesIO
from typing import NamedTuple, Tuple

import emcee
import h5py

# import pio
import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np

# import error_propagation as er
# from uncertainties import ufloat
import pandas as pd
import plotly as plty
import plotly.express as px
import requests
import scipy.integrate as it
from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.utils import data
from ipywidgets import FloatSlider, IntSlider, interact
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm, Normalize, SymLogNorm
from matplotlib.ticker import PercentFormatter
from numpy import linalg as LA
from PIL import Image
from scipy.optimize import curve_fit, leastsq, minimize
from scipy.signal import find_peaks
from scipy.special import iv, kv
from scipy.stats import binned_statistic_2d
from sklearn.mixture import GaussianMixture

# import illustris_python as il


# from photutils.isophote import Ellipse, EllipseGeometry, EllipseSample, EllipseFitter ,build_ellipse_model
# from photutils.aperture import EllipticalAperture
# from photutils.morphology import data_properties

# from spectral_cube import SpectralCube


mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
# Parametros:
h = 0.6774

# custom colormap that transitions from black (opaque) to black (transparent)
cmap = mpl.colormaps.get_cmap("Greys_r")
colors = cmap(np.linspace(0, 1, 256))
a = 200
alpha_values = np.concatenate([np.ones(256 - a), np.zeros(a)])
colors[:, 3] = alpha_values  # alpha channel to go from opaque to transparent
custom_cmap = ListedColormap(colors)


# Some relevant functions
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Normalize the value, with the midpoint being shifted to zero
        value = np.asarray(value)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint
        if not np.ma.is_masked(value):
            value = np.ma.masked_invalid(value)

        # Interpolating values around the midpoint
        rescaled_value = np.interp(value, [vmin, midpoint, vmax], [0, 0.5, 1])
        return rescaled_value


# Function to calculate specific angular momentum (L_total) for particles within r_out
def cal_specific_ang_mom(mass, pos, vel, r_out):
    r = LA.norm(pos, axis=1)  # Calculate the 3D distances for all particles
    mask = np.where(r <= r_out)[0]  # Identify particles within radius r_out
    m = mass[mask]  # Select masses for the particles in the region
    p = pos[mask]  # Select positions within the radius
    v = vel[mask]  # Select velocities within the radius
    L = np.cross(p, v)  # Compute angular momentum for each particle
    L_total = np.sum(m * L.T, axis=1)  # Mass-weighted total angular momentum vector
    return L_total  # Return the 3D vector of specific angular momentum


# Function to compute a rotation matrix to align a coordinate system
def vector_rotation_matrix(vec1, vec2):
    v1 = vec1 / LA.norm(vec1)  # Normalize vec1
    v2 = vec2 / LA.norm(vec2)  # Normalize vec2
    dim = v1.size  # Determine vector dimensions
    I = np.identity(dim)  # Identity matrix
    c = np.dot(v1, v2)  # Cosine of the angle between v1 and v2
    K = np.outer(v2, v1) - np.outer(v1, v2)  # Cross-product matrix
    return I + K + (K @ K) / (1 + c)  # Rodrigues' rotation formula


# Function to compute azimuthal angle (phi) based on Cartesian coordinates
def phi_ang(x, y):
    phi = np.arctan(y / x)  # Basic arctan computation
    if np.isscalar(x):  # Handle scalar case
        # Adjust phi for quadrant conditions
        if x < 0:
            phi += np.pi
        elif (x > 0) and (y < 0):
            phi += 2.0 * np.pi
        elif x == 0 and y > 0:
            phi = np.pi / 2.0
        elif x == 0 and y < 0:
            phi = 3 * np.pi / 2.0
    else:
        phi[np.where(x < 0.0)] += np.pi
        phi[np.where((x > 0.0) & (y < 0.0))] += 2.0 * np.pi
        uu = np.where((x == 0.0) & (y > 0.0))
        if uu[0].size != 0:
            phi[uu] = np.pi / 2.0
        uu2 = np.where((x == 0.0) & (y < 0.0))
        if uu2[0].size != 0:
            phi[uu2] = 3 * np.pi / 2.0
    return phi


# Function to calculate the radial distance in the xy-plane
def R(x, y):
    return np.sqrt(x**2 + y**2)


# Function to calculate radial velocity component
def rad_vel(x, y, vx, vy):
    r = R(x, y)  # Radial distance
    return (x * vx + y * vy) / r  # Projection of velocity onto the radial direction


# Function to calculate tangential velocity component
def tang_vel(x, y, vx, vy):
    r = R(x, y)  # Radial distance
    return (x * vy - y * vx) / r  # Tangential velocity component


# Function to calculate line-of-sight velocity (v_los)
def V_los(x, y, vx, vy, vz, phi, i):
    v_los = (
        tang_vel(x, y, vx, vy) * np.cos(phi) + rad_vel(x, y, vx, vy) * np.sin(phi)
    ) * np.sin(i * np.pi / 180) + vz * np.cos(
        phi
    )  # Combine velocity contributions
    return v_los


def galactocentrictosky(rgal, phigal, incli, pa, centre, asc, nx, ny):
    # variables are set up as:
    # ; variable_name [physical unit] = meaning

    # ; incli [°]=inclination
    # ; pa [°]=position angle
    # ; centre = vector [xc,yc]/centre of image
    # ; asc [''/px]= angular scale
    # ; theta [rad] = azimuthal angle
    # ; rg [''] = galactocentric radius

    x0 = centre[0]
    y0 = centre[1]
    pa0 = pa * np.pi / 180.0  # ° -> rad : [°]*pi/180° = rad
    inc0 = incli * np.pi / 180.0  # ° -> rad : [°]*pi/180° = rad
    nphi = len(phigal)
    ix0 = np.zeros(nphi)
    iy0 = np.zeros(nphi)
    for iphi in range(nphi):
        rr = rgal[iphi]
        theta = phigal[iphi]  # be sure that it is in radians
        ix0[iphi] = (  # tilted ring method for projection
            -rr
            / asc
            * (np.cos(theta) * np.sin(pa0) + np.sin(theta) * np.cos(pa0) * np.cos(inc0))
            + x0
        )
        iy0[iphi] = (
            rr
            / asc
            * (np.cos(theta) * np.cos(pa0) - np.sin(theta) * np.sin(pa0) * np.cos(inc0))
            + y0
        )

    return ix0, iy0

def radius1(
    size: Tuple[int, int],
    inc: float,
    pa: float,
    centre: Tuple[float, float],
    asc: float,
) -> Deprojected:
    """
    Generate deprojected coordinates, radius, and azimuthal angle
    in the plane of an artificial disk, given the projected sky-plane
    parameters (inclination, disk center, position angle of the major axis).

    :param size: Tuple[int, int]
        A tuple specifying the size of the 2D sky image (nx, ny).
    :param inc: float
        Inclination angle in degrees.
    :param pa: float
        Position angle of the major axis in degrees.
    :param centre: Tuple[float, float]
        A tuple (x0, y0) specifying the center of the disk on the sky-plane.
    :param asc: float
        Angular scale in arcsec/pixel for the sky-plane pixels.
    :return: Deprojected
        A named tuple containing:
        - rg: Deprojected galactocentric radius (arcsec).
        - xg: Deprojected x-coordinate in the disk plane (arcsec).
        - yg: Deprojected y-coordinate in the disk plane (arcsec).
        - psig: Azimuthal angle in the disk plane (radians).
    """
    # Extract center coordinates
    (x0, y0) = centre

    # (PA) & (INC) from degrees to radians
    pa0 = pa * np.pi / 180.0
    inc0 = inc * np.pi / 180.0

    # Create 2D coordinate grids for the image
    j = np.outer(np.ones(size[0]), np.arange(size[1]))  # y-coordinates
    i = np.outer(np.arange(size[0]), np.ones(size[1]))  # x-coordinates

    # Transform sky-plane coordinates to galaxy-plane coordinates
    xg = -(i - x0) * asc * np.sin(pa0) + (j - y0) * asc * np.cos(pa0)
    yg = -((i - x0) * asc * np.cos(pa0) + (j - y0) * asc * np.sin(pa0)) / np.cos(inc0)

    # Galactocentric radius in arcseconds
    rg = np.sqrt((xg**2) + (yg**2))

    # Calculate azimuthal angle (in radians) in the disk plane
    tanpsi = yg / xg
    psig = np.arctan(tanpsi)

    # Adjust azimuthal angle for different quadrants
    psig[np.where(xg < 0.0)] += np.pi  # Second and third quadrants
    psig[np.where((xg > 0.0) & (yg < 0.0))] += 2.0 * np.pi  # Fourth quadrant

    # Special cases for xg = 0
    uu = np.where((xg == 0.0) & (yg > 0.0))  # Positive yg
    if uu[0].size != 0:
        psig[uu] = np.pi / 2.0
    uu2 = np.where((xg == 0.0) & (yg < 0.0))  # Negative yg
    if uu2[0].size != 0:
        psig[uu2] = 3 * np.pi / 2.0

    # Return deprojected data as a named tuple
    return Deprojected(rg, xg, yg, psig)

# Full rotation and rotation curve method
def galaxy_projection(dim, asc, x, y, z, vx, vy, vz, INC, PA, dphi, gal_exp, dist):
    # Input parameters:
    # dim: Dimensions of the grid
    # asc ['']: Arcseconds per unit scale
    # x, y, z [kpc]: Particle positions
    # vx, vy, vz [km/s]: Velocities of particles
    # INC [°]: Inclination of the galaxy
    # PA [°]: Position angle of the galaxy
    # dphi [°]: Rotation offset
    # gal_exp: Identifier for the galaxy
    # dist: Imposed Distance to the galaxy in Mpc

    # (PA) & (INC) from degrees to radians
    pa = PA * np.pi / 180
    inc = INC * np.pi / 180

    # Initialize the sum_img array, which stores values for each pixel
    sum_img = np.zeros((dim, dim), dtype=object)
    for i in range(dim):
        for j in range(dim):
            sum_img[i, j] = []  # Initialize as empty lists to imitate a mock datacube

    # Compute angular position (phi) with offset and wrap into [0, 2π]
    phi_t = (phi_ang(x, y) + dphi * np.pi / 180) % (2 * np.pi)

    # Calculate the radius in the plane of the Galaxy
    r2d = np.sqrt(x**2 + y**2)
    r2d = (
        3600 * 180 / np.pi * np.arctan(r2d / dist * 1e-3)
    )  # radius [kpc] & dist [Mpc]=[1e3 kpc]

    # Define the central pixel for the grid
    centre_test = [int(dim / 2), int(dim / 2)]

    # Perform the projection using the tilted ring method
    res = galactocentrictosky(r2d, phi_t, INC, PA, centre_test, asc, dim, dim)
    xvec, yvec = res[0], res[1]
    xveci = np.round(xvec).astype(int)  # Round to nearest pixel
    yveci = np.round(yvec).astype(int)

    # Initialize the line-of-sight velocity array
    vlos0 = np.zeros(len(xvec))

    # Calculate line-of-sight velocities
    for idx in range(len(xvec)):
        x_t, y_t = x[idx], y[idx]
        vx_t, vy_t, vz_t = vx[idx], vy[idx], vz[idx]
        pos_x, pos_y = xveci[idx], yveci[idx]

        # Line-of-sight velocity
        vlos0[idx] = (
            tang_vel(x_t, y_t, vx_t, vy_t) * np.cos(phi_t[idx])
            + rad_vel(x_t, y_t, vx_t, vy_t) * np.sin(phi_t[idx])
        ) * np.sin(inc) + vz_t * np.cos(inc)

    # Initialize density and velocity field arrays
    dens0 = np.zeros((dim, dim)) + np.nan
    vf0 = np.zeros((dim, dim)) + np.nan

    # Create density and velocity field maps
    for ii in range(dim):
        # Select points within pixel column ii
        oo2 = np.where((xvec >= ii) & (xvec < ii + 1))
        iykeep0 = np.array([])
        vloskeep0 = np.array([])

        if len(oo2[0]) != 0:
            iykeep0 = yvec[oo2]
            vloskeep0 = vlos0[oo2]

        for jj in range(dim):
            # Select points within pixel row jj
            oo3 = np.where((iykeep0 >= jj) & (iykeep0 < jj + 1))
            if len(oo3[0]) > 0:
                vf0[jj, ii] = np.nanmedian(vloskeep0[oo3])  # Median velocity
                dens0[jj, ii] = len(oo3[0])  # Density
            if len(oo3[0]) == 1:
                vf0[jj, ii] = vloskeep0[oo3[0]]
                dens0[jj, ii] = len(oo3[0])

    # Compute galactic radius and angular position on a grid
    rgaltmp, xgaltmp, ygaltmp, phigaltmp = radius1((1024, 1024), INC, PA, (512, 512), 1)
    phigaltmp = (
        phigaltmp.T
    )  # Consider the transpose due to the geometry of the generated image
    rgaltmp = rgaltmp.T

    # Extract observed velocity field, angle and radius
    vfobs = fits.PrimaryHDU(data=vf0).data
    phiobs = fits.PrimaryHDU(data=phigaltmp).data
    radobs = fits.PrimaryHDU(data=rgaltmp).data

    # Mask to extract the observed parameters
    index_vlos = np.where(
        ~np.isnan(vfobs)
    )  # Note: If image has noise or objects surrounding it, consider a model to filter and select the galaxy's parameters.
    # In this case, there's no noise and it's the isolated galaxy
    val_vlos = vfobs[index_vlos]
    test_rad = radobs[index_vlos]
    test_phi = phiobs[index_vlos]

    # Free memory
    del vfobs, phiobs, radobs, phigaltmp, rgaltmp, xgaltmp, ygaltmp

    # Initialize rotation curve fitting
    inv = 50
    interv = np.linspace(0, 300, inv)  # Radial bins
    nr = len(interv) - 1
    test_fit = [0]  # To initialize the fit
    test_fit_er = [0]

    # Fit a rotation curve for each radial bin
    for a in range(nr):
        rad_min, rad_max = interv[a], interv[a + 1]
        mask_pos = np.where(
            (test_rad > rad_min) & (test_rad < rad_max) & (val_vlos > 0)
        )[0]

        vlostmp = val_vlos[mask_pos]
        phitmp = test_phi[mask_pos]

        if len(vlostmp) > 0:
            # Fit the velocity model
            p0 = 0
            popt, pcov = curve_fit(vfmodel, phitmp, vlostmp, p0)
            test_fit.append(abs(popt[0]))  # Fit
            test_fit_er.append(np.sqrt(pcov[0][0]))  # Error from the fit
        else:
            test_fit.append(np.nan)
            test_fit_er.append(0)

    # Compile results into a DataFrame
    d = {
        "Radius ['']": interv,
        "V_phi [km/s]": test_fit,
        "V_phi error [km/s]": test_fit_er,
    }
    val = pd.DataFrame(d)

    # Save results to CSV to specified directory based on the galaxy and projection parameters
    newpath = f"./ID/{gal_exp}/dist_{int(dist)}/inc_{int(INC)}/PA_{int(PA)}/"  # Change directory to your personal computer / folder of interest
    os.makedirs(newpath, exist_ok=True)  # In the case you don't create the folder
    val.to_csv(f"{newpath}/dphi_{dphi}.csv", encoding="utf-8", index=False)

    # Free memory and output path
    del interv, test_fit, test_fit_er, d, val
    return print(
        f"Data saved in: {newpath}dphi_{dphi}.csv"
    )  # Print the final status of the process


def galaxy_projection_img(dim, asc, x, y, z, vx, vy, vz, INC, PA, dphi, gal_exp, dist):
    # Function to generate a velocity field projection image for a galaxy

    # (PA) & (INC) from degrees to radians
    pa = PA * np.pi / 180
    inc = INC * np.pi / 180

    # Initialize a 2D array to store density and velocity field data
    sum_img = np.zeros((dim, dim), dtype=object)
    for i in range(dim):
        for j in range(dim):
            sum_img[i, j] = []

    # Compute azimuthal angles and 2D projected radii
    phi_t = (phi_ang(x, y) + dphi * np.pi / 180) % (2 * np.pi)
    r2d = np.sqrt(x**2 + y**2)
    r2d = 3600 * 180 / np.pi * np.arctan(r2d / dist * 1e-3)  # Convert to arcseconds
    centre_test = [
        int(dim / 2),
        int(dim / 2),
    ]  # Assume galaxy center in the middle of the grid

    # Project 3D coordinates to 2D sky plane
    res = galactocentrictosky(r2d, phi_t, INC, PA, centre_test, asc, dim, dim)
    xvec, yvec = res[0], res[1]
    xveci = np.round(xvec).astype(int)  # Round projected positions to integers
    yveci = np.round(yvec).astype(int)

    # Initialize LOS (line-of-sight) velocity array
    vlos0 = np.zeros(len(xvec))
    for idx in range(len(xvec)):
        # Extract coordinates and velocities for each particle
        x_t, y_t = x[idx], y[idx]
        vx_t, vy_t, vz_t = vx[idx], vy[idx], vz[idx]
        pos_x, pos_y = xveci[idx], yveci[idx]  # Map positions to pixel indices

        # Calculate LOS velocity using tangential, radial components and inclination
        vlos0[idx] = (
            tang_vel(x_t, y_t, vx_t, vy_t) * np.cos(phi_t[idx])
            + rad_vel(x_t, y_t, vx_t, vy_t) * np.sin(phi_t[idx])
        ) * np.sin(inc) + vz_t * np.cos(inc)

    # Initialize velocity field array
    vf0 = np.zeros((dim, dim)) + np.nan
    # Populate velocity field grid
    for ii in range(dim):
        oo2 = np.where((xvec >= ii) & (xvec < ii + 1))
        iykeep0 = np.array([])
        vloskeep0 = np.array([])

        if len(oo2[0]) != 0:
            iykeep0 = yvec[oo2]
            vloskeep0 = vlos0[oo2]
        for jj in range(dim):
            oo3 = np.where((iykeep0 >= jj) & (iykeep0 < jj + 1))
            if len(oo3[0]) > 0:
                vf0[jj, ii] = np.nanmedian(vloskeep0[oo3])
            if len(oo3[0]) == 1:
                vf0[jj, ii] = vloskeep0[oo3[0]]

    # Compute galactic radius and angular position on a grid
    rgaltmp, xgaltmp, ygaltmp, phigaltmp = radius1((1024, 1024), INC, PA, (512, 512), 1)
    phigaltmp = (
        phigaltmp.T
    )  # Consider the transpose due to the geometry of the generated image
    rgaltmp = rgaltmp.T

    # Extract velocity data for plotting
    vfobs = fits.PrimaryHDU(data=vf0).data
    phiobs = fits.PrimaryHDU(data=phigaltmp).data
    radobs = fits.PrimaryHDU(data=rgaltmp).data

    index_vlos = np.where(~np.isnan(vfobs))  # Mask to extract the observed parameters
    val_vlos = vfobs[index_vlos]  # Extract velocity values

    # Note: If image has noise or objects surrounding it, consider a model to filter and select the galaxy's parameters.
    # In this case, there's no noise and it's the isolated galaxy

    # Determine velocity range for consistent colormap scaling
    max_val = np.max(np.abs(val_vlos))
    vmin, vmax = -max_val, max_val

    # Create scatter plot for velocity field
    fig, ax = plt.subplots(
        1, 1, figsize=(7.5, 6), gridspec_kw={"hspace": 0.35, "wspace": 0.25}
    )
    sc1 = ax.scatter(
        index_vlos[0],
        index_vlos[1],
        s=0.025,
        c=val_vlos,
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(
        sc1, ax=ax, label="$V_{los}$ [km/s]"
    )  # Colorbar for velocity field's values
    cbar.set_label("$V_{los}$ [km/s]", fontsize=15)

    # Plot limits and titles
    ax.set_xlim([1024, 0])  # Invert x-axis to fix PA orientation
    ax.set_ylim([0, 1024])
    ax.set_title("d$\phi$=" + f"{dphi}°", fontsize=15)

    # Save plot to specified directory based on the galaxy and projection parameters
    newpath = f"./ID/{gal_exp}_img/dist_{int(dist)}/inc_{int(INC)}/PA_{int(PA)}/"  # Change directory to your personal computer / folder of interest
    os.makedirs(newpath, exist_ok=True)  # In the case you don't create the folder
    plt.savefig(f"{newpath}/dphi_{dphi}.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # Print the final status of the process
    print(f"Data saved in: {newpath}dphi_{dphi}.png")
