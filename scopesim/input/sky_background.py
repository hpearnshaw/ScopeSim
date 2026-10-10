"""Sky background for UVEX.

The sky is the sum of three parts:
    - zodiacal light (sunlight scattered off dust in the solar system)
    - galactic light (starlight scattered off dust in our galaxy)
    - Lyman-alpha (glow from hydrogen around the Earth, FUV only)

The sky changes slowly across the field, so we only work out how bright it is
at the four corners and fill in the rest with straight-line (bilinear)
interpolation. Each part becomes its own ScopeSim Source with its own spectrum,
and the parts are added together.

The sky models are copied from uvex-imager-etc/backgrounds.py.
"""
import warnings
from functools import reduce
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, get_sun, GeocentricTrueEcliptic
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from synphot import SourceSpectrum
from synphot.models import ConstFlux1D, Empirical1D, GaussianFlux1D
from synphot.units import PHOTLAM

from ..source.source import Source


# The models give brightness per steradian; divide by this to get per arcsec^2.
ARCSEC2_PER_SR = (1 * u.sr).to_value(u.arcsec**2)


# --- Zodiacal light -----------------------------------------------------------
# Leinert et al. (1998) Table 17.
# Rows: ecliptic longitude away from the Sun. Columns: ecliptic latitude.
# NaN means too close to the Sun to observe.
ZODI_LON = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                     60, 75, 90, 105, 120, 135, 150, 165, 180], float)
ZODI_LAT = np.array([0, 5, 10, 15, 20, 25, 30, 45, 60, 75], float)
_ = np.nan
ZODI_TABLE = np.array([
    [    _,    _,    _, 3140, 1610,  985,  640,  275,  150, 100],
    [    _,    _,    _, 2940, 1540,  945,  625,  271,  150, 100],
    [    _,    _, 4740, 2470, 1370,  865,  590,  264,  148, 100],
    [11500, 6780, 3440, 1860, 1110,  755,  525,  251,  146, 100],
    [ 6400, 4480, 2410, 1410,  910,  635,  454,  237,  141,  99],
    [ 3840, 2830, 1730, 1100,  749,  545,  410,  223,  136,  97],
    [ 2480, 1870, 1220,  845,  615,  467,  365,  207,  131,  95],
    [ 1650, 1270,  910,  680,  510,  397,  320,  193,  125,  93],
    [ 1180,  940,  700,  530,  416,  338,  282,  179,  120,  92],
    [  910,  730,  555,  442,  356,  292,  250,  166,  116,  90],
    [  505,  442,  352,  292,  243,  209,  183,  134,  104,  86],
    [  338,  317,  269,  227,  196,  172,  151,  116,   93,  82],
    [  259,  251,  225,  193,  166,  147,  132,  104,   86,  79],
    [  212,  210,  197,  170,  150,  133,  119,   96,   82,  77],
    [  188,  186,  177,  154,  138,  125,  113,   90,   77,  74],
    [  179,  178,  166,  147,  134,  122,  110,   90,   77,  73],
    [  179,  178,  165,  148,  137,  127,  116,   96,   79,  72],
    [  196,  192,  179,  165,  151,  141,  131,  104,   82,  72],
    [  230,  212,  195,  178,  163,  148,  134,  105,   83,  72],
])
zodi_lookup = RegularGridInterpolator((ZODI_LON, ZODI_LAT), ZODI_TABLE)

# The zodiacal spectrum, per unit of the table value above (same file the ETC uses).
ZODI_SPEC = np.loadtxt(Path(__file__).parent / "data" / "scaled_zodiacal_spec.txt")


def zodi_level(coords, time):
    """Zodiacal table value toward each coordinate on a given date."""
    frame = GeocentricTrueEcliptic(equinox=time)
    sun = get_sun(time).transform_to(frame)
    ecl = coords.transform_to(frame)
    lat = np.abs(np.atleast_1d(ecl.lat.deg))
    lon = np.abs(np.atleast_1d((ecl.lon - sun.lon).wrap_at(180 * u.deg).deg))

    # The table stops at 75 deg latitude; beyond that the sky is flat at 72.
    level = np.where(lat > 75, 72.0, zodi_lookup(np.c_[lon, np.minimum(lat, 75)]))
    if not np.all(np.isfinite(level)):
        raise ValueError("Pointing is too close to the Sun for the zodiacal model")
    return level


def zodi_spectrum(level):
    flux = ZODI_SPEC[:, 1] * level / ARCSEC2_PER_SR
    return SourceSpectrum(Empirical1D, points=ZODI_SPEC[:, 0] * u.AA,
                          lookup_table=flux * PHOTLAM)


# --- Galactic light -----------------------------------------------------------
# Fit of the form a + c / sin|b|, with a different fit for each half of the sky.
GALACTIC_FIT = {"fuv": {"north": (93.4, 133.2), "south": (-205.5, 401.8)},
                "nuv": {"north": (257.5, 185.1), "south": (66.7, 356.3)}}


def galactic_level(coords, mode):
    """Galactic light toward each coordinate, in photons/s/cm^2/A/arcsec^2."""
    b = np.atleast_1d(coords.galactic.b.deg)
    if np.any(np.abs(b) < 15):
        warnings.warn("The galactic light model isn't reliable within 15 deg of the galactic plane")
    csc = 1.0 / np.abs(np.sin(np.radians(b)))
    a_north, c_north = GALACTIC_FIT[mode]["north"]
    a_south, c_south = GALACTIC_FIT[mode]["south"]
    per_sr = np.where(b >= 0, a_north + c_north * csc, a_south + c_south * csc)
    return per_sr / ARCSEC2_PER_SR


def flat_spectrum(flux):
    """Same brightness at every wavelength."""
    return SourceSpectrum(ConstFlux1D, amplitude=flux * PHOTLAM)


# --- Lyman-alpha --------------------------------------------------------------
RAYLEIGH = 3.15e-17 * u.erg / (u.cm**2 * u.s)   # 1 rayleigh, per arcsec^2

# The ETC uses a 0.1 A wide line. ScopeSim samples the spectrum too coarsely
# to see a line that narrow, so we widen it to 5 A. The total flux is the same.
LYMAN_ALPHA_WIDTH = 5 * u.AA


def lyman_alpha_spectrum(kilorayleighs):
    return SourceSpectrum(GaussianFlux1D, mean=1216 * u.AA, fwhm=LYMAN_ALPHA_WIDTH,
                          total_flux=kilorayleighs * 1e3 * RAYLEIGH)


# --- Corners and interpolation ------------------------------------------------
def field_corners(pointing, field_width, roll_angle=0 * u.deg):
    """Sky positions of the four field corners: lower-left, lower-right,
    upper-left, upper-right.

    With roll_angle = 0, north is up and east is left. A positive roll
    turns the field east of north.
    """
    half = 0.5 * field_width * u.arcsec
    corners = []
    for x, y in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        east, north = -x * half, y * half
        angle = np.arctan2(east, north) + roll_angle
        corners.append(pointing.directional_offset_by(angle, np.hypot(east, north)))
    return SkyCoord(corners)


def bilinear(x, y, corners):
    """Fill in the field from its four corner values. x and y go from -1 to +1."""
    lower_left, lower_right, upper_left, upper_right = corners
    s, t = (x + 1) / 2, (y + 1) / 2
    return ((1 - s) * (1 - t) * lower_left + s * (1 - t) * lower_right
            + (1 - s) * t * upper_left + s * t * upper_right)


# --- Turning it into ScopeSim sources -----------------------------------------
class SkySource(Source):
    """A ScopeSim Source that can be added to other sources.

    ScopeSim has a bug: adding sources that each have their own spectrum mixes
    up which spectrum belongs to which field. This puts them back after adding.
    """

    def append(self, source_to_add):
        n = len(self.fields)
        super().append(source_to_add)
        for field in self.fields[n:]:
            ref = field.header.get("SPEC_REF") if hasattr(field, "header") else None
            if isinstance(ref, int) and ref in field.spectra:
                field.spectra = {ref: field.spectra[ref]}

    def __add__(self, other):
        total = SkySource()
        total.append(self)
        total.append(other)
        return total

    def __radd__(self, other):
        total = SkySource()
        total.append(other)
        total.append(self)
        return total


def sky_part_source(corner_levels, make_spectrum, field_width, grid_pitch, margin):
    """One part of the sky as a ScopeSim source.

    The spectrum is set by the average of the four corners, and an image
    (1 on average) spreads it across the field.
    """
    corner_levels = np.asarray(corner_levels, float)
    average = corner_levels.mean()
    spectrum = make_spectrum(average)

    # Grid of points covering the field plus a margin, in units where the
    # field edges are at -1 and +1.
    half = 0.5 * field_width
    n = int(np.ceil((half + margin) / grid_pitch))
    # The +0.5 fixes ScopeSim placing image sources half a pixel off.
    axis = (np.arange(-n, n + 1) + 0.5) * grid_pitch / half
    x, y = np.meshgrid(axis, axis)
    image = bilinear(x, y, corner_levels / average)

    header = fits.Header({
        "NAXIS": 2, "NAXIS1": axis.size, "NAXIS2": axis.size,
        "CTYPE1": "LINEAR", "CTYPE2": "LINEAR", "CUNIT1": "deg", "CUNIT2": "deg",
        "CDELT1": grid_pitch / 3600, "CDELT2": grid_pitch / 3600,
        "CRPIX1": n + 1, "CRPIX2": n + 1, "CRVAL1": 0.0, "CRVAL2": 0.0,
        "BUNIT": "photlam arcsec-2", "SPEC_REF": 0,
    })
    hdu = fits.ImageHDU(data=image.astype(np.float32), header=header)
    return SkySource(image_hdu=hdu, spectra=[spectrum])


def make_sky_background(pointing, time, mode, field_width, roll_angle=0 * u.deg,
                        pixel_scale=1.03, grid_pitch=None,
                        lyman_alpha_kr=2.0, scattered_light=0.0):
    """Build the UVEX sky background as a ScopeSim Source.

    pointing        SkyCoord of the field center
    time            astropy Time of the observation (tells us where the Sun is)
    mode            "nuv" or "fuv"
    field_width     width of the square field to cover, in arcsec
    roll_angle      rotation of the field, east of north
    pixel_scale     detector pixel size in arcsec
    grid_pitch      spacing of the background grid in arcsec. Defaults to
                    4 pixels, which is accurate to ~0.1% and keeps memory low
                    for the full mosaic.
    lyman_alpha_kr  Lyman-alpha brightness in kilorayleighs (FUV only)
    scattered_light extra fraction of zodiacal + galactic light for stray
                    light (the ETC uses 0.5)
    """
    if mode not in ("nuv", "fuv"):
        raise ValueError(f"Sky backgrounds are only set up for 'nuv' and 'fuv', not {mode!r}")
    if grid_pitch is None:
        grid_pitch = 4 * pixel_scale
    margin = 8 * grid_pitch   # extend a little past the edges so nothing is cut off

    corners = field_corners(pointing, field_width, roll_angle)
    boost = 1 + scattered_light
    grid = dict(field_width=field_width, grid_pitch=grid_pitch, margin=margin)

    parts = [
        sky_part_source(zodi_level(corners, time) * boost, zodi_spectrum, **grid),
        sky_part_source(galactic_level(corners, mode) * boost, flat_spectrum, **grid),
    ]
    if mode == "fuv":   # the NUV band can't see 1216 A
        lya = lyman_alpha_spectrum(lyman_alpha_kr * boost)
        parts.append(sky_part_source(np.ones(4), lambda level: lya, **grid))

    return reduce(lambda a, b: a + b, parts)
