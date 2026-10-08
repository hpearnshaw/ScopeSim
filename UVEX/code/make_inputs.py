import numpy as np
from astropy.io import fits
from astropy import units as u
import os
import yaml

class UVEXInputs:

    def __init__(self):
    
        # Define input and output directories
        self.uvex_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.inputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "inputs/"))
        self.outputs_dir = os.path.abspath(os.path.join(self.uvex_dir, "data_files/"))
    
        # Ingest configuration file
        with open(os.path.join(self.uvex_dir,"config.yaml"), 'r') as f:
            config = yaml.safe_load(f)
        
        # Make LSS inputs
        self.y_distortion = bool(config['lss']['y_distortion'])
        self.make_spectral_trace(indir=config['lss']['detector_psf_dir'])

    def make_spectral_trace(self, outfile="UVIM_LSS_spectral_trace.fits", indir="LSS_DET_PSF"):
        """Create a spectral trace file for the LSS mode which encodes the distortion along the slit spatial axis."""
        
        det_psf_dir = os.path.abspath(os.path.join(self.outputs_dir, indir))
        det_psf_files = [f for f in os.listdir(det_psf_dir) if f.endswith('.fits')]
        det_psf_files = sorted(det_psf_files)

        x_pos_det = []
        y_pos_det = []
        #x_fld_det = []
        y_fld_det = []
        cen_wave_det = []
        for f in det_psf_files:
            hdu = fits.open(os.path.join(det_psf_dir, f))[0]
            x_pos_det.append(hdu.header["XPOS"])
            y_pos_det.append(hdu.header["YPOS"])
            #x_fld_det.append(hdu.header["XFLD"])
            y_fld_det.append(hdu.header["YFLD"])
            cen_wave_det.append(hdu.header["CEN_WAVE"])
        
        cen_wave_det, y_fld_det = np.array(cen_wave_det), np.array(y_fld_det)
        x_pos_det, y_pos_det = np.array(x_pos_det), np.array(y_pos_det)
        sortidx = np.lexsort((cen_wave_det, y_fld_det))
        cen_wave_det, y_fld_det = cen_wave_det[sortidx], y_fld_det[sortidx]
        x_pos_det, y_pos_det = x_pos_det[sortidx], y_pos_det[sortidx]
        
        # 11 points along slit spatial direction, 25 points along the wavelength direction
        # Position along slit s maps to detector position y, and wavelength maps to detector position x 
        if self.y_distortion:
            s_grid = (np.array(y_fld_det) * u.deg).to(u.arcsec).value # convert from deg to arcsec
            y_grid = np.array(y_pos_det) # already in mm
            wavelength_grid = (np.array(cen_wave_det) * u.nm).to(u.um).value # convert from nm to microns
            x_grid = np.array(x_pos_det) # already in mm
        else:
            unique_xi = np.unique(y_fld_det)
            ids_xi = [25*i for i in range(11)]
            my, by = np.polyfit(unique_xi, y_pos_det[ids_xi],1)
            y_grid = np.repeat(my * unique_xi + by, 25)
            s_grid = (np.array(y_fld_det) * u.deg).to(u.arcsec).value
            wave_arr = (np.array(cen_wave_det) * u.nm).to(u.um).value
            wavelength_grid = (np.array(cen_wave_det) * u.nm).to(u.um).value
            x_grid = np.array(x_pos_det)

        # Write to fits file in the format SpectralTraceList expects
        hdu0 = fits.PrimaryHDU()
        hdu0.header["ECAT"] = 1
        hdu0.header["EDATA"] = 2
        hdu0.header["DATE"] = np.datetime64('today', 'D').astype(str)
        hdu0.header["ORIGFILE"] = str(indir)
        hdu1 = fits.BinTableHDU.from_columns(
            [fits.Column(name="description", format="20A", array=["UVIM_LSS_trace"]),
            fits.Column(name="extension_id", format="I", array=[2]),
            fits.Column(name="aperture_id", format="I", array=[0]),
            fits.Column(name="image_plane_id", format="I", array=[0])]
        )
        hdu2 = fits.BinTableHDU.from_columns(
            [fits.Column(name="wavelength", format="E", array=wavelength_grid),
            fits.Column(name="s", format="E", array=s_grid),
            fits.Column(name="x", format="E", array=x_grid),
            fits.Column(name="y", format="E", array=y_grid)]
        )
        hdu2.header["EXTNAME"] = "UVIM_LSS_trace"
        hdu2.header["DISPDIR"] = "x"
        hdu2.header["TUNIT1"] = "um"
        hdu2.header["TUNIT2"] = "arcsec"
        hdu2.header["TUNIT3"] = "mm"
        hdu2.header["TUNIT4"] = "mm"
        hdu2.header["WAVECOLN"] = "wavelength"
        hdu2.header["SLITPOSN"] = "s"
        hdul = fits.HDUList([hdu0, hdu1, hdu2])
        hdul.writeto(os.path.join(self.outputs_dir, outfile), overwrite=True)

    
if __name__ == "__main__":
    # run python3 make_inputs.py from command line
    # for now, this just makes all input files at once
    config = UVEXInputs()
