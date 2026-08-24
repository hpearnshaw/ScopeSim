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
            
        # Check if you want to back up existing config
            
        # Generate IRDB data files from given inputs
        self.make_reflectivity(infile=config['telescope']['mirror_reflectivity_file'])
        self.make_contamination(thickness_infile=config['telescope']['contamination']['thickness_file'], 
                                coeff_infile=config['telescope']['contamination']['absorption_coefficient_file'],
                                stage=config['telescope']['contamination']['stage'])
        
        # Load detector parameters
        self.n_pixels = config['detector']['n_pixels']
        self.pix_size = u.Quantity(config['detector']['pix_size'])
        self.x_gap = u.Quantity(config['detector']['x_gap'])
        self.y_gap = u.Quantity(config['detector']['y_gap'])
        self.gain = u.Quantity(config['detector']['gain'])
        self.angle = u.Quantity(config['detector']['angle'])

        # Make detector layout
        self.make_detector_layout()
        
        # Load imager parameters
        self.im_pixel_scale = u.Quantity(config['imager']['pixel_scale'])
        self.im_plate_scale = 103.0 * u.arcsec / u.mm
        
        # Make imager inputs
        self.make_qe_curve(infile=config['imager']['nuv_qe_file'])
        self.make_nuv_filter(infile=config['imager']['nuv_filter_file'])
        self.make_fuv_filter(infile=config['imager']['fuv_filter_file'])
        self.make_dichroic_response(infile=config['imager']['dichroic_file'])
        
        # Load LSS parameters
        self.lss_x_0 = u.Quantity(config['lss']['slit_x_0'])
        self.lss_y_0 = u.Quantity(config['lss']['slit_y_0'])
        self.slit_length = u.Quantity(config['lss']['slit_length'])
        self.slit_width = u.Quantity(config['lss']['slit_width'])
        self.lss_pixel_scale = u.Quantity(config['lss']['pixel_scale'])
        self.lss_plate_scale = self.lss_pixel_scale / self.pix_size
        self.lss_wave_min = u.Quantity(config['lss']['wave_min'])
        self.lss_wave_max = u.Quantity(config['lss']['wave_max'])
        self.y_distortion = bool(config['lss']['y_distortion'])
        
        # Make LSS inputs
        self.make_slit_geometry()
        self.make_spectral_efficiency(infile=config['lss']['spectral_efficiency_file'])
        self.make_spectral_trace(indir=config['lss']['detector_psf_dir'])
        self.make_dispersion_file(infile=config['lss']['dispersion_file'])
        self.make_lss_filter_response(infile=config['lss']['filter_file'])

        # Load LSS detector parameters
        self.lss_n_pixels = config['lss_detector']['n_pixels']
        self.lss_pix_size = u.Quantity(config['lss_detector']['pix_size'])
        self.lss_x_gap = u.Quantity(config['lss_detector']['x_gap'])
        self.lss_gain = u.Quantity(config['lss_detector']['gain'])
        self.lss_angle = u.Quantity(config['lss_detector']['angle'])
        self.lss_x1_cen = u.Quantity(config['lss_detector']['x1_cen'])
        self.lss_y1_cen = u.Quantity(config['lss_detector']['y1_cen'])

        # Make LSS detector layout
        self.make_detector_layout_lss()
                
    def make_reflectivity(self, infile="mirror_reflectivity.dat", outfile="mirror_reflectivity.dat"):
        # For now, straight up copy the file over
        # We'll make a parser once new technical data comes in
        import shutil
        shutil.copyfile(os.path.join(self.inputs_dir,infile), os.path.join(self.outputs_dir,outfile))
        
    def make_spectral_efficiency(self, infile="zeiss_blaze_v1.txt", outfile="UVIM_LSS_spectral_efficiency.fits"):
        # Load spectral efficiency file
        spec_eff = np.loadtxt(os.path.join(self.inputs_dir, infile))
        spec_eff_dict = {"wavelength": spec_eff[:, 0] * u.nm, "efficiency": spec_eff[:, 1]}
        # convert from nm to microns
        spec_eff_dict["wavelength"] = spec_eff_dict["wavelength"].to(u.um).value
        # only one trace
        # required fits structure is located in spectral_efficiency in scopesim
        hdu0 = fits.PrimaryHDU()
        hdu0.header["ECAT"] = 1
        hdu0.header["EDATA"] = 2
        hdu0.header["DATE"] = np.datetime64('today', 'D').astype(str)
        hdu0.header["ORIGFILE"] = infile
        hdu1 = fits.BinTableHDU.from_columns(
            [fits.Column(name="description", format="20A", array=["UVIM_LSS_trace"]),
            fits.Column(name="extension_id", format="I", array=[2])]
        )
        hdu2 = fits.BinTableHDU.from_columns(
            [fits.Column(name="wavelength", format="E", array=spec_eff_dict["wavelength"]),
            fits.Column(name="efficiency", format="E", array=spec_eff_dict["efficiency"])]
        )
        hdu2.header["EXTNAME"] = "UVIM_LSS_trace"
        hdul = fits.HDUList([hdu0, hdu1, hdu2])
        hdul.writeto(os.path.join(self.outputs_dir, outfile), overwrite=True)
        
        
    def make_slit_geometry(self, outfile="UVIM_LSS_slit_geometry.dat"):
        # Ensure slit dimensions are in the right units
        slit_length = (self.slit_length).to(u.arcsec).value
        slit_width = (self.slit_width).to(u.arcsec).value
        # relative to the field origin, located at x=3.5 deg, y=0 deg
        # y is the spatial direction, x is the spectral
        x_0 = (self.lss_x_0).to(u.arcsec).value
        y_0 = (self.lss_y_0).to(u.arcsec).value
        slit_coords = np.array([[x_0 - slit_width/2, y_0 - slit_length/2],
                                [x_0 + slit_width/2, y_0 - slit_length/2],
                                [x_0 + slit_width/2, y_0 + slit_length/2],
                                [x_0 - slit_width/2, y_0 + slit_length/2]])
        # write to dat file (allow overwrite)
        with open(os.path.join(self.outputs_dir, outfile), 'w') as f:
            f.write(f"# date_modified : {np.datetime64('today', 'D').astype(str)}\n")
            f.write("# x_unit : arcsec\n")
            f.write("# y_unit : arcsec\n")
            f.write("x    y\n")
            for x, y in zip(slit_coords[:,0], slit_coords[:,1]):
                f.write(f"{x}    {y}\n")
        
    def make_spectral_trace(self, outfile="UVIM_LSS_spectral_trace.fits", indir="LSS_DET_PSF"):
        """Create a spectral trace file for the LSS mode which encodes the distortion along the slit spatial axis."""
        
        det_psf_dir = os.path.abspath(os.path.join(self.inputs_dir, indir))
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

    def make_lss_filter_response(self, infile="graded_overcoat_00nm.csv", outfile="UVIM_LSS_filter_response.dat"):
        # filter response file contains wavelength to transmission mapping
        data = np.loadtxt(os.path.join(self.inputs_dir, infile), skiprows=1, unpack=True, delimiter=",")
        wavelength = data[0] * u.nm
        transmission = data[1] / 100.0 # convert from percentage to fraction
        
        if wavelength[-1] < self.lss_wave_max:
            wavelength = np.append(wavelength, self.lss_wave_max)
            transmission = np.append(transmission, transmission[-1])

        with open(os.path.join(self.outputs_dir, outfile), 'w') as f:
            f.write(f"# date_modified : {np.datetime64('today', 'D').astype(str)}\n")
            f.write(f"# orig_filename: {infile}\n")
            f.write("# wavelength_unit: nm\n")
            f.write("wavelength    transmission\n")
            for wl, trans in zip(wavelength, transmission):
                f.write(f"{wl.value}    {trans:.6f}\n")
                    
    def make_dispersion_file(self, infile="UVEXS_Spectral_Resolution_R2000.txt", outfile="UVIM_LSS_dispersion.dat"):
        data = np.loadtxt(os.path.join(self.inputs_dir, infile), skiprows=2, unpack=True)
        wavelength = data[0] * u.nm
        dispersion = data[2] * u.nm # per pixel
        wavelength = wavelength.to(u.um) # convert to microns
        dispersion = dispersion.to(u.um) # convert to microns per pixel

        # write to dat file (allow overwrite)
        with open(os.path.join(self.outputs_dir, outfile), 'w') as f:
            f.write(f"# date_modified : {np.datetime64('today', 'D').astype(str)}\n")
            f.write(f"# orig_filename: {infile}\n")
            f.write("# wavelength_unit: um\n")
            f.write("# dispersion_unit: um\n")
            f.write("wavelength    dispersion\n")
            for wl, d in zip(wavelength, dispersion):
                f.write(f"{wl.value:.3f}    {d.value:.3g}\n")
        
    def make_qe_curve(self, infile="nuv_qe_Hf02.csv", outfile="UVIM_NUV_QE.dat"):
        data = np.loadtxt(os.path.join(self.inputs_dir, infile), delimiter=',', skiprows=4, unpack=True)
        wavelength = data[0] * u.nm
        qe = data[3] # already a fraction
        wavelength = wavelength.to(u.um) # convert to microns

        with open(os.path.join(self.outputs_dir, outfile), 'w') as f:
            f.write(f"# date_modified : {np.datetime64('today', 'D').astype(str)}\n")
            f.write(f"# orig_filename: {infile}\n")
            f.write("# wavelength_unit: um\n")
            f.write("wavelength    transmission\n")
            for wl, q in zip(wavelength, qe):
                f.write(f"{wl.value:.3f}    {q:.6f}\n")
    
    def make_nuv_filter(self, infile="Materion-NUV-Design-%T.txt", outfile="UVIM_NUV_filter_response.dat"):
        data = np.loadtxt(os.path.join(self.inputs_dir, infile), unpack=True)
        wavelength = data[0] * u.nm
        transmission = data[1] / 100.0 # convert from percentage to fraction

        with open(os.path.join(self.outputs_dir, outfile), 'w') as f:
            f.write(f"# date_modified : {np.datetime64('today', 'D').astype(str)}\n")
            f.write(f"# orig_filename: {infile}\n")
            f.write("# wavelength_unit: nm\n")
            f.write("wavelength    transmission\n")
            for wl, trans in zip(wavelength, transmission):
                f.write(f"{wl.value:.1f}    {trans:.9g}\n")
    
    def make_dichroic_response(self, infile="135_180nm_UVEX_Dichroic_vs_AOI.csv",
                                     outfile="UVIM_dichroic_response.dat"):
        # Note: this same file should be used for the FUV surfaces list, too
        if 'AOI' in infile:
            # Different unpacking for this input file format
            data = np.loadtxt(os.path.join(self.inputs_dir, infile), delimiter=',', skiprows=6, unpack=True)
            wavelength = data[0] * u.nm
            wavelength = wavelength.to(u.um) # convert to microns
            reflection = np.average(data[1:6], axis=0) / 100. # Convert to fraction
            transmission = np.average(data[6:11], axis=0) / 100. # Convert to fraction
        else:
            data = np.loadtxt(os.path.join(self.inputs_dir, infile), delimiter=',', skiprows=4, unpack=True)
            wavelength = data[0] * u.nm
            wavelength = wavelength.to(u.um) # convert to microns
            reflection = data[1]
            transmission = data[2] # already a fraction
        
        with open(os.path.join(self.outputs_dir, outfile), 'w') as f:
            f.write(f"# date_modified : {np.datetime64('today', 'D').astype(str)}\n")
            f.write(f"# orig_filename: {infile}\n")
            f.write("# wavelength_unit: um\n")
            f.write("wavelength    reflection    transmission\n")
            for wl, re, tr in zip(wavelength, reflection, transmission):
                f.write(f"{wl.value:.4f}        {re:.6f}      {tr:.6f}\n")
    
    def make_fuv_filter(self, infile="uvex_fuv_150nmcenter_detector_20250522_longwavelength_correction.csv", outfile="UVIM_FUV_filter_response.dat"):
        data = np.loadtxt(os.path.join(self.inputs_dir, infile), delimiter=',', skiprows=2, unpack=True)
        wavelength = data[0] * u.nm
        transmission = data[2]

        with open(os.path.join(self.outputs_dir, outfile), 'w') as f:
            f.write(f"# date_modified : {np.datetime64('today', 'D').astype(str)}\n")
            f.write(f"# orig_filename: {infile}\n")
            f.write("# wavelength_unit: nm\n")
            f.write("wavelength    transmission\n")
            for wl, trans in zip(wavelength, transmission):
                f.write(f"{wl.value:.1f}    {trans:.9g}\n")

    def make_detector_layout(self, outfile="UVIM_DET_layout.dat"):
        # active detector size in mm
        det_size = (self.n_pixels * self.pix_size).to(u.mm).value

        # gaps in mm
        gx = self.x_gap.to(u.mm).value
        gy = self.y_gap.to(u.mm).value

        # per-detector values
        pixsize = self.pix_size.to(u.mm).value
        gain = self.gain.to(u.electron / u.adu).value
        angle = self.angle.to(u.deg).value

        # center-to-center pitch
        pitch_x = det_size + gx
        pitch_y = det_size + gy

        # detector centers, row-major: A1 A2 A3 / B1 B2 B3 / C1 C2 C3
        x_centers = [-pitch_x, 0.0, pitch_x]
        y_centers = [pitch_y, 0.0, -pitch_y]
        names = [["A1", "A2", "A3"],
                 ["B1", "B2", "B3"],
                 ["C1", "C2", "C3"]]

        with open(os.path.join(self.outputs_dir, outfile), "w") as f:
            # metadata header
            f.write("# author: O. Adegoke\n")
            f.write("# sources:\n")
            f.write("# date_created: 2026-03-30\n")
            f.write(f"# date_modified: {np.datetime64('today', 'D').astype(str)}\n")
            f.write("\n")

            # column header
            f.write("id     x_cen      y_cen     x_size   y_size     pixsize  angle  gain   name\n")

            # detector rows
            det_id = 1
            for i, y in enumerate(y_centers):
                for j, x in enumerate(x_centers):

                    f.write(
                        f"{det_id:>2d}   "
                        f"{x:>8.2f}   "
                        f"{y:>8.2f}   "
                        f"{det_size:>6.2f}   "
                        f"{det_size:>6.2f}   "
                        f"{pixsize:>6.2f}   "
                        f"{angle:>6.1f}   "
                        f"{gain:>5.2f}   "
                        f"{names[i][j]}\n"
                    )
                    det_id += 1

    def make_detector_layout_lss(self, outfile="UVIM_DET_LSS_layout.dat"):
        # active detector size in mm
        det_size = (self.lss_n_pixels * self.lss_pix_size).to(u.mm).value

        # only x-gap is used for LSS
        gx = self.lss_x_gap.to(u.mm).value

        # per-detector values
        pixsize = self.lss_pix_size.to(u.mm).value
        gain = self.lss_gain.to(u.electron / u.adu).value
        angle = self.lss_angle.to(u.deg).value

        # preserve off-center placement of LSS1 from config
        x1 = self.lss_x1_cen.to(u.mm).value
        y1 = self.lss_y1_cen.to(u.mm).value

        # LSS2 is placed to the right of LSS1 by detector width + gap
        x2 = x1 + det_size + gx
        y2 = y1

        names = ["LSS1", "LSS2"]
        centers = [(x1, y1), (x2, y2)]

        with open(os.path.join(self.outputs_dir, outfile), "w") as f:
            f.write("# author: O. Adegoke\n")
            f.write("# sources:\n")
            f.write("# date_created: 2026-05-18\n")
            f.write(f"# date_modified: {np.datetime64('today', 'D').astype(str)}\n")
            f.write("\n")
            f.write("id    x_cen       y_cen     x_size   y_size     pixsize  angle  gain   name\n")

            for det_id, ((x, y), name) in enumerate(zip(centers, names), start=1):
                f.write(
                    f"{det_id:>2d}   "
                    f"{x:>8.4f}   "
                    f"{y:>8.4f}   "
                    f"{det_size:>6.2f}   "
                    f"{det_size:>6.2f}   "
                    f"{pixsize:>7.2f}   "
                    f"{angle:>5.1f}   "
                    f"{gain:>5.2f}   "
                    f"{name}\n"
                )
        
    def make_contamination(self, thickness_infile="contam_thickness.csv", coeff_infile="contam_absorption_coeff.txt", stage='eol'):
        # load thickness file and num film passes to dict on component by component basis
        thickness = {}
        num_films = {}
        components = {}
        # in contam_thickness.csv each row corresponds to a different component
        # columns are component name, bol thickness, eol thickness, number of film passes
        with open(os.path.join(self.inputs_dir, thickness_infile), 'r', encoding='utf-8') as f:
            lines = f.readlines()[2:] # comment and header at first two lines
            for line in lines:
                if line.strip():
                    data = line.strip().split(',')
                    dict_str = str(data[0].strip())
                    thickness[dict_str] = {'bol': (int(data[1])*u.AA).to(u.nm), 'eol': (int(data[2])*u.AA).to(u.nm)} # convert from angstrom to nm
                    num_films[dict_str] = int(data[3])
        
        components['NUV'] = ['M1', 'M2', 'M3', 'Dichroic', 'NUVSurface']
        components['FUV'] = ['M1', 'M2', 'M3', 'Dichroic', 'FUVSurface']
        components['LSS'] = ['M1', 'M2', 'M3', 'SM1', 'Grating', 'SM2', 'LSSSurface']
        
        eff_thickness = {'NUV': 0, 'FUV': 0, 'LSS': 0} # units nm
        for inst in components.keys():
            for comp in components[inst]:
                if comp not in thickness:
                    raise ValueError(f"Components must be one of 'M1', 'M2', 'M3', "
                                    f"'Dichroic', 'NUVSurface', 'FUVSurface', 'SM1', "
                                    f"'Grating', 'SM2', 'LSSSurface' and received {comp}")
                else:
                    eff_thickness[inst] += thickness[comp][stage] * num_films[comp]

        # coefficients per nm wavelength
        data = np.genfromtxt(os.path.join(self.inputs_dir, coeff_infile), delimiter=None)
        wave = data[:,0] * u.nm
        abs_coeff = data[:,1] / u.nm

        for inst in ('NUV', 'FUV', 'LSS'):
            outfile = 'UVIM_' + inst + '_contamination.dat'
            response = np.exp(-(abs_coeff * eff_thickness[inst]).value)
            with open(os.path.join(self.outputs_dir, outfile), 'w') as f:
                f.write(f"# date_modified : {np.datetime64('today', 'D').astype(str)}\n")
                f.write(f"# stage: {stage}\n")
                f.write(f"# orig_filename: {thickness_infile}  {coeff_infile}\n")
                f.write("# action: transmission\n")
                f.write("# wavelength_unit: nm\n")
                f.write(" \n")
                f.write("wavelength    transmission\n")
                for wl, r in zip(wave, response):
                    f.write(f"{wl.value:.1f}    {r:.9g}\n")
    
if __name__ == "__main__":
    # run python3 make_inputs.py from command line
    # for now, this just makes all input files at once
    config = UVEXInputs()
