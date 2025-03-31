import argparse
import numpy as np
import camb
import fitsio
import healpy as hp
import os
from cosmology import Cosmology
import time

#GLASS modules
import glass
import glass.ext.camb

def get_parser():
    """
    Creates the parser
    """
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description='''
        Creates a UNIONS simulations using GLASS
        ''',
        fromfile_prefix_chars='@'
    )

    parser.add_argument("-s", "--seed", help="Random seed", type=int, default=42)
    parser.add_argument("-N", "--number", help="Mock Number", type=int, default=0)
    parser.add_argument("-n", "--nside", help="Nside for the simulation. Nside=Lmax", type=int, default=32)
    parser.add_argument("-ne", "--neff", help="Effective number of galaxies per arcmin^2", type=float, default=7.127619407438543)
    parser.add_argument("-p", "--path", help="Output path to save the mocks", type=str, default='./')
    parser.add_argument("-c", "--cls", help="Pre-compute and saves the matter shell cls", action="store_true")
    parser.add_argument("-t", "--test", help="Test run", action="store_true")
    parser.add_argument("-sg", "--sigmae", help="Set sigma of intrinsic ellipticity", type=float, default=0.3093694972219397)
    parser.add_argument("-cb", "--camb", help="get Camb C_ell", action="store_true")
    parser.add_argument("-nz", "--pathnz", help="Path to the n(z) file", type=str, default='/n17data/mkilbing/astro/data/CFIS/v1.0/nz/dndz_SP_A.txt')
    parser.add_argument("-m", "--mask", help="Path to the mask", type=str, default='/n09data/guerrini/glass_mock/mask_v1.4.5_nside4096.fits')
    return parser

def downgrade_mask(mask, nside):
    """
    Downgrades a mask to a lower nside
    """
    nside_mask = hp.get_nside(mask)
    if nside_mask == nside:
        return mask
    else:
        print(f"[!] Downgrading mask from Nside {nside_mask} to Nside {nside}.")
        print(f"[!] Pixels with values <0.75 will be set to 0.0")
        print(f"[!] Pixels with values >0.75 will be set to 1.0")
        mask_down = hp.ud_grade(mask, nside)
        mask_down[mask_down < 0.75] = 0.0
        mask_down[mask_down >= 0.75] = 1.0
        print(f"[!] Done.")
        return mask_down
    
class Sky:
    def __init__(self):
        #set the constants
        self.nbins = 1 #number of shells
        #constant bias
        self.bz = 1.2,
        self.phz_sigma_0 = 0.03
        #sets up the simulation parameters
        self.h = 0.7
        self.Oc = 0.25
        self.Ob = 0.05
        self.dx = 200.0
        self.zmax = 3.0
        #get variables from parser
        parser = get_parser()
        args = parser.parse_args()
        self.n_arcmin2 = args.neff
        self.nside = args.nside
        self.test = args.test
        self.camb = args.camb
        self.pre_cls = args.cls
        self.number = args.number
        self.n_sim = str(args.number).zfill(5)
        self.path = args.path
        self.path_mask = args.mask
        self.sigma_e = args.sigmae

        #sets the random seed
        self.rng = np.random.default_rng(args.seed)
        self.lmax = self.nside
        self.path_nz = args.pathnz
        if args.test:
            print("[!!!] Running in test mode [!!!]")
            self.n_arcmin2 = 0.0824
            self.nside = 32
            self.lmax = 32
            self.dx = 120.0
            self.zmax = 0.5

        print("------------------------------------------------------------------")
        print(f"Creating a UNIONS GLASS simulation with NSide = {self.nside}")
        print(f"Mock number: {self.number}")
        print(f"Random seed: {args.seed}")
        print(f"Test mode or not: {args.test}")
        print("------------------------------------------------------------------")

        #Result folder
        self.root = f'{self.path}/results'
        if not os.path.exists(self.root):
            os.makedirs(self.root)
            print("A new directory" + str(self.root)+"is created!")

        self.pars = camb.set_params(H0=100*self.h, omch2=self.Oc*self.h**2, ombh2=self.Ob*self.h**2,
                                    NonLinear=camb.model.NonLinear_both, WantTransfer=True)
        
        params_dict = vars(self.pars)
        print("------------------------------------------------------------------")
        print("Parameters used for the simulation:")
        print(f"H0: {self.pars.H0}")
        print(f"omch2: {self.pars.omch2}")
        print(f"ombh2: {self.pars.ombh2}")
        print(f"n_s: {self.pars.InitPower.ns}")
        results = camb.get_results(self.pars)
        sigma8 = results.get_sigma8()
        print(f"sigma_8: {sigma8}")
        print(f"Omega_m: {self.pars.omegam}")
        print(f"S8: {sigma8*np.sqrt(self.pars.omegam/0.3)}")
        print("------------------------------------------------------------------")
        
        #Initialize the variables for the functions
        self.z = None
        self.bin_z = None
        self.camb_cls = None
        self.numbers = None
        self.shear = None
        self.shear_noise = None
        self.sim_cls = None
        self.noise_cls = None
        return
    
    #get a simulation of the galaxies in the sky, with ellipticity and redshift
    def galaxies_simulation(self):
        """
        Creates a UNIONS simulation using GLASS
        """
        #reads the parser
        cosmo = Cosmology.from_camb(self.pars)

        #reads the mask
        unions_mask = hp.read_map(self.path_mask)

        mask_file_name = f'{self.path}/mask_nside{self.nside}.fits'

        if not os.path.isfile(mask_file_name):
            if not os.path.exists(f'{self.path}'):
                os.makedirs(f'{self.path}')
            #downgrades the mask to the desired nside
            unions_mask = downgrade_mask(unions_mask, self.nside)
            hp.write_map(mask_file_name, unions_mask, overwrite=True)
        else:
            unions_mask = hp.read_map(mask_file_name).astype(np.float64)

        #Setting up the matter density fields
        zb = glass.distance_grid(cosmo, 0.0, self.zmax, dx=self.dx)
        #Shells in comoving distance
        shells = glass.linear_windows(zb)

        #Gets the matter angular power spectrum from camb
        if self.pre_cls:
            cls = self.pre_cls
            np.save(f'{self.root}/cls_test{self.test}.npy')
        else:
            try:
                cls = np.load(f'{self.root}/cls_test{self.test}.npy')
            except FileNotFoundError:
                print(f"[!] The cls file for lmax={self.lmax} does not exist.")
                print(f"[!] Running the matter density cls calculation ...")
                cls = glass.ext.camb.matter_cls(self.pars, self.lmax, shells)
                np.save(f'{self.root}/cls_test{self.test}.npy', cls)
                print(f"[!] Done.")

        #Set up lognormal fields for simulations
        fields = glass.lognormal_fields(shells)
        #Computing the gaussian cls for lognormal fields using 3 correlated shells
        gls = glass.solve_gaussian_spectra(fields, cls)

        #Building the generator for the lognormal fields:
        matter = glass.generate(fields, gls, self.nside, ncorr=3, rng=self.rng)

        #The generator fo the convergence field
        convergence = glass.MultiPlaneConvergence(cosmo)

        #true redshift distribution from UNIONS nz
        nz = np.loadtxt(self.path_nz)
        nz = nz[nz[:, 0] <= self.zmax]
        self.z = nz[:, 0]
        dndz = nz[:, 1]
        dndz *= self.n_arcmin2

        #compute the galaxy number density for each shell
        ngal = glass.partition(self.z, dndz, shells)

        #Compute bin edges with equal density, assuming photometric redshift errors
        zbins = glass.equal_dens_zbins(self.z, dndz, nbins=self.nbins)

        #Setting up the output file
        out_file = f'{self.root}/unions_glass_sim_{self.n_sim}_{self.nside}.fits'

        #creating the catalogue using fits
        fits = fitsio.FITS(out_file, 'rw', clobber=True)
        fits.write(None)
        fits.create_table_hdu(names=['RA', 'Dec', 'e1', 'e2', 'w', 'n1', 'n2', 'TOM_BIN_ID', 'TRUE_Z', 'PHOTO_Z'],
                              formats=['D', 'D', 'E', 'E', 'D', 'E', 'E', 'J', 'D', 'D'],
                              extname='SOURCE_CATALOGUE')
        cat_dtype = fits['SOURCE_CATALOGUE'].get_rec_dtype()[0]

        #Iterate and store the quantities of interest for our mock catalogue
        ngal_tot = 0
        c = 0
        """ kappa_bar = np.zeros(hp.nside2npix(self.nside))
        gamm1_bar = np.zeros(hp.nside2npix(self.nside))
        gamm2_bar = np.zeros(hp.nside2npix(self.nside)) """
        print("Generating shell ", end='')
        for i, delta_i in enumerate(matter):
            #compute the lensing maps
            convergence.add_window(delta_i, shells[i])
            kappa_i = convergence.kappa
            gamm1_i, gamm2_i = glass.shear_from_convergence(kappa_i)

            """ kappa_bar += ngal[i] * kappa_i
            gamm1_bar += ngal[i] * gamm1_i
            gamm2_bar += ngal[i] * gamm2_i """

            #generate the galaxy positions from the matter density field:
            for gal_lon, gal_lat, gal_count in glass.points.positions_from_delta(ngal[i], delta_i, self.bz, unions_mask, rng=self.rng):

                ngal_tot += gal_count

                #generate random redshifts given the provided n(z) distribution
                gal_z = glass.redshifts(gal_count, shells[i], rng=self.rng)

                # generator photometric redshifts using a Gaussian model
                gal_phz = glass.gaussian_phz(gal_z, self.phz_sigma_0, rng=self.rng)

                # attach tomographic bin IDs to galaxies, based on photometric redshifts
                tomo_id = np.digitize(gal_phz, np.unique(zbins)) - 1

                #galaxy ellipticities
                gal_ellip = glass.ellipticity_intnorm(gal_count, self.sigma_e, rng=self.rng)
                #galaxy shears:
                gal_she = glass.galaxy_shear(gal_lon, gal_lat, gal_ellip, kappa_i, gamm1_i, gamm2_i)

                noise_she = glass.galaxies.galaxy_shear(gal_lon, gal_lat, gal_ellip, np.zeros(np.shape(kappa_i)),
                                                        np.zeros(np.shape(gamm1_i)), np.zeros(np.shape(gamm2_i)))
            
                #constructs the catalogue
                catalogue = np.empty(gal_count, dtype=cat_dtype)
                catalogue['RA'] = gal_lon
                catalogue['Dec'] = gal_lat
                catalogue['e1'] = gal_she.real
                catalogue['e2'] = -gal_she.imag
                catalogue['w'] = np.ones_like(gal_lon)
                catalogue['n1'] = noise_she.real
                catalogue['n2'] = noise_she.imag
                catalogue['TOM_BIN_ID'] = tomo_id
                catalogue['TRUE_Z'] = gal_z
                catalogue['PHOTO_Z'] = gal_phz

                fits['SOURCE_CATALOGUE'].append(catalogue)
                c += 1

        """
        kappa_bar /= ngal.sum()
        gamm1_bar /= ngal.sum()
        gamm2_bar /= ngal.sum()
        """

        print("[DONE] \n")
        print(f"Total number of galaxies sampled: {ngal_tot} using {c} z-shells for integration")
        fits.close()
        print("Saved simulation to: ", out_file)

        if self.camb:
            dictionary = self.get_camb_cls()
        return
    
    def get_camb_cls(self, sav=True):
        bin_nz = self.bin_nz
        z = self.z
        nbins = self.nbins
        lmax = self.lmax

        if bin_nz is None or z is None:
            print("ERROR: You should run galaxies_simulation() at first to get the necessary parameters")
            return None
        
        sources = []
        for i in range(nbins):
            s = camb.sources.SplinedSourceWindow(z=z, W=bin_nz[i], source_type='lensing')
            sources.append(s)
        self.pars.SourceWindows = sources
        self.pars.InitPower.set_params(As=2e-9, ns=0.965)
        results = camb.get_results(self.pars)
        cl_camb = results.get_source_cls_dict(lmax=lmax, raw_cl=True)

        dic = dict()
        dic['ell'] = np.arange(lmax+1)+1
        for i, key in enumerate(cl_camb):
            if 'P' not in key:
                st = key.replace('W', '').replace('x', '-')
                a, b = st.split('-')
                a = str(int(a)-1)
                b = str(int(b)-1)
                new_key = a + '-' + b
                dic[new_key] = cl_camb[key]

        if sav:
            out_file_camb_cl = f'{self.root}/camb_cls.fits'
            fits = fitsio.FITS(out_file_camb_cl, 'rw', clobber=True)
            fits.write(dic)
            fits.close()
        self.camb_cls = dic
        return dic
    
if __name__ == "__main__":
    print("Starting the simulation")
    start_time = time.time()
    new_sky = Sky()
    print("--- init %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    new_sky.galaxies_simulation()
    print("--- simulation %s seconds ---" % (time.time() - start_time))