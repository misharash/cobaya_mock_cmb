###################################
# MOCK CMB TYPE LIKELIHOOD
# --> mock planck, cmbpol, etc.
# Based on MontePython's Likelihood_mock_cmb
# and Cobaya's example of likelihood class
###################################
from cobaya.likelihood import Likelihood
import numpy as np
import os


class MockCMBLikelihood(Likelihood):

    def initialize(self):
        """
         Compute noise and load fiducial power spectrum, if it exists
        """
        ################
        # Noise spectrum
        ################

        # convert arcmin to radians
        self.theta_fwhm *= np.array([np.pi/60/180])
        self.sigma_T *= np.array([np.pi/60/180])
        self.sigma_P *= np.array([np.pi/60/180])

        # compute noise in muK**2
        self.noise_T = np.zeros(self.l_max+1, 'float64')
        self.noise_P = np.zeros(self.l_max+1, 'float64')

        ell = np.arange(self.l_min, self.l_max+1)
        for channel in range(self.num_channels):
            self.noise_T[ell] += self.sigma_T[channel]**-2 *\
                np.exp(
                    -ell*(ell+1)*self.theta_fwhm[channel]**2/8/np.log(2))
            self.noise_P[ell] += self.sigma_P[channel]**-2 *\
                np.exp(
                    -ell*(ell+1)*self.theta_fwhm[channel]**2/8/np.log(2))
        self.noise_T[ell] = 1/self.noise_T[ell]
        self.noise_P[ell] = 1/self.noise_P[ell]

        ###########
        # Read data
        ###########

        # If the file exists, initialize the fiducial values
        self.Cl_fid = np.zeros((3, self.l_max+1), 'float64')
        self.fid_values_exist = False
        if not self.data_directory:
            self.data_directory = os.path.dirname(os.path.realpath(__file__))
        fiducial_filename = os.path.join(
                self.data_directory, self.fiducial_file)
        if os.path.exists(fiducial_filename):
            self.fid_values_exist = True
            fiducial_content = np.loadtxt(fiducial_filename).T
            ll = fiducial_content[0].astype(int)
            self.Cl_fid[:, ll] = fiducial_content[1:4]

        # Else the file should be created in the create_fid_values() function.

    def get_requirements(self):
        """
        here we need C_L^{tt, te, ee} to l_max
        """
        return {'Cl': {'tt': self.l_max, 'te': self.l_max, 'ee': self.l_max}}

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values
        and return a log-likelihood.

        here we calculate chi^2  using cl's
        """
        # if fiducial values exist
        if self.fid_values_exist:
            # get Cl's from the cosmological code in muK**2
            cl = self.provider.get_Cl(ell_factor=True, units='muK2')

            # get likelihood
            return self.compute_lkl(cl, params_values)

        # otherwise warn and return -inf
        self.log.warning(
            "Fiducial model not loaded")
        return -np.inf

    def compute_lkl(self, cl, params_values):

        # compute likelihood

        ell = np.arange(self.l_min, self.l_max+1)

        Cov_obs = np.array([
            [self.Cl_fid[0, self.l_min:self.l_max+1],
             self.Cl_fid[2, self.l_min:self.l_max+1]],
            [self.Cl_fid[2, self.l_min:self.l_max+1],
             self.Cl_fid[1, self.l_min:self.l_max+1]]])
        Cov_the = np.array([
            [cl['tt'][self.l_min:self.l_max+1] +
             self.noise_T[self.l_min:self.l_max+1],
             cl['te'][self.l_min:self.l_max+1]],
            [cl['te'][self.l_min:self.l_max+1],
             cl['ee'][self.l_min:self.l_max+1] +
             self.noise_P[self.l_min:self.l_max+1]]])

        det_obs = Cov_obs[1, 1]*Cov_obs[0, 0]-Cov_obs[1, 0]*Cov_obs[0, 1]
        det_the = Cov_the[1, 1]*Cov_the[0, 0]-Cov_the[1, 0]*Cov_the[0, 1]
        det_mix = np.zeros(self.l_max-self.l_min+1)

        for i in range(2):
            Cov_mix = np.copy(Cov_the)
            Cov_mix[i] = Cov_obs[i]
            det_mix += Cov_mix[1, 1]*Cov_mix[0, 0]-Cov_mix[1, 0]*Cov_mix[0, 1]

        chi2 = np.sum((2.*ell+1.)*self.f_sky *
                      (det_mix/det_the + np.log(det_the/det_obs) - 2))

        return -chi2/2

    def create_fid_values(self, cl, params, override=False):
        # Write fiducial model spectra if needed
        # params should be the dictionary of cosmological parameters
        # corresponding to the cl provided
        fid_filename = os.path.join(self.data_directory,
                                    self.fiducial_file)
        if not self.fid_values_exist or override:
            # header string with parameter values
            header_str = 'Fiducial parameters: '
            for key, value in params.items():
                value = str(value)
                header_str += '%s = %s, ' % (key, value)
            header_str = header_str[:-2]
            # output arrays
            ell = np.arange(self.l_min, self.l_max+1)
            out_data = np.array((ell, cl['tt'][ell]+self.noise_T[ell],
                                cl['ee'][ell]+self.noise_P[ell],
                                cl['te'][ell])).T
            # write data
            np.savetxt(fid_filename, out_data, "%.8g", header=header_str)
            self.fid_values_exist = True
            self.log.info(
                "Writing fiducial model in %s" % fid_filename)
            return True
        self.log.warning(
            "Fiducial model in %s already exists" % fid_filename)
        return False
