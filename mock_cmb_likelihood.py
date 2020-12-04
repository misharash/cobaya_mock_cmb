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

        for l in range(self.l_min, self.l_max+1):
            self.noise_T[l] = 0
            self.noise_P[l] = 0
            for channel in range(self.num_channels):
                self.noise_T[l] += self.sigma_T[channel]**-2 *\
                    np.exp(
                        -l*(l+1)*self.theta_fwhm[channel]**2/8/np.log(2))
                self.noise_P[l] += self.sigma_P[channel]**-2 *\
                    np.exp(
                        -l*(l+1)*self.theta_fwhm[channel]**2/8/np.log(2))
            self.noise_T[l] = 1/self.noise_T[l]
            self.noise_P[l] = 1/self.noise_P[l]

        ###########
        # Read data
        ###########

        # If the file exists, initialize the fiducial values
        self.Cl_fid = np.zeros((3, self.l_max+1), 'float64')
        self.fid_values_exist = False
        if not self.data_directory:
            self.data_directory = os.path.dirname(os.path.realpath(__file__))
        if os.path.exists(os.path.join(
                self.data_directory, self.fiducial_file)):
            self.fid_values_exist = True
            fid_file = open(os.path.join(
                self.data_directory, self.fiducial_file), 'r')
            line = fid_file.readline()
            while line.find('#') != -1:
                line = fid_file.readline()
            while (line.find('\n') != -1 and len(line) == 1):
                line = fid_file.readline()
            for l in range(self.l_min, self.l_max+1):
                ll = int(line.split()[0])
                self.Cl_fid[0, ll] = float(line.split()[1])
                self.Cl_fid[1, ll] = float(line.split()[2])
                self.Cl_fid[2, ll] = float(line.split()[3])
                line = fid_file.readline()

        # Else the file will be created in the loglkl() function.

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
        # get Cl's from the cosmological code in muK**2
        cl = self.provider.get_Cl(ell_factor=True, units='muK2')

        # get likelihood
        lkl = self.compute_lkl(cl, params_values)

        return lkl

    def compute_lkl(self, cl, params_values):

        # compute likelihood

        chi2 = 0

        l = np.arange(self.l_min, self.l_max+1)

        Cov_obs = np.array([
            [self.Cl_fid[0, self.l_min:self.l_max+1],
             self.Cl_fid[2, self.l_min:self.l_max+1]],
            [self.Cl_fid[2, self.l_min:self.l_max+1],
             self.Cl_fid[1, self.l_min:self.l_max+1]]])
        Cov_the = np.array([
            [cl['tt'][self.l_min:self.l_max+1] +\
              self.noise_T[self.l_min:self.l_max+1],
             cl['te'][self.l_min:self.l_max+1]],
            [cl['te'][self.l_min:self.l_max+1],
             cl['ee'][self.l_min:self.l_max+1] +\
              self.noise_P[self.l_min:self.l_max+1]]])

        det_obs = Cov_obs[1,1]*Cov_obs[0,0]-Cov_obs[1,0]*Cov_obs[0,1]
        det_the = Cov_the[1,1]*Cov_the[0,0]-Cov_the[1,0]*Cov_the[0,1]
        det_mix = np.zeros(self.l_max-self.l_min+1)

        for i in range(2):
            Cov_mix = np.copy(Cov_the)
            Cov_mix[i] = Cov_obs[i]
            det_mix += Cov_mix[1,1]*Cov_mix[0,0]-Cov_mix[1,0]*Cov_mix[0,1]

        chi2 = np.sum((2.*l+1.)*self.f_sky *\
            (det_mix/det_the + np.log(det_the/det_obs) - 2))

        return -chi2/2

    def create_fid_values(self, cl, params, override=False):
        # Write fiducial model spectra if needed
        # params should be the dictionary of cosmological parameters
        # corresponding to the cl provided
        if not self.fid_values_exist or override:
            # Store the values now.
            fid_file = open(os.path.join(
                self.data_directory, self.fiducial_file), 'w')
            fid_file.write('# Fiducial parameters')
            for key, value in params:
                fid_file.write(', %s = %.5g' % (key, value))
            fid_file.write('\n')
            for l in range(self.l_min, self.l_max+1):
                self.Cl_fid[0, l] = cl['tt'][l]+self.noise_T[l]
                self.Cl_fid[1, l] = cl['ee'][l]+self.noise_P[l]
                self.Cl_fid[2, l] = cl['te'][l]
                fid_file.write("%5d  " % l)
                fid_file.write("%.8g  " % self.Cl_fid[0, l])
                fid_file.write("%.8g  " % self.Cl_fid[1, l])
                fid_file.write("%.8g  " % self.Cl_fid[2, l])
                fid_file.write("\n")
            # print('\n\n')
            self.fid_values_exist = True
            fid_file.close()
            self.log.warning(
                "Writing fiducial model in %s, for %s likelihood" % (
                    self.data_directory+self.fiducial_file,
                    type(self).__name__))
            return True
        self.log.warning(
            "Fiducial model in %s for %s likelihood already exists" % (
                self.data_directory+self.fiducial_file,
                type(self).__name__))
        return False
