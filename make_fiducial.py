#!/usr/bin/env python
# Example script to create fiducial values for mock CMB likelihoods

fiducial_params = {
    # LambdaCDM parameters
    'H0':67.37,
    'omega_b':0.02237,
    'N_ur':2.037,
    'omega_cdm':0.1200,
    'N_ncdm':1,
    'omega_ncdm':0.0006451439,
    'm_ncdm':0.06,
    'A_s':2.108e-9,
    #'sigma8':0.8113,
    'n_s':0.9619,
    'tau_reio':0.0546,
    # Take fixed value for primordial Helium (instead of automatic BBN adjustment)
    'YHe':0.2454006
}

fiducial_params_extra = {
    'recombination':'recfast',
    'non linear': 'halofit'
}

fiducial_params_full = fiducial_params.copy()
fiducial_params_full.update(fiducial_params_extra)

l_max = 5000

info_fiducial = {
    'params': fiducial_params,
    'likelihood': {'mock_SO.MockSO': {'python_path': '.'},
                   'mock_CMBS4.MockCMBS4': {'python_path': '.'}},
    'theory': {'classy': {"extra_args": fiducial_params_extra}}}

from cobaya.model import get_model
model_fiducial = get_model(info_fiducial)

model_fiducial.logposterior({})

Cls = model_fiducial.provider.get_Cl(ell_factor=True, units="muK2")

from mock_SO import MockSO
MockSO().create_fid_values(Cls, fiducial_params_full, override=True)

from mock_CMBS4 import MockCMBS4
MockCMBS4().create_fid_values(Cls, fiducial_params_full, override=True)