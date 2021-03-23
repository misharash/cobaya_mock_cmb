#!/usr/bin/env python3
# Example script to create fiducial values for mock CMB likelihoods
# with non-LCDM fiducial, namely with small-scale baryon clumping
from cobaya.model import get_model
from cobaya_mock_cmb import MockSOClumping, MockCMBS4Clumping

#from best fit to Planck+SH0ES with fixed massless neutrinos and nuisance-marginalized high-l
fiducial_params = {
    # LambdaCDM parameters
    'H0': 7.077805000e+01,
    # '100*theta_s': 1.042042099e+00,
    'omega_b': 2.276093940e-02,
    'N_ur': 3.046, #three massless neutrinos
    'omega_cdm': 1.181255826e-01,
    'A_s': 2.176893349e-09,
    # 'sigma8': 8.384906370e-01,
    'n_s': 9.715475094e-01,
    'tau_reio': 7.369021942e-02,
    # clumping space
    'delta_m': -9.549925860e-01,
    'delta_p': 1.574459983e+00,
    'f2V': 7.852602218e-01
}

fiducial_params_extra = {
    'recombination': 'recfast_3zones_lowlevel',
    'non linear': 'halofit'
}

fiducial_params_full = fiducial_params.copy()
fiducial_params_full.update(fiducial_params_extra)

info_fiducial = {
    'params': fiducial_params,
    'likelihood': {'cobaya_mock_cmb.MockSOClumping': {'python_path': '.'},
                   'cobaya_mock_cmb.MockCMBS4Clumping': {'python_path': '.'}},
    'theory': {'classy': {"extra_args": fiducial_params_extra}}}

model_fiducial = get_model(info_fiducial)

model_fiducial.logposterior({})

Cls = model_fiducial.provider.get_Cl(units="muK2")

MockSOClumping().create_fid_values(Cls, fiducial_params_full, override=True)

MockCMBS4Clumping().create_fid_values(Cls, fiducial_params_full, override=True)
