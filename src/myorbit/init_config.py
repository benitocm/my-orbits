"""Module to handle the configuration.
"""
from myorbit.util.general import mu_Sun

from pathlib import Path
CONFIG_INI=Path(__file__).resolve().parents[2].joinpath('conf','config.ini')
from configparser import ConfigParser
cfg = ConfigParser()
cfg.read(CONFIG_INI)

LAGUERRE_ABS_TOL = float(cfg.get('general','laguerre_abs_tol'))
NEAR_PARABOLIC_ABS_TOL = float(cfg.get('general','near_parabollic_abs_tol'))
H_ABS_TOL = float(cfg.get('general','angular_momentum_abs_tol'))
STUMPFF_ABS_TOL = float(cfg.get('general','stumpff_abs_tol'))
STUMPFF_METHOD = 0 if cfg.get('general','stumpff_method') == 'as_series' else 1
ENKES_ABS_TOL = float(cfg.get('general','enkes_abs_tol'))
H_ABS_TOL = float(cfg.get('general','angular_momentum_abs_tol'))
E_ABS_TOL= float(cfg.get('general','eccentricity_abs_tol'))
VSOP87_DATA_DIR=Path(cfg.get('general','vso87_data_dir_path'))

