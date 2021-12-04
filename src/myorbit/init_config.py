"""Module to handle the configuration.
"""

# Standard library imports
from pathlib import Path

# Third party imports

CONFIG_INI=Path(__file__).resolve().parents[2].joinpath('conf','config.ini')
from configparser import ConfigParser
cfg = ConfigParser()
cfg.read(CONFIG_INI)

LAGUERRE_ABS_TOL = float(cfg.get('general','laguerre_abs_tol'))
NEAR_PARABOLIC_ABS_TOL = float(cfg.get('general','near_parabollic_abs_tol'))
STUMPFF_ABS_TOL = float(cfg.get('general','stumpff_abs_tol'))
STUMPFF_METHOD = 0 if cfg.get('general','stumpff_method') == 'as_series' else 1
ENKES_ABS_TOL = float(cfg.get('general','enkes_abs_tol'))
H_ABS_TOL = float(cfg.get('general','angular_momentum_abs_tol'))
EC_ABS_TOL= float(cfg.get('general','eccentricity_abs_tol'))
VELOCITY_ABS_TOL= float(cfg.get('general','velocity_abs_tol'))
CONSIDER_PARABOLIC_TOL=float(cfg.get('general','to_consider_parabolic_tol'))
VSOP87_DATA_DIR=Path(cfg.get('general','vso87_data_dir_path'))


