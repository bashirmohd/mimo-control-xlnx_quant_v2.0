import argparse
import json
import logging

import numpy as np
import torch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gym'))
from laser_cbc.physics.model import Controller

np.set_printoptions(precision=0, suppress=True, linewidth=150)
logger = logging.getLogger(__name__)


class CombineNN(Controller):
    ''' 9x9 combining combine using NN '''

    def __init__(self, **kwargs):
        super(CombineNN, self).__init__(**kwargs)
        self.M = kwargs.get('M', 9)
        self.double_frame = kwargs.get('double_frame', False)
        self.load_nn_model(**kwargs)

    def load_nn_model(self, **kwargs):
        nn_weight = kwargs.get(
            'nn_weight', 'nn_trained/{}by{}_90deg.pth'.format(self.M, self.M))
        nn_config = kwargs.get(
            'nn_config', 'nn_trained/{}by{}_90deg.json'.format(self.M, self.M))
        # self.net.load_state_dict(torch.load(nn_weight, map_location=torch.device('cpu')))
        self.net = torch.load(nn_weight, map_location=torch.device('cpu'))
        print(self.net)

        with open(nn_config, 'r') as f:
            nn_param = json.load(f)
        self.mu_X = nn_param['mu_X']
        self.sigma_X = nn_param['sigma_X']
        self.mu_Y = nn_param['mu_Y']
        self.sigma_Y = nn_param['sigma_Y']

    def predict_phs(self, s_power):
        ''' return predicted phase array in radians '''
        s_power_norm = (s_power - self.mu_X) / self.sigma_X
        y_norm = self.net(torch.from_numpy(s_power_norm).float()).detach()
        y = y_norm.numpy() * self.sigma_Y + self.mu_Y
        return self.model.wrap_phase(np.deg2rad(y))

    def diagnose(self):
        ph_err = (self.model.beam_phs_ideal - self.model.beam_phs)[self.model.beam_en_mask]
        ph_err -= ph_err[0]  # relative to first beam
        ph_err_deg = np.rad2deg(self.model.wrap_phase(ph_err))
        logger.debug('Phase err to ideal[deg]:\n{}'.format(ph_err_deg))
        logger.debug('RMS prediction error: {:.1f} deg'.format(ph_err_deg.std()))
        logger.debug('Pattern:\n{}'.format(self.model))
        return ph_err_deg

    def iterate(self):
        if self.double_frame:
            pattern = np.concatenate((self.model.pattern_ideal,
                                      self.model.nonzero_pattern))
        else:
            pattern = self.model.nonzero_pattern
        self.phs_hat = self.predict_phs(pattern)
        self.correct(self.phs_hat)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('-g', '--gain', default=1.0, type=float)
    p.add_argument('-n', '--n_steps', default=50, type=int)
    p.add_argument('-m', type=int, default=9, choices=[3, 9], help='MxM beam shape')
    p.add_argument('--test_3_in_9', action='store_true')
    p.add_argument('--weight', help='NN weight file')
    p.add_argument('--double_frame', action='store_true', help='doubled frame training')
    p.add_argument("--rms_measure_noise", type=float, default=0.1, help="rms camera noise")
    p.add_argument("--phs_drift_step_deg", type=float, default=5, help="phs_drift_step_deg")
    args = p.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.weight is None:
        fname = 'nn_trained/{}by{}_90deg.pth'.format(args.m, args.m)
    else:
        fname = args.weight

    print('Loading nn_weight ' + fname + '...')
    config = {
        'gain': args.gain,
        'M': args.m,
        'nn_weight': fname,
        'nn_config': fname.replace('pth', 'json'),
        'test_3_in_9': args.test_3_in_9,
        'double_frame': args.double_frame,
        'rms_measure_noise': args.rms_measure_noise,
        'phs_drift_step_deg': args.phs_drift_step_deg
    }
    combine = CombineNN(**config)

    # start from random state
    combine.model.perturb_random(180)

    n_skip = 1 if args.verbose else 5
    for i in range(args.n_steps):
        if i % n_skip == 0:
            print(f'''===== Step: {i:2d}: Norm Efficiency: {100 *
            combine.model.norm_eta:5.1f} % =====''')
        combine.iterate()
        combine.diagnose()
