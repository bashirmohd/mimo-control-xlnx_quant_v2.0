import argparse
import logging

import numpy as np
from scipy.stats import special_ortho_group

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gym'))
from laser_cbc.physics.model import Controller

np.set_printoptions(precision=0, suppress=True, linewidth=150)
logger = logging.getLogger(__name__)


class SPGD(Controller):
    def __init__(self, **kwargs):
        super(SPGD, self).__init__(**kwargs)
        self._dither_rms_deg = kwargs.get('dither_rms_deg', 3)
        self.ortho_sampling = kwargs.get('ortho_sampling', False)
        self.n_steps = 0
        self.rng = np.random.default_rng()

    @property
    def dither_u(self):
        return self._dither_u

    @dither_u.setter
    def dither_u(self, value):
        assert value.size == self.model.n_beams
        self._dither_u = value

    @property
    def dither_rms_deg(self):
        return self._dither_rms_deg

    @dither_rms_deg.setter
    def dither_rms_deg(self, value):
        self._dither_rms_deg = value

    def reset_drive(self):
        self._u = np.zeros(self.model.n_beams)
        self._dither_u = np.zeros_like(self._u)

    def gen_dither_u(self):
        '''
        Orthonormal dither sets are arrays consisting of (N - 1) N-element
        dither vectors, where each dither vector is normalized to have a
        standard deviation of 1 and where all dither vectors are orthogonal
        to each other.
        Only N - 1 dither vectors are required for a N-element active phasing
        system as there are only N - 1 relative phase differences.
        TODO: implement N-1, and normalization
        '''
        if self.ortho_sampling:
            idx = self.n_steps % self.model.n_beams
            if idx == 0:
                self.ortho_samples = special_ortho_group.rvs(self.model.n_beams)
            self.dither_u = self.ortho_samples[idx] * np.deg2rad(self.dither_rms_deg)
        else:
            self.dither_u = (self.rng.random(self.model.n_beams) * 2 - 1) * np.deg2rad(self.dither_rms_deg)

    def dither(self, gain=1):
        self.u += gain * self.dither_u
        self.model.beam_phs[self.model.beam_en_mask] += gain * self.dither_u
        self.model.propagate()
        return self.model.combined_power

    def diagnose(self):
        logger.debug("{:20s}: {:4.1f} deg".format("dither_u", np.rad2deg(self.dither_u.std())))
        logger.debug("{:20s}: {:4.1f} deg".format("u", np.rad2deg(self.u.std())))
        ph_err = (self.model.beam_phs_ideal - self.model.beam_phs)[self.model.beam_en_mask]
        ph_err -= ph_err[0]  # relative to first beam
        ph_err_deg = np.rad2deg(self.model.wrap_phase(ph_err))
        logger.debug("{:20s}: {:4.1f} deg".format("Beam phase err", ph_err_deg.std()))
        return self.dither_u

    def iterate(self):
        self.gen_dither_u()
        J_p = self.dither(1)
        J_n = self.dither(-2)
        # logger.debug('Jp: {:.0f}, Jn: {:.0f}'.format(J_p, J_n))
        delta_J = J_p - J_n
        phs_err = -delta_J * self.dither_u / 33  # emperical 33
        self.dither(1)  # return to start point
        self.correct(phs_err)
        self.n_steps += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-n", "--n_steps", default=1200, type=int)
    p.add_argument("-g", "--gain", default=1.0, type=float)
    p.add_argument("-m", type=int, default=3, choices=[3, 9], help="MxM beam shape")
    p.add_argument('--ortho_sampling', action='store_true', help='use random orthogonal sampling')
    p.add_argument("--rms_measure_noise", type=float, default=0.1, help="rms camera noise")
    p.add_argument("--phs_drift_step_deg", type=float, default=5, help="environmental phase drift")
    p.add_argument("--dither_rms_deg", type=float, default=5, help="SPGD dither rms in deg")
    args = p.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    config = {
        'M': args.m,
        'gain': args.gain,
        'dither_rms_deg': args.dither_rms_deg,
        'ortho_sampling': args.ortho_sampling,
        'rms_measure_noise': args.rms_measure_noise,
        'phs_drift_step_deg': args.phs_drift_step_deg
    }

    spgd = SPGD(**config)

    print("Introducing perturbation...")
    spgd.model.perturb_random(90)
    # print('Efficiency: {:.1f} %'.format(spgd.model.efficiency))
    print(f"Combined power: {spgd.model.combined_power:.0f}")

    print("START SPGD...")
    for i in range(args.n_steps):
        spgd.iterate()
        if i % 20 == 0:
            spgd.diagnose()
            print(f"Step: {i:3d}: Norm Efficiency: {spgd.model.norm_eta * 100:.1f} %")
