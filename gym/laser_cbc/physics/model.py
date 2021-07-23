import argparse
import os

import numpy as np
from scipy import signal
from scipy.stats import norm

np.set_printoptions(precision=0, suppress=True, linewidth=150)


class CBCModel:
    """
    filled aperture diffractive Coherent Beam Combining model, first order only

    M:                  square array of DOE shape MxM
    beam_en_mask:       enabled beams mask as boolean array of shape MxM
    b(x,y):             beam (complex ndarray(M,M))
    d(x,y):             diffractional optical element(doe) (complex ndarray(M,M))
    s(x,y):             superposition diffraction pattern (complex ndarray(2M-1,2M-1))
    b_f(u,v):           fft2d(b) in spatial-frequency domain (complex ndarray(M,M))
    d_f(u,v):           fft2d(d) in spatial-frequency domain (complex ndarray(M,M))
    s_f(u,v):           fft2d(s) in spatial-frequency domain (complex ndarray(2M-1,2M-1))

    rms_measure_noise:  RMS measurement noise on pattern, at unit of one input beam power
    phs_drift_step_deg: Environmental phase drift scale per step, in degrees
    """

    def __init__(self, **kwargs):
        self.M = M = kwargs.get("M", 9)
        self.test_3_in_9 = kwargs.get("test_3_in_9", False)
        self.beam_shape = beam_shape = (M, M)
        self.pattern_shape = pattern_shape = (2 * M - 1, 2 * M - 1)

        self.beam_en_mask = kwargs.get(
            "beam_en_mask", np.ones(beam_shape, dtype=bool))
        self.pattern_en_mask = kwargs.get(
            "pattern_en_mask", np.ones(pattern_shape, dtype=bool))
        self.rms_measure_noise = kwargs.get("rms_measure_noise", 0.1)
        self.phs_drift_step = np.deg2rad(kwargs.get("phs_drift_step_deg", 0))

        if M == 9:
            fname = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "doe81.npy")
            doe_full = kwargs.get("doe81", np.load(fname))
            # default is 15x15, use first order 9x9 response only
            self.doe = doe = doe_full[3:-3, 3:-3]
            # scale total transmission function to be lossless
            doe /= np.sqrt((doe * doe.conj()).sum())
            if self.test_3_in_9:
                self.beam_en_mask = np.zeros(beam_shape, dtype=bool)
                self.beam_en_mask[3:-3, 3:-3] = True
                self.pattern_en_mask = np.zeros(pattern_shape, dtype=bool)
                self.pattern_en_mask[3:-3, 3:-3] = True
        elif M == 3:
            doe_phs = np.deg2rad(
                np.array([0, 90, 0, 0, 0, -90, 180, 0, 0]).reshape(3, 3))
            doe_amp = np.ones_like(doe_phs) / np.sqrt(8)
            doe_amp[1, 1] = 0
            self.doe = doe = doe_amp * np.exp(1j * doe_phs)
            self.beam_en_mask[1, 1] = False
        else:
            raise ValueError("Invalid DOE shape")
        # assume equal splitting, calculate power splitting factor
        self.Dn = np.sqrt(np.count_nonzero(doe))
        self.n_beams = np.count_nonzero(self.beam_en_mask)
        # self.max_pwr = self.Dn ** 4
        self.max_pwr = self.n_beams**2
        self.n_pattern_beams = np.count_nonzero(self.pattern_en_mask)
        self.reset()

    @property
    def beam_amp(self):
        return self._beam_amp

    @beam_amp.setter
    def beam_amp(self, value):
        self._beam_amp = value

    @property
    def beam_phs(self):
        return self._beam_phs

    @beam_phs.setter
    def beam_phs(self, value):
        self._beam_phs = self.wrap_phase(value)

    @property
    def pattern(self):
        if self._pattern is None:
            self.propagate()
        noise = np.random.randn(*self.pattern_shape) * self.rms_measure_noise
        return np.clip(self._pattern + noise, 0, self.max_pwr)

    @property
    def nonzero_pattern(self):
        return self.pattern[self.pattern_en_mask]

    @property
    def combined_power(self):
        h, w = self.pattern.shape
        return self.pattern[h // 2, w // 2]

    @property
    def efficiency(self):
        return self.combined_power / np.power(self.beam_amp, 2).sum() * 100

    @property
    def norm_eta(self):
        return self.efficiency / self._eta_ref

    def __repr__(self):
        return np.array2string(self.pattern, precision=0, max_line_width=150)

    def propagate(self):
        self.beam_phs += norm.rvs(scale=self.phs_drift_step,
                                  size=self.beam_shape)
        self._pattern = self.sim(self.beam_amp, self.beam_phs)
        return self.nonzero_pattern

    def sim(self, beam_amp, beam_phs):
        s = self.conv2d(beam_amp, beam_phs)
        return np.abs(s * s.conj())

    def conv2d(self, beam_amp, beam_phs):
        """ s = b * d """
        b = beam_amp * np.exp(1j * beam_phs)
        return signal.convolve2d(b, self.doe)

    def deconv2d(self, s):
        """ B = S / D """
        M2 = self.beam_shape[0] // 2
        doe_pad = np.pad(self.doe, ((M2, M2), (M2, M2)), "constant")
        d_f = np.fft.fft2(doe_pad)

        s_f = np.fft.fft2(s)
        kernel = np.fft.ifft2(s_f / d_f)
        kernel = np.fft.fftshift(kernel)
        off = self.doe.shape[0] // 2
        return kernel[off:-off, off:-off]

    def verify(self, beam_amp):
        # random input phase, forward simulation
        beam_phs = np.random.rand(*self.beam_shape) * np.pi * 2
        s = self.conv2d(beam_amp, beam_phs)
        # find b from s using d
        b = self.deconv2d(s)
        # compare with known beam array
        b_want = beam_amp * np.exp(1j * beam_phs)
        return np.allclose(b, b_want)

    def wrap_phase(self, phases):
        return (phases + np.pi) % (2 * np.pi) - np.pi

    def reset(self):
        """ Reset system to ideal state, measure ideal pattern and combining efficiency """
        self.beam_amp = np.rot90(np.rot90(np.abs(self.doe))) * self.Dn**2 * self.beam_en_mask
        self.beam_phs = -np.rot90(np.rot90(np.angle(self.doe)))
        self.beam_phs_ideal = self.beam_phs.copy()
        self.pattern_ideal = self.propagate()
        self._eta_ref = self.efficiency

    def perturb_random(self, phs_deg=10, np_random=None):
        """
        Randomly perturb system in beam phases
        phs_deg:    uniform distributed random phase perturb range in deg
        return:     nonzero pattern after perturb, perturbed phase array (flatten)
        """
        if np_random is None:
            np_random = np.random.RandomState()
        phs_perturb_deg = np_random.uniform(low=-phs_deg, high=phs_deg, size=self.beam_shape)
        self.beam_phs += np.deg2rad(phs_perturb_deg)
        self.propagate()
        return self.nonzero_pattern, phs_perturb_deg[self.beam_en_mask]

    def perturb_phase_arr(self, phs_perturb_deg):
        """
        phs_perturb_deg:    phase perturbation array in deg with only enabled beams
        return:             nonzero pattern after perturb
        """
        self.beam_phs[self.beam_en_mask] += np.deg2rad(phs_perturb_deg)
        return self.propagate()


class Controller:
    def __init__(self, **kwargs):
        self.model = CBCModel(**kwargs)
        self._gain = kwargs.get("gain", 1)
        self.reset_drive()

    def reset_drive(self):
        self._u = np.zeros(self.model.n_beams)

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, value):
        assert value.size == self.model.n_beams
        self._u = value

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self._gain = value

    def correct(self, phs_err):
        u_delta = self.gain * phs_err
        self.u -= u_delta
        self.model.beam_phs[self.model.beam_en_mask] -= u_delta
        self.model.propagate()

    def iterate(self):
        self.correct(0)

    def diagnose(self):
        return None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-m", type=int, default=3, choices=[3, 9], help="MxM beam shape")
    p.add_argument("--test_3_in_9", action="store_true")
    p.add_argument("--rms_measure_noise", type=float, default=0.1, help="rms camera noise")
    p.add_argument("--phs_drift_step_deg", type=float, default=5, help="phs_drift_step_deg")

    args = p.parse_args()
    config = {
        'M': args.m,
        'test_3_in_9': args.test_3_in_9,
        'rms_measure_noise': args.rms_measure_noise,
        'phs_drift_step_deg': args.phs_drift_step_deg
    }
    model = CBCModel(**config)

    print("Asserting de-convolution...")
    beam_amp = np.random.rand(model.M, model.M)
    assert model.verify(beam_amp) is True

    np.set_printoptions(precision=1, suppress=True, linewidth=150)
    print(f"DOE amp response * {model.M}: (expect 1)")
    print(np.abs(model.doe) * model.M)
    print("Input power:")
    print(model.beam_amp ** 2)
    print(f"Total Input power: {np.sum(model.beam_amp ** 2):.1f}")
    print("Ouput power:")
    print(model)
    print(f"Combined center: {model.combined_power:.1f}")
    print(f"Efficiency: {model.efficiency:.3f} %")
    # print(model)

    print(f"Natural phase drifting...RMS {args.phs_drift_step_deg:d} deg per step")
    for i in range(10):
        model.propagate()
        print(f"[{i:3d}] Normalized Efficiency: {100 * model.norm_eta:.1f} %")

    rms_deg = 10
    print(f"Applying perturbation...RMS {rms_deg:d} deg per step")
    for i in range(10):
        model.perturb_random(rms_deg)
        print(f"[{i:3d}] Normalized Efficiency: {100 * model.norm_eta:.1f} %")
