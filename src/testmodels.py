from corv_test import utils
from corv_test import readspectra as rs
import matplotlib.pyplot as plt
import re
import numpy as np

spec = rs.Spectrum('3d_da_lte_old')
wav_1d = spec.wavl
interp_1d = spec.model_spec

wav_3d, interp_3d, a, b = utils.build_warwick_da()

params = (10000, 8)
mask_1d = (6550 < wav_1d) * (wav_1d < 6570)
mask_3d = (6550 < wav_3d) * (wav_3d < 6570)

plt.plot(wav_1d[mask_1d], interp_1d(params)[mask_1d], lw = 4, alpha = 0.5)
plt.plot(wav_3d[mask_3d], interp_3d(params)[mask_3d], lw = 4, alpha = 0.5, ls = '--')
plt.show()
