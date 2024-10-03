from corv import utils
from corv import readspectra as rs
import matplotlib.pyplot as plt
import re
import numpy as np

spec = rs.Spectrum('1d_da_nlte')
wav_1d = spec.wavl
interp_1d = spec.model_spec

wav_3d, interp_3d, a, b = utils.build_warwick_da()

params = (10000, 8)
mask_1d = (3600 < wav_1d) * (wav_1d < 9000)
mask_3d = (3600 < wav_3d) * (wav_3d < 9000)

plt.plot(wav_1d[mask_1d], interp_1d(params)[mask_1d])
plt.plot(wav_3d[mask_3d], interp_3d(params)[mask_3d])
plt.show()
