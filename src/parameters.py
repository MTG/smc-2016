# A file with parameters for tabla stroke analysis
import numpy as np
import math
# Frame parameters first
Fs = 44100.0
hop = int(np.round(11.6e-3*Fs))
frmSize = int(np.round(23.23e-3*Fs))
Nfft = int(np.power(2, np.ceil(np.log2(frmSize))))
zeropadLen = Nfft - frmSize
speedUpFactor = 2
selectInd = np.arange(1,13)

# Probability parameters
strokeDoublingP = 0.4
rollP = 0.4