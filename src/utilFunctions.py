import numpy as np
from scipy.signal import resample, blackmanharris, triang
from scipy.fftpack import fft, ifft, fftshift
import math, copy, sys, os
from scipy.io.wavfile import write, read
from sys import platform
import subprocess

def isPower2(num):
	"""
	Check if num is power of two
	"""
	return ((num & (num - 1)) == 0) and num > 0

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

def wavread(filename):
	"""
	Read a sound file and convert it to a normalized floating point array
	filename: name of file to read
	returns fs: samplint rate of file, x: floating point array
	"""

	if (os.path.isfile(filename) == False):                  # raise error if wrong input file
		raise ValueError("Input file is wrong")

	fs, x = read(filename)

	if (len(x.shape) !=1):                                   # raise error if more than one channel
                x = np.mean(x,axis = 1)
		print "Audio file is stereo, converting to mono"

	#scale down and convert audio into floating point number in range of -1 to 1
	x = np.float32(x)/norm_fact[x.dtype.name]
	return fs, x

def wavplay(filename):
	"""
	Play a wav audio file from system using OS calls
	filename: name of file to read
	"""
	if (os.path.isfile(filename) == False):                  # raise error if wrong input file
		print("Input file does not exist. Make sure you computed the analysis/synthesis")
	else:
		if sys.platform == "linux" or sys.platform == "linux2":
		    # linux
		    subprocess.call(["aplay", filename])

		elif sys.platform == "darwin":
			# OS X
			subprocess.call(["afplay", filename])
		else:
			print("Platform not recognized")

def wavwrite(y, fs, filename):
	"""
	Write a sound file from an array with the sound and the sampling rate
	y: floating point array of one dimension, fs: sampling rate
	filename: name of file to create
	"""

	x = copy.deepcopy(y)                         # copy array
	x *= INT16_FAC                               # scaling floating point -1 to 1 range signal to int16 range
	x = np.int16(x)                              # converting to int16 type
	write(filename, fs, x)
