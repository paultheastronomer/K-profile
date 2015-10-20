# This software is open.
# Please give credit if you use it:
# Paul A. Wilson

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import special

from matplotlib.ticker import LinearLocator, FormatStrFormatter

def read_params():
	with open('params/reduction.json') as param_file:    
		param = json.load(param_file)
	return param

def voigt(x, y):
	# The Voigt function is also the real part of 
	# w(z) = exp(-z^2) erfc(iz), the complex probability function,
	# which is also known as the Faddeeva function. Scipy has 
	# implemented this function under the name wofz()
	z = x + 1j*y
	I = special.wofz(z).real
	return I

def H(nu_,nu,c,k,T,mu_,gamma):
	v_=nu_/c*np.sqrt(2*k*T/mu_) 	# Doppler
	u=(nu-nu_)/(v_) 	 	# Normalized Doppler displacement
	a=gamma/(4*np.pi*v_)	 	# Damping Constant
	H = voigt(u,a)
	return H,v_

def crossection(sigma_,H,v_):
	sigma_voigt=sigma_*H/(np.sqrt(np.pi)*v_)
	return sigma_voigt

def lecavelier(cross_sec,k,T,mu,g,abundance,P_ref,t_eq,Rp):  # Absorption as a function of wavelength
	H = (k*T)/(mu*g)
	print "Scale Height: \t\t",round(H/100000.,2)," km"
	return H*np.log(  (abundance*cross_sec*P_ref/t_eq)*np.sqrt(2.*np.pi*Rp/(k*T*mu*g))  )

def main():    
	param = read_params()	# Read parameter file reduction.json

	# Contants
	k		= param["constants"]["k"]	# gm*cm^2/s^2
	c		= param["constants"]["c"]	# cm/s
	me		= param["constants"]["me"]	# electron mass cgs
	e_charge= param["constants"]["e"]	# electron charge
	u 		= param["constants"]["u"]	# mean molecular weight
	mu 		= param["system parameters"]["mu"]*u  # mean molecular mass

	# Parameters
	R_sun	= param["constants"]["R_Sun"]	# Solar radius in [cm]
	Rs		= param["system parameters"]["R*/R_Sun"]*R_sun	# Star Radius in [cm] (6.95508e10 cm is R_Sun)
	Rp		= param["system parameters"]["Rp/R_Sun"]*Rs		# Planet Radius in [cm] (7.1492e9 cm is R_Jup)
	T		= param["system parameters"]["temp"]			# K Atmospheric Temperature
	g		= param["system parameters"]["g"]				# Gravity in cgs
	P_ref	= param["system parameters"]["P_ref"]			# Reference Pressure (Base of atmosphere) P_0 [Barye]
	t_eq 	= param["system parameters"]["t_eq"]			# Optical depth

	# H2
	mu_H2	= u*2 	# g H2 mass

	# K
	mu_K	= u*39 	# g Na mass
	gamma_K_nu1 = param["K"]["gamma_nu1"] 	# hz Decay Rate Na
	gamma_K_nu2 = param["K"]["gamma_nu1"] 	# hz Decay Rate Na
	fd1_k		= param["K"]["fd1"] 		# absorption oscillator strength  - Handbook chemistry and physics
	fd2_k 		= param["K"]["fd2"] 		# absorption oscillator strength
	abundK		= param["K"]["abund"]	# Potassium abundance

	# Wavelength & Frequency Array
	w_aa = np.arange(7682-500,7682+500,0.01) # Wavelength in Angstrom
	w_cm = w_aa*1e-8  			 # Wavelength in cm
	nu=c/w_cm 				 # Frequency

	S=np.pi*e_charge**2 /(me*c)

	sigma_k_1=S*fd1_k
	sigma_k_2=S*fd2_k

	nu_2K=c/param["K"]["K2"]
	nu_1K=c/param["K"]["K1"]

	H_K_1,v_K_d1 = H(nu_1K,nu,c,k,T,mu_K,gamma_K_nu1)
	H_K_2,v_K_d2 = H(nu_2K,nu,c,k,T,mu_K,gamma_K_nu2)

	sigma_K_voigt1 = crossection(sigma_k_1,H_K_1,v_K_d1)
	sigma_K_voigt2 = crossection(sigma_k_2,H_K_2,v_K_d2)

	sigma_K_voigt=sigma_K_voigt1+sigma_K_voigt2 # Cross Section of Doublet

	cross_sec_K = sigma_K_voigt

	unbinned_high = lecavelier(cross_sec_K,k,2000.,mu,g,abundK,P_ref,t_eq,Rp)
	unbinned_med = lecavelier(cross_sec_K,k,1500.,mu,g,abundK,P_ref,t_eq,Rp)
	unbinned_low = lecavelier(cross_sec_K,k,1000.,mu,g,abundK,P_ref,t_eq,Rp)

	# Plot the data
	fig = plt.figure(figsize=(7,8.5/1.5))

	fontlabel_size = 19
	tick_size = 19
	params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': fontlabel_size, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
	plt.rcParams.update(params)
	plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	ax=fig.add_subplot(1,1,1)

	# Profiles with various temperatures
	plt.plot(w_aa,(unbinned_high+Rp)/Rs,'-',color='red',linewidth=2.)
	plt.plot(w_aa,(unbinned_med+Rp)/Rs,'-',color='#FFBC40',linewidth=2.)
	plt.plot(w_aa,(unbinned_low+Rp)/Rs,'-',color='#375FA9',linewidth=2.)

	ax.legend(('2000~K','1500~K','1000~K'),loc='upper left', numpoints = 1)

	plt.minorticks_on()
	plt.ylim(0.113,0.145)
	plt.xlim(7681-400,7681+400)
	plt.xlabel('Wavelength [\AA]')
	plt.ylabel('$R_p / R_*$')
	plt.savefig('K_profile_temp.pdf',transparent=True, bbox_inches='tight', pad_inches=0.1)
	plt.show()
	
if __name__ == '__main__':
    main()