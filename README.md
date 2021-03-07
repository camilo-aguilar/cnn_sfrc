# Paper
C.Aguilar , M. Comer, I. Hanhan, R. Agyei, M. Sangid. “Void Detection and fiber extraction for statistical characterization of fiber-reinforced polymers”. IS&T/SPIE Electronic Imaging, January 2020, Burlingame, CA. (https://par.nsf.gov/biblio/10179419)
# cnn_sfrc
Scripts to extract fibers and voids from short fiber reinforced polymers V2

REQUIREMENTS:

	-Python 2.7  
	-GPU memory > 12 GBs  
	-RAM > 20 GBs


SET UP:


	First time:
	-run command: conda env create -f fiber_env.yml
	-download folder "info_files", unzip it and copy it in the same directory as main.py
	info files: https://drive.google.com/open?id=1WYCYYjV8cE48-4WIflFkPIFwQr-pg-MJ
	To run:
	-modify "data_path" variable in script main.py  
	-run shell commands:
		>> conda activate fiber_env
		>> python main.py


INPUT:

	Directory with path containing 2560x2560x1350 UINT16 images (these numbers are hard corded)

OUTPUT:

	H5 FILES (+ XMF FILES):
		volume_segmentation.h5: 1025x1025x501 numpy uint16. Matrix=0, fibers=1 and voids=2
		volume_fiber_voids.h5: 1025x1025x501 numpy uint16. Matrix=0, voids=1, fiber=2...N (each number is a different fiber. Starting at 2...N)
		volume_fiber_voids_labeled_voids.h5: 1025x1025x501 numpy uint16. Matrix=0, fiber=2,3,4,...<999999.  voids=1000001,1000002...N (Each void is given an offset of +1E6)

		volume_fiber_voids_labeled_voids_full_resolution.h5: 2050x2050x1002: Upsample version of volume_fiber_voids_labeled_voids.

	TXT FILES:
		fiber_dictionary.txt: fiber dictionary for volume_fiber_voids.h5 containing:
		<< fiber_number, center[0], center[1], center[2], radious, length, direction_vector[0], diretion_vector[1], diretion_vector[2] >>
		
		void_dictionary.txt: void dictionary for volume_fiber_voids_labeled_voids.h5 containing:
		<< void_number, center[0], center[1], center[2], radious, volume, direction_vector[0], diretion_vector[1], diretion_vector[2] >> (voids are fitted as cylinder point clouds)

