# cnn_sfrc
Scripts to extract fibers and voids from short fiber reinforced polymers V2

REQUIREMENTS:

	-Python 2.7  
	-GPU memory > 6 GBs  
	-RAM ~ 30 GBs


SET UP:


	First time:
	-download file fiber_env.yml
	-run command: conda env create -f fiber_env.yml
	-download info_files and copy it in the same directory as main.py
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
		volume_segmentation.h5: 2050x2050x1008 numpy uint16. Matrix=0, fibers=1 and voids=2


		volume_fiber_voids.h5: 2050x2050x1008 numpy uint16. Matrix=0, voids=1, fiber=2...N (each number is a different fiber. Starting at 2...N)



	TXT FILES:
		fiber_dictionary.txt: fiber dictionary containing:
		<< fiber_number, center[0], center[1], center[2], radious, length, direction_vector[0], diretion_vector[1], diretion_vector[2] >>

