# cnn_sfrc
Scripts to extract fibers and voids from short fiber reinforced polymers V2

REQUIREMENTS:

-Python 2.7  
-GPU memory 10 GBs  
-Anaconda Command:

SET UP:
First time:
	-download file fiber_env.yml
	-run command: conda env create -f fiber_env.yml

To run:
	-modify "data_path" variable in script main.py  
	-run shell commands:
		>> conda activate fiber_env
		>> python main.py


INPUT:
	-Directory with path containing 2560x2560x1350 UINT16 images (these numbers are hard corded)

OUTPUT:

-Folder:
	output_files:

		H5 FILES:
		volume_segmentation.h5: 1025x1025x504 numpy uint16. Matrix=0, fibers=1 and voids=2
		volume_fiber_voids.h5: 1025x1025x504 numpy uint16. Matrix=0, voids=1, fiber=2...N (each number is a different fiber. Starting at 2...N)
		data_volume.h5: 1025x1025x504 numpy uint16. Reference volume of original sample scaled

		XMF FILES:
		volume_segmentation.xmf: file to vizualise volume_segmentation.h5
		volume_fiber_voids.xmf: file to vizualise volume_fiber_voids.h5
		data_volume.xmf: file to vizualise data_volume.h5

		TXT FILES:
		fiber_dictionary.txt: fiber dictionary containing:
		<< fiber_number, center[0], center[1], center[2], radious, length, direction_vector[0], diretion_vector[1], diretion_vector[2] >>

