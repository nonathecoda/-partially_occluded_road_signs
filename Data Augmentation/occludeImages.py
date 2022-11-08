from PIL import Image
import numpy as np
import matplotlib as plt
from os import walk

occlusion_level = 0.1

filenames = next(walk("/Users/antonia/Documents/EIT/KTH/RMSW/Mapillary/croppedImages"), (None, None, []))[2]  # [] if no file


for filename in filenames:
	# Create an image as input:
	input_image = Image.open('/Users/antonia/Documents/EIT/KTH/RMSW/Mapillary/croppedImages/'+filename)
	pixel_map = input_image.load()
	width, height = input_image.size
	num_pixels = width * height

	print(width, height)

	total_occ = int(num_pixels * occlusion_level)


	w_l, h_l = 0, 0 
	while not ((w_l * h_l <= total_occ+total_occ*0.5) and (w_l * h_l >= total_occ-total_occ*0.5)):
		w_l = np.random.randint(0, width-1)
		h_l = np.random.randint(0, height-1)
	    
	w_0 = np.random.randint(0, width-w_l)
	h_0 = np.random.randint(0, height-h_l)

	#plt.imshow(img[w_0:w_0+w_l, h_0:h_0+h_l])
	#plt.show()

	for i in range(w_0, w_0+w_l):
		for j in range(h_0, h_0+h_l):
			pixel_map[i,j] = (0, 0, 0)


	savePath = "/Users/antonia/Documents/EIT/KTH/RMSW/Mapillary/occludedPictures2/"+filename
	print("I saved something")
	input_image.save(savePath, format="png")