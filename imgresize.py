from PIL import Image
import os, sys

path = "./not_rain2/"
dirs = os.listdir( path )

f = 0
name = './rain_data/not_rain/no'
minsize = 900


for item in dirs:
	if os.path.isfile(path+item) and '.jpg' in item:
		try:
			im = Image.open(path+item)

			width = im.size[0]
			height = im.size[1]
			print(im.size)
			if width > height:
				size = (int(width/height * minsize), minsize)
				print(size)
				imResize = im.resize(size, Image.ANTIALIAS)
				imResize.save(name + str(f) + '.jpg', 'JPEG', quality=100)
			else:
				size = (minsize, int(height/width * minsize))
				print(size)
				imResize = im.resize(size, Image.ANTIALIAS)
				imResize.save(name + str(f) + '.jpg', 'JPEG', quality=100)
		except:
			print("oopsies")

		f += 1



		
