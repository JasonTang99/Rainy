from PIL import Image
import os, sys

path = "./rain"
dirs = os.listdir( path )

i = 0

for item in dirs:
	print(item)
	if os.path.isfile(path+item):
		im = Image.open(path+item)
		f, _ = os.path.splitext(path+item)
		print(im.size)
		i += 1
		# imResize = im.resize((200,200), Image.ANTIALIAS)
		# imResize.save(f + ' resized.jpg', 'JPEG', quality=100)

print(i)
