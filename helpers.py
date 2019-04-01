import cv2

def preprocessImg(img, brightnessFactor):
		
	hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	# change brightness
	hsv_image[:,:,2] = hsv_image[:,:,2] * brightnessFactor  
	# change back to RGB
	image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
	#convert to gray scale
	image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
	#return the image with dashboard cropped out
    
	return image_gray[:350, :]



