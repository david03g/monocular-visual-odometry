import numpy as np
import cv2
from frame import Frame
import os





if __name__ == "__main__":
	
	cap = cv2.VideoCapture("train.mp4")
	fps = cap.get(cv2.CAP_PROP_FPS)


    '''
    F = int(os.getenv("F", "800"))
	W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
	Kinv = np.linalg.inv(K)
    '''

	prev = None
	frames = []

	while(cap.isOpened()):
	    ret, frame = cap.read()
	    if ret == True:
	    	next = Frame(frame, prev)
	    	frames.append(next)
	    	next.processFrame()
	    	prev = next
	    	if cv2.waitKey(1) & 0xFF == ord('q'):
	    		break
	    else:
	    	break

	cap.release()
	cv2.destroyAllWindows()


	


	#int( 1 / fps * 1000 / 1)
