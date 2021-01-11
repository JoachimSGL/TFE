import cv2
import numpy as np
import mapper
from scipy import signal
import matplotlib.pyplot as plt
import time
#import time
image=cv2.imread("page6.jpg")   #read in the image
image=cv2.resize(image,(1300,800)) #resizing because opencv does not work well with bigger images
orig=image.copy()

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #RGB To Gray Scale
#cv2.imshow("Title",gray)

blurred=cv2.GaussianBlur(gray,(5,5),0)  #(5,5) is the kernel size and 0 is sigma that determines the amount of blur
#cv2.imshow("Blur",blurred)

edged=cv2.Canny(blurred,30,50)  #30 MinThreshold and 50 is the MaxThreshold
#cv2.imshow("Canny",edged)


contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
contours=sorted(contours,key=cv2.contourArea,reverse=True)

#the loop extracts the boundary contours of the page
target =[]
for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*p,True)

    if len(approx)==4:
        target=approx
        break
approx=mapper.mapp(target) #find endpoints of the sheet

pts=np.float32([[0,0],[420,0],[420,594],[0,594]])  #map to 800*800 target window

op=cv2.getPerspectiveTransform(approx,pts)  #get the top or bird eye view effect
dst=cv2.warpPerspective(gray,op,(420,594))

image_lettre=cv2.imread("lettre.jpg") 
lettre=cv2.cvtColor(image_lettre,cv2.COLOR_BGR2GRAY) 
print(np.shape(lettre))
print(np.shape(dst))
template = np.copy(dst[330:365, 298:320])
correlation = signal.correlate2d(dst,template, boundary='symm', mode='same')
y, x = np.unravel_index(np.argmax(correlation), correlation.shape)

print(y)
print(x)
print(correlation.shape)
cv2.imshow("partie",template)


dst[x:300,y:300] = 0


print(correlation[x,y])




cv2.imshow("Scanned",dst)

# press q or Esc to close
cv2.waitKey(0)
cv2.destroyAllWindows()

    