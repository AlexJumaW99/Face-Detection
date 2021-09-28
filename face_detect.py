import cv2 #a module that allows us to carry out computer vision with python 

#store the xml file in a variable
#the file basically contains all the info needed by the computer to identify a face within an image 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#load the image
img = cv2.imread('photo.jpg')

#convert the image to greyscale 
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#get the coordinates of the rectangle that will surround the face
faces = face_cascade.detectMultiScale(gray_img,
    scaleFactor = 1.05, 
    minNeighbors = 5)

#draw a rectangle around the face using the coordinates 
#x,y are the coordinates for the top left vertice of the rectangle
#(x+w),(y+h) are the coordinates for the bottom right vertice 
for x, y, w, h in faces: 
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

#display the color image, with a rectangle drawn around the face 
cv2.imshow('Color',img)

#finally, we'll write code to close the window once the user presses any button
cv2.waitKey(0) 
cv2.destroyAllWindows()
