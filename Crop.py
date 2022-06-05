import cv2
# Read the input image
img = cv2.imread("C://Users//owner//Desktop//MS_Engage'22//trainimg//Cristiano_Ronaldo.jpg")

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the cascade
face_cascade = cv2.CascadeClassifier('C://Users//owner//AppData//Local//Programs//Python//Python310//Lib//site-packages//cv2//data//haarcascade_frontalface_alt2.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces and crop the faces
for (x, y, w, h) in faces:
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
	faces = img[y:y + h, x:x + w]
	cv2.imshow("face",faces)
	cv2.imwrite('face.jpg', faces)
	
# Display the output
cv2.imwrite('detected.jpg', img)
cv2.imshow('img', img)
cv2.waitKey()

