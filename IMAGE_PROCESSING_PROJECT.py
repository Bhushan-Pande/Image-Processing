#!/usr/bin/env python
# coding: utf-8

# ## PROJECT FOR IMAGE PROCESSING

# In[1]:


import cv2                        #Cv2 is used for computer vision
import numpy as np                # Numpy is used for creating multidimensional arrays
import pytesseract                #pytesseract is used for taking out the text from image


# In[19]:


pytesseract.pytesseract.tesseract_cmd
= r'C:\Program Files\Tesseract-OCR\tesseract.exe' #setting up the path of tesseract.exe
large = cv2.imread(r'C:\Users\Dell\3D Objects\DWDM\Envirya assignment\FILES.png')     #Image is read
print(large)


# In[13]:


#CONVERSION OF AN FILE IMAGE
rgb = cv2.pyrDown(large)   #Gaussian pyramids using cv2. pyrDown() 
small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)    #Converting to GrayScalar
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #Cv2.GetStructuringElement Method (MorphShapes, Size). Returns a structuring element of the specified size and shape for morphological operations
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel) #difference between dilation and erosion of an image.   
_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)   #threshold image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)# using RETR_EXTERNAL instead of RETR_CCOMP
print(hierarchy)   #seeing the heirachy
l1=[]
mask = np.zeros(bw.shape, dtype=np.uint8)
print(contours)    #Looking for contours


# In[14]:


for idx in range(len(contours)):                            #Applying for loop on countors
    x, y, w, h = cv2.boundingRect(contours[idx])            #Creating bounding rectangle
    mask[y:y+h, x:x+w] = 0                                          
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)  #drawing countors for creating rectangle later
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)   #counting non zeros      
    print(r)
    if r > 0.10 and w > 4 and h >2:                             #Declaring the size on which rectangle to be made
        rect=cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)  #Making rectangle
        l1.append(rect)
print(l1)


# In[ ]:


#VISUALISING THE REAL IMAGE
cv2.imshow('Image',large)

#VISUALISING THE IMAGE IN WHICH RECTANGLES ARE PROVIDED TO GIVE US THE INFERENCE WHICH TEXT ARE MORE LIKELY TO BE EXTRACTED
#THROUGH PYTESSERACT
cv2.imshow('rects', rgb)


cv2.waitKey(0)
# Destroying present windows on screen
cv2.destroyAllWindows()


# In[18]:



custom_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1 -l eng+ita' #CUSTOM CONFIGURATION

details = pytesseract.image_to_string(large,config=custom_config, lang='eng')    #Conversion of image to string



print(details)                                          #printing all thats written on image

# saving to a txt file

with open("FIRST.csv", "w") as text_file:              #Appending in CSV file
    text_file.write(details) 


# In[ ]:




