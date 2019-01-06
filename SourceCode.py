import cv2
import tkinter
from tkinter import filedialog
import copy
import numpy


# Global declarations
face_cascade = cv2.CascadeClassifier("C:\PythonInterpreter\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:\PythonInterpreter\Lib\site-packages\cv2\data\haarcascade_eye.xml")
filePathName = None
imagePtr = None
imagePtrCurrentCopy = None
imagePtrPrevCopy = None


#Function creating an Open Button and GUI
def GUI():
    root = tkinter.Tk()
    root.wm_iconbitmap("chotu.ico")
    #root.wm_iconbitmap('chotu.ico')
    root.minsize(300,300)
    root.title("Image Optimizer")
    frame = tkinter.Frame(root)
    frame.pack()

    canvas = tkinter.Canvas(frame, width=952, height=500)
    canvas.pack(side=tkinter.BOTTOM)
    photo = tkinter.PhotoImage(file="gify2.gif")
    #photo = tkinter.PhotoImage(file="gify2.gif")
    canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

    openImageButton = tkinter.Button(frame,text="Open",fg="Black",bg="Gray",command=openImage,height=2,width=11)
    openImageButton.pack(side=tkinter.LEFT)
    createSaveButton(root,frame)
    createSaveAsButton(root, frame)
    createBWButton(root,frame)
    createGrayscaleButton(root,frame)
    createBlurImageButton(root,frame)
    createFaceDetectButton(root,frame)
    createEyeDetectButton(root,frame)
    createRotateby90Button(root,frame)
    createResetButton(root,frame)
    createUndoButton(root,frame)
    createBorderButton(root,frame)
    root.mainloop()

#Function creating a save button
def createSaveButton(root,frame):
    # only gif type buttons are accepted. Use tkinter.PhotoImage(file="image.ext")
    saveImageButton = tkinter.Button(frame,text="Save",fg="Black",bg="Gray",command=saveImage,height=2,width=11)
    #saveImageButton.photo = pic
    saveImageButton.pack(side=tkinter.LEFT)

#Function creating a B/W Button
def createBWButton(root,frame):
    BlacknWhiteImageButton = tkinter.Button(frame, text="B/W", fg="Black", bg="Gray", command=BlacknWhite, height=2, width=11)
    BlacknWhiteImageButton.pack(side=tkinter.LEFT)

#Function creating a grayscale button
def createGrayscaleButton(root,frame):
    grayscaleImageButton = tkinter.Button(frame,text="Grayscale",fg="Black",bg="Gray",command=grayscaleConverter,height=2,width=11)
    grayscaleImageButton.pack(side=tkinter.LEFT)

#Function creating a saveas button
def createSaveAsButton(root,frame):
    saveAsImageButton = tkinter.Button(frame,text=".Extension",fg="Black",bg="Gray",command=saveAsImage,height=2,width=11)
    saveAsImageButton.pack(side=tkinter.LEFT)

#Function top create a blur image button
def createBlurImageButton(root,frame):
    blurImageButton = tkinter.Button(frame, text="Blur", fg="Black", bg="Gray", command=blurImage,height=2, width=11)
    blurImageButton.pack(side=tkinter.LEFT)

#Function to create a faceDetectionButton
def createFaceDetectButton(root,frame):
    faceDetectButton = tkinter.Button(frame, text="Face Detect", fg="Black", bg="Gray", command=faceDetect, height=2, width=11)
    faceDetectButton.pack(side=tkinter.LEFT)

#Function creating a eyeDetectionButton
def createEyeDetectButton(root,frame):
    eyeDetectButton = tkinter.Button(frame,text="Eye Detect",fg="Black",bg="Gray",command=eyeDetect,height=2,width=11)
    eyeDetectButton.pack(side=tkinter.LEFT)

#Function creating a rotation by 90 degree button
def createRotateby90Button(root,frame):
    RotationBy90Button = tkinter.Button(frame, text="Rotate", fg="Black", bg="Gray", command=rotate90, height=2,width=11)
    RotationBy90Button.pack(side=tkinter.LEFT)

#Function creating a reset button
def createResetButton(root,frame):
    resetButton = tkinter.Button(frame, text="Reset", fg="Black", bg="Gray", command=reset, height=2, width=11)
    resetButton.pack(side=tkinter.LEFT)

#Function creating an undo button
def createUndoButton(root,frame):
    undoButton = tkinter.Button(frame, text="Undo", fg="black", bg="Gray", command=undo, height=2, width=11)
    undoButton.pack(side=tkinter.LEFT)

#Function creating the border button
def createBorderButton(root,frame):
    borderButton = tkinter.Button(frame,text="Add Border",fg="black",bg="Gray",command=setBorder,height=2,width=11)
    borderButton.pack()

#Function to convert image to black and white
def BlacknWhite():
    if(filePathName):
        global imagePtr
        global imagePtrCurrentCopy
        global imagePtrPrevCopy


        # Convert image to grayscale
        #grayscaleConverter()
        if(len(imagePtrCurrentCopy.shape) == 3):
            #imagePtrPrevCopy = numpy.concatenate([imagePtrPrevCopy,imagePtrCurrentCopy])
            imagePtrPrevCopy = copy.deepcopy(imagePtrCurrentCopy)
            imagePtrCurrentCopy = cv2.cvtColor(imagePtrCurrentCopy, cv2.COLOR_BGR2GRAY)
            #Convert grayscale image to binary
            print("Converting to Black and white")
            (threshold , im_bw) = cv2.threshold(imagePtrCurrentCopy,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cv2.imshow(filePathName, im_bw)
            imagePtrCurrentCopy = copy.deepcopy(im_bw)
            # It automatically determines the threshold using Otsu's method
            print("Black and white accomplished...")
        else:
            print("\nAlready colorless...:",imagePtrCurrentCopy.shape)
    else:
        print("No file selected")

#Function to convert image to grayscale mode
def grayscaleConverter():
    if(filePathName):
        global imagePtr
        global imagePtrCurrentCopy
        global imagePtrPrevCopy

        if(len(imagePtrCurrentCopy.shape) ==3):
            #imagePtrPrevCopy = numpy.concatenate([imagePtrPrevCopy,imagePtrCurrentCopy])
            imagePtrPrevCopy = copy.deepcopy(imagePtrCurrentCopy)
            print("Converting to grayscale...")
            imagePtrCurrentCopy = cv2.cvtColor(imagePtrCurrentCopy,cv2.COLOR_BGR2GRAY)
            cv2.imshow(filePathName, imagePtrCurrentCopy)
            print("Grayscale conversion successful...")
        else:
            print("Already Grayscale:",imagePtrCurrentCopy.shape)
    else:
        print("No file selected")

#Function to browse an image and open it
def openImage():
    global filePathName
    global imagePtr
    global imagePtrCurrentCopy
    global imagePtrPrevCopy

    print("Opening Image...")

    filePathName = filedialog.askopenfilename(initialdir = "C:\\",title = "Select Image file",filetypes =(("jpeg files","*.jpg"),("gif files","*.gif"),("png files","*.png"),("All","*.*")))

    print(filePathName)

    imagePtr = cv2.imread(filePathName,1) # 1 means with color info
    height,width = imagePtr.shape[:2]

    print("Width:",width," Height:",height)

    cv2.namedWindow(filePathName,cv2.WINDOW_AUTOSIZE)
    #windowName = cv2.resizeWindow(filePathName,1920, 1080)
    if(height < 1600):
        imagePtr = cv2.resize(imagePtr,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    elif(height < 2000 ):
        imagePtr = cv2.resize(imagePtr, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    else:
        imagePtr = cv2.resize(imagePtr, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    print(imagePtr)

    imagePtrCurrentCopy = copy.deepcopy(imagePtr)
    imagePtrPrevCopy = copy.deepcopy(imagePtr)
    cv2.imshow(filePathName,imagePtrCurrentCopy)
    cv2.waitKey(1)
    print("Image Opened successfully!")


# Function to save an image
def saveImage():
    global imagePtr
    global imagePtrCurrentCopy

    fileName = ""
    if(filePathName):
        print("Saving your Image...")

        cvtToList = list(filePathName.split("/"))
        fileName = fileName + cvtToList[-1]

        print(fileName)

        imagePtr = copy.deepcopy(imagePtrCurrentCopy)
        cv2.imwrite(filePathName,imagePtr)

        print(filePathName)
        print("Saved successfully...")
    else:
        print("No file selected")

#Function for saveAs dialog box
def saveAsImage():
    global filePathName

    if(filePathName):
        print("Saving your Image...")

        filePathName = filedialog.asksaveasfilename(initialdir ="C:/",title="Select file", filetypes = (("jpeg files","*.jpg"),("all files","*.*")),defaultextension=".jpg")

        print(filePathName)

        saveImage()

        print("Image saved successfully...")
    else:
        print("No file selected")

#Function to blur the image
def blurImage():
    if(filePathName):
        global imagePtr
        global imagePtrCurrentCopy
        global imagePtrPrevCopy

        print("Blurring your Image...")

        #imagePtrPrevCopy = numpy.concatenate([imagePtrPrevCopy,imagePtrCurrentCopy])
        imagePtrPrevCopy = copy.deepcopy(imagePtrCurrentCopy)
        imagePtrCurrentCopy = cv2.blur(imagePtrCurrentCopy,(3,3))
        #print("Test: ",imagePtrPrevCopy.shape)
        cv2.imshow(filePathName,imagePtrCurrentCopy)

        print("Blur successful...")
    else:
        print("No file selected")

#Function to detect faces in the image
def faceDetect():
    if(filePathName):
        global imagePtr
        global imagePtrCurrentCopy
        global imagePtrPrevCopy

        #grayscaleConverter()
        print("Detecting faces...")
        #imagePtrPrevCopy = numpy.concatenate([imagePtrPrevCopy,imagePtrCurrentCopy])
        imagePtrPrevCopy = copy.deepcopy(imagePtrCurrentCopy)
        faces = face_cascade.detectMultiScale(imagePtrCurrentCopy,1.1,5)
        if(len(faces)!=0):
            print("Face(s) detected.")
        else:
            print("Face(s) not found.")
        for (x,y,w,h) in faces:
            cv2.rectangle(imagePtrCurrentCopy,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow(filePathName,imagePtrCurrentCopy)
    else:
        print("No file selected")

#Function to detect eyes in the image
def eyeDetect():
    if(filePathName):
        global imagePtr
        global imagePtrCurrentCopy
        global imagePtrPrevCopy

        #imagePtrPrevCopy = numpy.concatenate([imagePtrPrevCopy,imagePtrCurrentCopy])
        imagePtrPrevCopy = copy.deepcopy(imagePtrCurrentCopy)
        print("Detecting eyes...")

        eyes = []
        if(len(imagePtrCurrentCopy.shape) ==3):
            grayImage = cv2.cvtColor(imagePtrCurrentCopy,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(grayImage,1.1,5)
            #cv2.imshow(filePathName,grayImage)
            for (x,y,w,h) in faces:
                roi_gray = grayImage[y:y+h,x:x+w]
                roi_color = imagePtrCurrentCopy[y:y+h,x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)

                for (eye_x,eye_y,eye_w,eye_h) in eyes:
                    cv2.rectangle(roi_color,(eye_x,eye_y),(eye_x+eye_w,eye_y+eye_h),(0,0,0),2)
        else:
            faces = face_cascade.detectMultiScale(imagePtrCurrentCopy, 1.1, 5)
            # cv2.imshow(filePathName,grayImage)
            for (x, y, w, h) in faces:
                #roi_gray = grayImage[y:y + h, x:x + w]
                roi_color = imagePtrCurrentCopy[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_color)

                for (eye_x, eye_y, eye_w, eye_h) in eyes:
                    cv2.rectangle(roi_color, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 0, 0), 2)
        if(len(eyes)!=0):
            print("Eyes detected.")
        else:
            print("Eyes not found...")
        cv2.imshow(filePathName, imagePtrCurrentCopy)
    else:
        print("No file selected")

#Function to rotate by 90 degrees
def rotate90():
    global imagePtr
    global imagePtrCurrentCopy
    global imagePtrPrevCopy

    if(filePathName):
        imagePtrPrevCopy = copy.deepcopy(imagePtrCurrentCopy)
        #imagePtrPrevCopy = numpy.concatenate([imagePtrPrevCopy,imagePtrCurrentCopy])
        print("Rotation initiated...")
        if(len(imagePtrCurrentCopy.shape) ==3):
            pixel_rows, pixel_columns, channels = imagePtrCurrentCopy.shape
        else:
            pixel_rows,pixel_columns = imagePtrCurrentCopy.shape
        #print(pixel_rows,pixel_columns)
        two_dimensional_rotation_matrix = cv2.getRotationMatrix2D((pixel_columns/2,pixel_rows/2),90,1)
        # Map changed matrix of pixels into original image
        imagePtrCurrentCopy = cv2.warpAffine(imagePtrCurrentCopy,two_dimensional_rotation_matrix,(pixel_columns,pixel_rows))
        # Display the change
        cv2.imshow(filePathName,imagePtrCurrentCopy)

        print("Rotation successful...")
    else:
        print("No file selected.")

# Function to reset all the changes made in the original image
def reset():
    global imagePtr
    global imagePtrCurrentCopy

    if(filePathName):
        print("Resetting your Image...")
        imagePtrCurrentCopy = copy.deepcopy(imagePtr)
        imagePtrPrevCopy = copy.deepcopy(imagePtr)
        #cv2.destroyWindow(filePathName)
        cv2.imshow(filePathName,imagePtrCurrentCopy)
        print("Image reset successful.")
    else:
        print("No file selected.")

# Function to undo the last change
def undo():
    global imagePtrCurrentCopy
    global imagePtrPrevCopy

    if(filePathName):
        print("Reverting your change...")

        imagePtrCurrentCopy = copy.deepcopy(imagePtrPrevCopy)
        print("Testing : ",imagePtrCurrentCopy)
        #imagePtrPrevCopy = list(imagePtrPrevCopy)
        imagePtrPrevCopy = imagePtr
        #numpy.delete(imagePtrPrevCopy,-1)
        #del imagePtrPrevCopy[-1]
        cv2.imshow(filePathName, imagePtrCurrentCopy)
        print("Undo successful.")
    else:
        print("No file selected.")

def setBorder():
    global imagePtrPrevCopy
    global imagePtrCurrentCopy
    #print("Border Crossing...")
    borderSet = cv2.copyMakeBorder(imagePtrCurrentCopy,3,3,3,3,cv2.BORDER_CONSTANT,value=[0,0,0])
    imagePtrPrevCopy = copy.deepcopy(imagePtrCurrentCopy)
    imagePtrCurrentCopy = copy.deepcopy(borderSet)
    cv2.imshow(filePathName,imagePtrCurrentCopy)

def main():
    GUI()
    print("Code terminated successfully...")

main()
