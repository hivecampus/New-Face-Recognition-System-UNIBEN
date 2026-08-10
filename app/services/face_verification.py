import cv2
import numpy as np
from fastapi import UploadFile

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_recognition_model = "models/face_recognition_sface_2021dec.onnx"
recognizer = cv2.FaceRecognizerSF.create(
    face_recognition_model,
    ""
)

async def verify_face(
    image : UploadFile,
    selfie : UploadFile
):
    # reading the image and selfie into bytes
    image_bytes = await image.read()
    selfie_bytes = await selfie.read()

    # converting the bytes into numbers or array
    image_array = np.frombuffer(image_bytes, dtype = np.uint8)
    selfie_array = np.frombuffer(selfie_bytes, dtype= np.uint8)

    #loading with open cv
    id_image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )
    id_selfie = cv2.imdecode(
        selfie_array,
        cv2.IMREAD_COLOR
    )


    #converting the images to gray scale
    image_gray = cv2.cvtColor(id_image, cv2.COLOR_BGR2GRAY)
    selfie_gray = cv2.cvtColor(id_selfie, cv2.COLOR_BGR2GRAY)

    #Detecting the faces
    image_face = detector.detectMultiScale(image_gray, scaleFactor = 1.2, minNeighbors = 7)
    selfie_face = detector.detectMultiScale(selfie_gray, scaleFactor = 1.2, minNeighbors = 7)
    

    #validating the face in the image 
    if len(image_face) == 0:
        return{
            "success" : False,
            "source" : "image",
            "error" : "no_face",
            "message" : "no face detected in the image"
        }
    if len(image_face) > 1:
        return{
            "success" : False,
            "source" : "image",
            "error" : "multiple_face",
            "message" : "multiple face detected in the image"
        }


    #validating the face in the selfie 
    if len(selfie_face) == 0:
        return{
            "success" : False,
            "source" : "selfie",
            "error" : "no_face",
            "message" : "no face detected in the selfie"
        }
    if len(selfie_face) > 1:
        return{
            "success" : False,
            "source" : "selfie",
            "error" : "multiple_face",
            "message" : "multiple face detected in the selfie"
        }    
    
    ###IMAGE
         
    #aligning and cropping the face
    aligned_image_face = recognizer.alignCrop(
        id_image,
        image_face[0]
    )

    #embedding the cropped and alined face
    image_embedding = recognizer.feature(
        aligned_image_face
    )

    ###SELFIE
         
    #aligning and cropping the face
    aligned_image_selfie = recognizer.alignCrop(
        id_selfie,
        selfie_face[0]
    )

    #embedding the cropped and alined face
    selfie_embedding = recognizer.feature(
        aligned_image_selfie
    )


    #comparing and scoring the two faces
    score = recognizer.match(
        image_embedding,
        selfie_embedding,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    #returning the results
    threshold = 0.363

    if score >= threshold:
        return {
            "success" : True,
            "score" : float(score),
            "message" : "face matched successfully" 
        }
    return {
        "success" : False,
        "score" : float(score),
        "message" : "face does not match" 
    }   