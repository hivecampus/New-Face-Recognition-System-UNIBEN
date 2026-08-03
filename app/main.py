from fastapi import FastAPI
from fastapi import UploadFile, File

app = FastAPI(
    title= "Face Verification I",
    description= "API for verifying students using facial recognition",
    version= "1.0.0"
)

#home
@app.get("/")
def home():
    return{
        "message": "Welcome to the facial verification API"
    }

#verify_face
@app.post("/verify_face")
def verify_face(
    image : UploadFile = File(...),
    selfie : UploadFile = File(...)
):
    
    return {
        "message": "Verification endpoint"
    }