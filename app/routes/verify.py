from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.face_verification import verify_face

router = APIRouter()


@router.get("/")
async def home():
    return{
        "message": "Welcome to the facial verification API"
    }


@router.post("/verify_faces")
async def verify_face_route(
    image : UploadFile = File(...),
    selfie : UploadFile = File(...)
):
    #face result
    result = await verify_face(image, selfie)
    return result