from fastapi import FastAPI
from app.routes.verify import router
from app.routes import verify

app = FastAPI(
    title= "Face Verification I",
    description= "API for verifying students using facial recognition",
    version= "1.0.0"
)
#verify_router
app.include_router(verify.router)