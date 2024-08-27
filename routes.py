from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from io import BytesIO
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from function import get_face_embedding, verify_face, get_gender
from config import settings

router = APIRouter()

class CombinedResponse(BaseModel):
    match: bool
    gender: str

class UserCredentials(BaseModel):
    username: str
    password: str

class Settings(BaseModel):
    authjwt_secret_key: str = settings.SECRET_KEY

@AuthJWT.load_config
def get_config():
    return Settings()

# @router.post("/login/")
# def login(user: UserCredentials, Authorize: AuthJWT = Depends()):
#     if user.username != "testuser" or user.password != "testpassword":
#         raise HTTPException(status_code=401, detail="Bad username or password")

#     access_token = Authorize.create_access_token(subject=user.username)
#     return {"access_token": access_token}

@router.post("/verify_and_gender/", response_model=CombinedResponse)
async def verify_and_detect_gender(
    id_card_file: UploadFile = File(...),
    video_file: UploadFile = File(...),
    Authorize: AuthJWT = Depends()
):
    try:
        Authorize.jwt_required()
    except AuthJWTException as e:
        raise HTTPException(status_code=401, detail="Token Expired")

    if not (id_card_file.filename.lower().endswith(('png', 'jpg', 'jpeg')) and video_file.filename.lower().endswith('mp4')):
        raise HTTPException(status_code=400, detail="File type not allowed")

    # Read the files
    id_card_data = await id_card_file.read()
    video_data = await video_file.read()

    # Process the ID card image to extract face embedding
    id_card_embedding = get_face_embedding(BytesIO(id_card_data), is_video=False)
    if id_card_embedding is None:
        raise HTTPException(status_code=500, detail="Unable to process the ID card image")

    # Process the video to extract face embeddings
    video_frame_embedding = get_face_embedding(BytesIO(video_data), is_video=True)
    if video_frame_embedding is None:
        raise HTTPException(status_code=500, detail="Unable to process the video file")

    # Perform face verification
    match = verify_face(id_card_embedding, video_frame_embedding)

    # Perform gender detection on the ID card image
    gender = get_gender(BytesIO(id_card_data))
    if gender is None:
        raise HTTPException(status_code=500, detail="Unable to detect gender")

    return CombinedResponse(match=match, gender=gender)