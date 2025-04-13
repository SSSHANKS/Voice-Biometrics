from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, WebSocket, WebSocketDisconnect
from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
from uuid import uuid4
from datetime import datetime, timezone
from data_processing import UBM, process_audio_file
from pydantic import BaseModel
from typing import List
import os
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
import httpx
import wave
import collections

DATABASE_URL = "sqlite:///./speaker_verification.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    username = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

class UserEnrollment(Base):
    __tablename__ = "user_enrollments"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    embedding = Column(JSON, nullable=False)
    created_at = Column(DateTime, default= datetime.now(timezone.utc))



Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

ubm_model = UBM('lstm.h5')



##########################   Users  ##########################


@app.post("/users/{username}")
def create_user(username: str, db: Session = Depends(get_db)):
    """
    Creates a new user in the database.
    
    Parameters:
    - username (str): Username.
    
    Returns:
    - dict: A dictionary containing the ID, username, and creation date.
    """
    user = User(username=username)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "username": user.username, "created_at": user.created_at}

@app.get("/users/")
def list_users(db: Session = Depends(get_db)):
    """
    Pobiera listę wszystkich użytkowników z bazy danych.

    """
    users = db.query(User).all()
    return users

class UpdateUserRequest(BaseModel):
    username: str

@app.put("/users/{user_id}")
def update_user(user_id: str, request: UpdateUserRequest, db: Session = Depends(get_db)):
    """
    Updates user data based on their ID.
    
    Parameters:
    - user_id (str): ID of the user to update.
    - request (UpdateUserRequest): Data model containing the new username.
    
    Returns:
    - dict: A dictionary containing the user ID, updated username, and a confirmation message.
    
    If a user with the given ID does not exist, returns an HTTP 404 error.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User with id {user_id} not found.")
    
    user.username = request.username
    db.commit()
    db.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "message": "User updated successfully."
    }


##########################   Enrolments  ##########################


@app.put("/users/enrollments/{user_id}")
async def create_enrollment(user_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Creates a new enrollment (user registration with an embedding) based on the uploaded audio file.
    
    Parameters:
    - user_id (str): ID of the user for whom the enrollment is being created.
    - file (UploadFile): Audio file uploaded by the user.
    
    Returns:
    - dict: A dictionary containing the enrollment ID, user ID, and creation date.
    
    If the user does not exist or already has an enrollment, an appropriate HTTP error is returned.
    """

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    existing_enrollment = db.query(UserEnrollment).filter(UserEnrollment.user_id == user_id).first()
    if existing_enrollment:
        raise HTTPException(status_code=400, detail="User already has an enrollment")
    
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        averaged_embedding = process_audio_file(temp_file_path, ubm_model)

        enrollment = UserEnrollment(user_id=user_id, embedding=averaged_embedding.tolist())
        db.add(enrollment)
        db.commit()
        db.refresh(enrollment)

        return {
            "id": enrollment.id,
            "user_id": enrollment.user_id,
            "created_at": enrollment.created_at,
        }
    finally:
        os.remove(temp_file_path)


@app.get("/users/enrollments/{user_id}")
def get_enrollments(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieves enrollment information for a given user.
    
    Parameters:
    - user_id (str): ID of the user whose enrollment data is being retrieved.
    
    Returns:
    - dict: A dictionary containing the enrollment ID, user ID, creation date, and a flag indicating whether an embedding exists.
    
    If the user does not have an enrollment, an HTTP 404 error is returned.
    """
    enrollment = db.query(UserEnrollment).filter(UserEnrollment.user_id == user_id).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail="No enrollment found for the user")
    return {
            "id": enrollment.id,
            "user_id": enrollment.user_id,
            "created_at": enrollment.created_at,
            "embedding_exist": enrollment.embedding is not None 
        }

@app.put("/users/enrollments/{user_id}")
async def update_enrollment(user_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Updates an existing enrollment based on the uploaded audio file.
    
    Parameters:
    - enrollment_id (str): ID of the existing enrollment.
    - file (UploadFile): New audio file uploaded by the user.
    
    Returns:
    - dict: A dictionary containing the enrollment ID, user ID, and a confirmation message.
    
    If the enrollment does not exist, an HTTP 404 error is returned.
    """
    
    enrollment = db.query(UserEnrollment).filter(UserEnrollment.user_id == user_id).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail=f"Enrollment with user id {user_id} not found.")
    
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    try:
        new_embedding = process_audio_file(temp_file_path, ubm_model)

        enrollment.embedding = new_embedding.tolist()
        db.commit()
        db.refresh(enrollment)

        return {
            "id": enrollment.id,
            "user_id": enrollment.user_id,
            "message": "Enrollment updated successfully."
        }
    finally:
        os.remove(temp_file_path)


##########################   Identification  ##########################


@app.post('/users/enrollments/identification')
async def identify_user(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Identifies a user based on the uploaded audio file by comparing its embedding
    to enrollments stored in the database.
    
    Parameters:
    - file (UploadFile): Audio file uploaded by the user.
    
    Returns:
    - dict: A list of up to 5 best matches, where each match includes:
      - user_id: User ID.
      - enrollment_id: Associated enrollment ID.
      - username: Username.
      - similarity: Similarity score (cosine similarity) in percent.
    
    If no users are found with similarity above the 0.8 threshold, an appropriate message is returned.
    If there are no enrollments in the database, an HTTP 404 error is returned.
    """

    
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        averaged_embedding = process_audio_file(temp_file_path, ubm_model)

        enrollments = db.query(UserEnrollment).all()
        if not enrollments:
            raise HTTPException(status_code=404, detail="No enrollments found in the database.")
        
        results = []

        averaged_embedding_2d = np.array(averaged_embedding).reshape(1, -1)

        for enrollment in enrollments:
            enrollment_embedding = np.array(enrollment.embedding)

            username = db.query(User).filter(User.id == enrollment.user_id).first()

            similarity = cosine_similarity(averaged_embedding_2d, enrollment_embedding.reshape(1, -1))[0][0]

            if similarity >= 0.8:
                results.append({
                    "user_id": enrollment.user_id,
                    "enrollment_id": enrollment.id,
                    "username": username.username,
                    "similarity": f"{round(similarity * 100, 2)}%"
                })

        results = sorted(results, key=lambda x: x['similarity'], reverse=True)

        if results:
            return {"top_matches": results[:5]}
        else:
            return {"detail": "No matches found."}

    finally:
        os.remove(temp_file_path)



##########################   Verification   ##########################


@app.post('/users/enrollments/verification/{user_id}')
async def verify_user(user_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Verifies a user's identity based on the uploaded audio file by comparing its embedding 
    to the stored enrollment in the database.
    
    Parameters:
    - user_id (str): ID of the user whose identity is being verified.
    - file (UploadFile): Audio file uploaded by the user for verification.
    
    Returns:
    - dict: A dictionary containing:
        - user_id: User ID.
        - verified (bool): Indicates whether the user was successfully verified.
        - similarity (float): Cosine similarity score between embeddings.
        - message (str): Message indicating the verification result.
    
    If the user does not have a stored enrollment, an HTTP 404 error is returned.
    """

    
    enrollment = db.query(UserEnrollment).filter(UserEnrollment.user_id == user_id).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail=f"No enrollment found for user_id {user_id}.")

    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    try:
        averaged_embedding = process_audio_file(temp_file_path, ubm_model)

        enrollment_embedding = np.array(enrollment.embedding)
        averaged_embedding = np.array(averaged_embedding)

        similarity = cosine_similarity(
            averaged_embedding.reshape(1, -1),
            enrollment_embedding.reshape(1, -1)
        )[0][0]

        threshold = 0.42

        if similarity >= threshold:
            return {
                "user_id": user_id,
                "verified": True,
                "similarity": similarity,
                "message": "User verified successfully."
            }
        else:
            return {
                "user_id": user_id,
                "verified": False,
                "similarity": similarity,
                "message": "Verification failed. Similarity below threshold."
            }

    finally:
        os.remove(temp_file_path)


##########################   Delete   ##########################


@app.delete("/users/{user_id}")
def delete_user(user_id: str, db: Session = Depends(get_db)):
    """
    Usuwa użytkownika i powiązany z nim enrollment z bazy danych.

    Parametry:
    - user_id (str): ID użytkownika, który ma zostać usunięty.

    Zwraca:
    - Słownik zawierający szczegóły operacji:
        - detail (str): Informacja o pomyślnym usunięciu użytkownika i powiązanego enrollmentu.

  
    Jeśli użytkownik o podanym `user_id` nie istnieje, zwracany jest błąd HTTP 404.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.query(UserEnrollment).filter(UserEnrollment.user_id == user_id).delete()
    db.delete(user)
    db.commit()
    return {"detail": "User and related enrollment deleted"}


##########################   Websocket   ##########################


@app.websocket("/ws/verify/{client_id}")
async def websocket_verify(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    try:
        async with httpx.AsyncClient() as client:
            audio_buffer = collections.deque(maxlen=10)  
            while True:
                data = await websocket.receive_bytes()

                user_id_length = int.from_bytes(data[:2], "big")
                user_id = data[2:2 + user_id_length].decode("utf-8")

                audio_data = data[2 + user_id_length:]

                audio_buffer.append(audio_data)
                combined_data = b"".join(audio_buffer)

                temp_file_path = "temp_audio_file.wav"
                combined_data = np.frombuffer(combined_data, dtype=np.float32)

                with wave.open(temp_file_path, "wb") as wav_file:
                    wav_file.setnchannels(1)  
                    wav_file.setsampwidth(2)  
                    wav_file.setframerate(8000)
                    wav_file.writeframes(combined_data.tobytes())

                with open(temp_file_path, "rb") as audio_file:
                        response = await client.post(
                            url=f"http://127.0.0.1:8000/users/enrollments/verification/{user_id}", 
                            files={"file": audio_file}
                        )

                if response.status_code == 200:
                    verification_results = response.json()
                    await websocket.send_json(verification_results)
                else:
                    await websocket.send_json({"error": "Verification failed.", "details": response.text})
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    
    finally:
        os.remove(temp_file_path)




@app.websocket("/ws/identify/{client_id}")
async def websocket_identify(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    try:
        async with httpx.AsyncClient() as client:
            audio_buffer = collections.deque(maxlen=10)  
            while True:

                data = await websocket.receive_bytes()

                audio_buffer.append(data)
                combined_data = b"".join(audio_buffer)

                temp_file_path = "temp_audio_file.wav"
                audio_data = np.frombuffer(combined_data, dtype=np.float32)

                with wave.open(temp_file_path, "wb") as wav_file:
                    wav_file.setnchannels(1)  
                    wav_file.setsampwidth(2)  
                    wav_file.setframerate(8000)
                    wav_file.writeframes(audio_data.tobytes())

                
                with open(temp_file_path, "rb") as audio_file:
                        response = await client.post(
                            url="http://127.0.0.1:8000/users/enrollments/identification", 
                            files={"file": audio_file}
                        )

                
                if response.status_code == 200:
                    identification_results = response.json()
                    await websocket.send_json(identification_results)
                else:
                    await websocket.send_json({"error": "Identification failed.", "details": response.text})
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    
    finally:
        os.remove(temp_file_path)
