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

# Konfiguracja bazy danych
DATABASE_URL = "sqlite:///./speaker_verification.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modele bazy danych
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



# Tworzenie tabel w bazie danych
Base.metadata.create_all(bind=engine)

# Inicjalizacja aplikacji
app = FastAPI()

# Dependency dla sesji bazy danych
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Definicja modelu
ubm_model = UBM('D:\Base work dir\python projects\REST api\lstm.h5')



##########################   Users  ##########################


@app.post("/users/{username}")
def create_user(username: str, db: Session = Depends(get_db)):
    """
    Tworzy nowego użytkownika w bazie danych.
    
    Parametry:
    - username (str): Nazwa użytkownika.

    Zwraca:
    - Słownik zawierający ID, nazwę użytkownika oraz datę utworzenia.
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
    Aktualizuje dane użytkownika na podstawie jego ID.
    
    Parametry:
    - user_id (str): ID użytkownika do zaktualizowania.
    - request (UpdateUserRequest): Model danych zawierający nową nazwę użytkownika.

    Zwraca:
    - Słownik zawierający ID użytkownika, zaktualizowaną nazwę użytkownika oraz komunikat potwierdzający.
    
    W przypadku, gdy użytkownik o podanym ID nie istnieje, zwraca błąd HTTP 404.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User with id {user_id} not found.")
    
    # Aktualizacja danych użytkownika
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
    Tworzy nowy enrollment (rejestracja użytkownika z embeddingiem) na podstawie przesłanego pliku audio.
    
    Parametry:
    - user_id (str): ID użytkownika, dla którego tworzony jest enrollment.
    - file (UploadFile): Plik audio przesłany przez użytkownika.

    Zwraca:
    - Słownik zawierający ID enrollmentu, ID użytkownika oraz datę utworzenia.

    Jeśli użytkownik nie istnieje lub już posiada enrollment, zwracany jest odpowiedni błąd HTTP.
    """
    # Sprawdzenie, czy użytkownik istnieje
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Sprawdzenie, czy użytkownik już ma enrollment
    existing_enrollment = db.query(UserEnrollment).filter(UserEnrollment.user_id == user_id).first()
    if existing_enrollment:
        raise HTTPException(status_code=400, detail="User already has an enrollment")
    
    # Tymczasowe zapisanie pliku audio
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # Przetwarzanie pliku audio i generowanie embeddingów
        averaged_embedding = process_audio_file(temp_file_path, ubm_model)

        # Tworzenie nowego enrollmentu
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
        # Usuwanie tymczasowego pliku
        os.remove(temp_file_path)


@app.get("/users/enrollments/{user_id}")
def get_enrollments(user_id: str, db: Session = Depends(get_db)):
    """
    Pobiera informacje o enrollmentcie dla danego użytkownika.

    Parametry:
    - user_id (str): ID użytkownika, dla którego pobierane są dane enrollmentu.

    Zwraca:
    - Słownik zawierający ID enrollmentu, ID użytkownika, datę utworzenia oraz informację,
      czy embedding istnieje.

    Jeśli użytkownik nie posiada enrollmentu, zwracany jest błąd HTTP 404.
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
    Aktualizuje istniejący enrollment na podstawie przesłanego pliku audio.

    Parametry:
    - enrollment_id (str): ID istniejącego enrollmentu.
    - file (UploadFile): Nowy plik audio przesłany przez użytkownika.

    Zwraca:
    - Słownik zawierający ID enrollmentu, ID użytkownika oraz komunikat potwierdzający aktualizację.

    Jeśli enrollment nie istnieje, zwracany jest błąd HTTP 404.
    """
    enrollment = db.query(UserEnrollment).filter(UserEnrollment.user_id == user_id).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail=f"Enrollment with user id {user_id} not found.")
    
    # Tymczasowe zapisanie pliku audio
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    try:
        # Przetwarzanie pliku audio i generowanie nowego embeddingu
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
        # Usuwanie tymczasowego pliku
        os.remove(temp_file_path)


##########################   Identification  ##########################


@app.post('/users/enrollments/identification')
async def identify_user(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Identyfikuje użytkownika na podstawie przesłanego pliku audio, porównując jego embedding
    z enrollmentami zapisanymi w bazie danych.

    Parametry:
    - file (UploadFile): Plik audio przesłany przez użytkownika.

    Zwraca:
    - Słownik z listą do 5 najlepszych dopasowań, gdzie każde dopasowanie zawiera:
      - user_id: ID użytkownika.
      - enrollment_id: ID powiązanego enrollmentu.
      - username: Nazwę użytkownika.
      - similarity: Wartość podobieństwa (cosine similarity) w procentach.

    Jeśli nie znaleziono użytkowników z podobieństwem powyżej progu 0.8, zwracana jest odpowiednia informacja.
    Jeśli w bazie nie ma żadnych enrollmentów, zwracany jest błąd HTTP 404.
    """
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        averaged_embedding = process_audio_file(temp_file_path, ubm_model)

        # Pobieranie wszystkich enrollmentów z bazy
        enrollments = db.query(UserEnrollment).all()
        if not enrollments:
            raise HTTPException(status_code=404, detail="No enrollments found in the database.")
        
        # Lista do przechowywania wyników porównań
        results = []

        averaged_embedding_2d = np.array(averaged_embedding).reshape(1, -1)

        for enrollment in enrollments:
            enrollment_embedding = np.array(enrollment.embedding)

            username = db.query(User).filter(User.id == enrollment.user_id).first()

            # Obliczanie cosine similarity
            similarity = cosine_similarity(averaged_embedding_2d, enrollment_embedding.reshape(1, -1))[0][0]

            # Dodawanie wyniku do listy, jeśli przekracza threshold
            if similarity >= 0.8:
                results.append({
                    "user_id": enrollment.user_id,
                    "enrollment_id": enrollment.id,
                    "username": username.username,
                    "similarity": f"{round(similarity * 100, 2)}%"
                })

        # Sortowanie wyników po similarity w kolejności malejącej
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)

        # Zwracanie 5 najlepszych wyników lub informacji, że brak wyników
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
    Weryfikuje tożsamość użytkownika na podstawie przesłanego pliku audio, 
    porównując jego embedding z zapisanym enrollmentem w bazie danych.

    Parametry:
    - user_id (str): ID użytkownika, którego tożsamość jest weryfikowana.
    - file (UploadFile): Plik audio przesłany przez użytkownika do weryfikacji.

    Zwraca:
    - Słownik zawierający:
        - user_id: ID użytkownika.
        - verified (bool): Informację, czy użytkownik został zweryfikowany.
        - similarity (float): Wartość podobieństwa (cosine similarity) między embeddingami.
        - message (str): Wiadomość wskazująca wynik weryfikacji.

    Jeśli użytkownik nie posiada zapisanego enrollmentu, zwracany jest błąd HTTP 404.
    """
    enrollment = db.query(UserEnrollment).filter(UserEnrollment.user_id == user_id).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail=f"No enrollment found for user_id {user_id}.")

    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    try:
        averaged_embedding = process_audio_file(temp_file_path, ubm_model)

        # Konwersja embeddingów na numpy array
        enrollment_embedding = np.array(enrollment.embedding)
        averaged_embedding = np.array(averaged_embedding)

        # Obliczanie cosine similarity
        similarity = cosine_similarity(
            averaged_embedding.reshape(1, -1),
            enrollment_embedding.reshape(1, -1)
        )[0][0]

        # Próg weryfikacji (threshold)
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
            audio_buffer = collections.deque(maxlen=10)  # Bufor do 10 sekund audio
            while True:
                # Odbieranie danych binarnych
                data = await websocket.receive_bytes()

                # Odczyt długości i user_id
                user_id_length = int.from_bytes(data[:2], "big")
                user_id = data[2:2 + user_id_length].decode("utf-8")
                # Pozostałe dane audio
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

                # Wysyłanie danych do endpointu weryfikacji
                with open(temp_file_path, "rb") as audio_file:
                        response = await client.post(
                            url=f"http://127.0.0.1:8000/users/enrollments/verification/{user_id}", 
                            files={"file": audio_file}
                        )

                # Obsługa odpowiedzi
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
            audio_buffer = collections.deque(maxlen=10)  # Bufor do 10 sekund audio
            while True:
                # Odbieranie danych binarnych
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

                # Wysyłanie danych do endpointu identyfikacji
                with open(temp_file_path, "rb") as audio_file:
                        response = await client.post(
                            url="http://127.0.0.1:8000/users/enrollments/identification", 
                            files={"file": audio_file}
                        )

                # Obsługa odpowiedzi
                if response.status_code == 200:
                    identification_results = response.json()
                    await websocket.send_json(identification_results)
                else:
                    await websocket.send_json({"error": "Identification failed.", "details": response.text})
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    
    finally:
        os.remove(temp_file_path)