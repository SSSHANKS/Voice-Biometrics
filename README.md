# Voice Biometrics REST api

## Description

Project is designed for training and evaluating a Recurrent Neural Network-based Universal Background Model (UBM) for client verification and identification using audio signal processing. The project extracts MFCC features from audio files, computes embeddings with a pre-trained RNN model (optionally transforming them using LDA), and provides a REST API for enrollment, identification, and verification. Additionally, WebSocket endpoints allow for real-time streaming of audio data for both verification and identification.

## File Structure

Below is a visual representation of the project file architecture:

```
RNN-UBM/
├── rnn-ubm-training.ipynb       # Training notebook for the UBM model
├── rnn-ubm-evaluation.ipynb     # Evaluation notebook for the UBM model
└── data-processing.ipynb        # Notebook for data processing and analysis
data_processing.py           # Audio processing and embedding generation module
main.py                      # FastAPI app with API endpoints for enrollment, verification, identification
client_identification.py     # WebSocket client for real-time identification
client_verification.py       # WebSocket client for real-time verification
```

## Installation and Setup

### 1. Requirements

- Python 3.7 or higher  
- Libraries:
  - `fastapi`
  - `uvicorn`
  - `tensorflow`
  - `librosa`
  - `numpy`
  - `scikit-learn`
  - `joblib`
  - `httpx`
  - `websockets`
  - `pydantic`
  - `sqlalchemy`
  - `wave`

### 2. Install Dependencies

Use pip to install the required libraries:

```bash
pip install fastapi uvicorn tensorflow librosa numpy scikit-learn joblib httpx websockets pydantic sqlalchemy
```

### 3. Running the Server

Start the FastAPI server using:

```bash
uvicorn main:app --reload
```

Then access the API docs at `http://127.0.0.1:8000/docs`.

### 4. Testing the Application

- Use the notebooks for training and evaluating the model:
  - `rnn-ubm-training.ipynb`
  - `rnn-ubm-evaluation.ipynb`
  - `data-processing.ipynb`

- Use the client scripts to test live audio input via WebSockets:
  - `client_verification.py`
  - `client_identification.py`

## API Endpoints and Features

### User Management

- `POST /users/{username}` – Create user  
- `GET /users/` – List all users  
- `PUT /users/{user_id}` – Update username  
- `DELETE /users/{user_id}` – Delete user and enrollment  

### Enrollment (User Registration)

- `PUT /users/enrollments/{user_id}` – Upload audio and create embedding  
- `GET /users/enrollments/{user_id}` – Get enrollment info  
- `PUT /users/enrollments/{user_id}` – Update enrollment with new audio  

### Identification

- `POST /users/enrollments/identification` – Identify user from audio by comparing embeddings  

### Verification

- `POST /users/enrollments/verification/{user_id}` – Verify identity by comparing audio embedding with stored enrollment  

### WebSocket (Real-Time)

- `ws://localhost:8000/ws/verify/{client_id}` – Real-time verification  
- `ws://localhost:8000/ws/identify/{client_id}` – Real-time identification  

## Project Architecture Overview

### Model and Audio Processing

- Audio is split into 1-second chunks
- MFCC features (13 coefficients) are extracted
- Embeddings are created via a pretrained RNN model (optionally passed through LDA)
- Final embeddings are averaged across time

### Server Component

- FastAPI handles HTTP and WebSocket endpoints
- SQLAlchemy and SQLite manage user and enrollment storage
- Audio files are processed temporarily and deleted after use

### WebSocket Clients

- `client_verification.py` sends user ID and audio in chunks to verify identity  
- `client_identification.py` sends audio without ID and receives similarity-based matches



## Conclusion

Project combines machine learning and real-time web technologies to deliver a functional client verification and identification system using audio recordings. It includes tools for training and evaluating models, managing users, performing recognition, and testing in real-time using WebSocket connections.
