import asyncio
import json
import websockets
import numpy as np
import time
import wave

async def data_generator(file_path, sample_rate=8000, num_channels=1, sample_width=2):
    """
    Generator odczytujący dane audio w segmentach odpowiadających 1 sekundzie.

    Parameters:
    - file_path (str): Ścieżka do pliku WAV.
    - sample_rate (int): Częstotliwość próbkowania
    - num_channels (int): Liczba kanałów 
    - sample_width (int): Liczba bajtów na próbkę
    """
    try:
        with wave.open(file_path, "rb") as wav_file:
            # Odczyt parametrów pliku
            #file_sample_rate = sample_rate
            #file_num_channels = wav_file.getnchannels()
            #file_sample_width = wav_file.getsampwidth()

            # Liczba bajtów w sekundzie
            bytes_per_second = sample_rate * num_channels * sample_width

            print(f"Sample rate: {sample_rate}, Channels: {num_channels}, Sample width: {sample_width} bytes")
            print(f"Bytes per second: {bytes_per_second}")

            while True:
                # Odczyt segmentu odpowiadającego 1 sekundzie
                chunk = wav_file.readframes(sample_rate)
                
                # Sprawdzenie długości chunku (musi odpowiadać co najmniej 1 sekundzie)
                if len(chunk) < bytes_per_second:
                    print("Skipping chunk smaller than 1 second.")
                    break  

                if not chunk:
                    break

                yield chunk

    except Exception as e:
        print(f"Error in data_generator: {e}")
        raise

    except Exception as e:
        print(f"Error reading audio file: {e}")

async def websocket_client(user_id: str):
    client_id = str(np.random.randint(0, 1000))  # Randomowy client_id
    uri = f"ws://localhost:8000/ws/verify/{client_id}"  
    gen = data_generator("combined_audio.wav", sample_rate=8000)

    async with websockets.connect(uri) as websocket:
        while True:
            try:
                # Generowanie i wysyłanie danych do serwera
                data = await gen.__anext__()

                user_id_bytes = user_id.encode("utf-8")  # Zakodowanie user_id jako bajtów
                user_id_length = len(user_id_bytes).to_bytes(2, "big")  # Długość user_id w 2 bajtach

                message = user_id_length + user_id_bytes + data

                await websocket.send(message)
                print(f"Sent data of size: {len(data)} bytes")

                
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5)
                    response = json.loads(message)
                    print(f"Received response: {response}")

                except asyncio.TimeoutError:
                    print("Timeout: No response received within 10 seconds")
                    time.sleep(0.5)  

            except Exception as e:
                print(f"Error: {e}")
                break

# Uruchomienie klienta
asyncio.run(websocket_client(user_id="86eb0193-b725-4c45-9c01-8f47c92a9aa9"))