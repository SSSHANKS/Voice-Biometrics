import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import joblib


class UBM:
    def __init__(self, pretrained_model, lda_model=None):
        if isinstance(pretrained_model, str):  
            self.model = tf.keras.models.load_model(pretrained_model)
        else:
            self.model = pretrained_model
        
        if lda_model is not None:
            if isinstance(lda_model, str):  
                self.lda = joblib.load(lda_model)
            else:
                self.lda = lda_model
        else:
            self.lda = None

    def extract_embedding(self, X, use_lda=0):
        intermediate_layer_model = tf.keras.Model(inputs=self.model.layers[0].input,
                                                  outputs=self.model.get_layer('bottleneck').output)
        embeddings = intermediate_layer_model.predict(X)
        
        if use_lda and self.lda is not None:
            embeddings = self.lda.transform(embeddings)

        return embeddings


def process_audio_file(file_path, ubm_model, sr=8000, n_mfcc=13, window_size_sec=1, use_lda=False):
    """
    Processes an audio file by splitting it into windows of a specified length, computing MFCCs, and generating embeddings,
    which are averaged over time.
    
    Parameters:
    - file_path (str): Path to the audio file.
    - ubm_model (UBM): UBM model object used for generating embeddings.
    - sr (int): Target sampling rate (default: 8000 Hz).
    - n_mfcc (int): Number of MFCC coefficients (default: 13).
    - window_size_sec (int): Length of each window in seconds (default: 1).
    - use_lda (bool): Whether to use LDA for transforming embeddings (default: False).
    
    Returns:
    - np.ndarray: Averaged embeddings for the entire recording.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr)

        samples_per_window = window_size_sec * sr
        num_windows = len(y) // samples_per_window

        embeddings = []

        for i in range(num_windows):
            start_idx = i * samples_per_window
            end_idx = start_idx + samples_per_window
            window = y[start_idx:end_idx]

            mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=n_mfcc).T

            mfcc_cnn_input = mfcc.reshape(mfcc.shape[0], mfcc.shape[1], 1)

            embedding = ubm_model.extract_embedding(np.array([mfcc_cnn_input]), use_lda=use_lda)
            embeddings.append(embedding)

        if len(embeddings) > 0:
            averaged_embedding = np.mean(embeddings, axis=0)
        else:
            raise ValueError("No valid audio segments found in the file.")

        return averaged_embedding
    
    except Exception as e:
        print(f"Error in processing audio: {e}")
        raise



