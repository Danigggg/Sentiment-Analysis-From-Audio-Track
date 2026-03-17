#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:18:24 2026

@author: denisye
"""

import librosa
import numpy as np
from config import SAMPLE_RATE, TARGET_DURATION

def load_and_pad_audio(file_path):
    
    #  Caricamento del segnale audio
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    #rimozione silenzi
    signal_trimmed, _ = librosa.effects.trim(signal, top_db=30)
    
    
    # 3Calcolo dei campioni per la lunghezza target
    target_length = int(TARGET_DURATION * SAMPLE_RATE)
    
    # 4. Troncamento o Padding
    if len(signal_trimmed) > target_length:
        # Se anche senza silenzio l'audio è troppo lungo, lo tronchiamo
        signal_final = signal_trimmed[:target_length]
        
    elif len(signal_trimmed) < target_length:
        # Se è più corto, calcoliamo quanto silenzio manca
        padding_length = target_length - len(signal_trimmed)
        
        # Centriamo l'audio distribuendo il padding a sinistra e a destra
        pad_left = padding_length // 2
        pad_right = padding_length - pad_left
        
        signal_final = np.pad(signal_trimmed, (pad_left, pad_right), mode='constant')
        
    else:
        signal_final = signal_trimmed
        
    return signal_final, sr