# train_models.py - Version complète et corrigée
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from preprocessing import load_and_scale, create_sequences

def train_and_save_model(file_path, column_index=0, sequence_length=50, epochs=10, batch_size=32, model_type='lstm', model_path='models/'):
    """
    Entraîne et sauvegarde un modèle - Version compatible avec votre app.py
    
    Parameters (deux modes supportés):
    - Mode 1: train_and_save_model(X_train, y_train, X_test, y_test, model_type, model_path)
    - Mode 2: train_and_save_model(file_path, column_index, sequence_length, epochs, batch_size)
    """
    
    print(f"🔧 Début de l'entraînement avec model_type: {model_type}")
    
    # Détection automatique du mode d'appel
    if isinstance(file_path, (np.ndarray, pd.DataFrame)) and not isinstance(file_path, str):
        # Mode 1: Données déjà préparées (X_train, y_train, X_test, y_test)
        return _train_with_prepared_data(file_path, column_index, sequence_length, epochs, batch_size, model_type, model_path)
    else:
        # Mode 2: Chemin de fichier (file_path, column_index, etc.)
        return _train_from_file(file_path, column_index, sequence_length, epochs, batch_size, model_type, model_path)

def _train_from_file(file_path, column_index, sequence_length, epochs, batch_size, model_type, model_path):
    """Entraîne à partir d'un fichier CSV - Version corrigée"""
    print(f"📂 Chargement des données depuis: {file_path}")
    
    # DEBUG: Afficher les paramètres
    print(f"🔍 DEBUG - column_index: {column_index}, type: {type(column_index)}")
    print(f"🔍 DEBUG - sequence_length: {sequence_length}")
    print(f"🔍 DEBUG - epochs: {epochs}")
    print(f"🔍 DEBUG - batch_size: {batch_size}")
    print(f"🔍 DEBUG - model_type: {model_type}")
    
    # Convertir column_index en entier de manière sécurisée
    try:
        col_idx = int(column_index)
        print(f"✅ column_index converti en: {col_idx}")
    except (ValueError, TypeError) as e:
        print(f"⚠️  Erreur conversion column_index: {e}, utilisation de 0")
        col_idx = 0
    
    # Charger et préparer les données
    scaled_data, scaler, original_data = load_and_scale(file_path)
    
    if scaled_data is None:
        raise ValueError("❌ Impossible de charger les données")
    
    print(f"🔍 DEBUG - Forme de scaled_data: {scaled_data.shape}")
    
    # Créer les séquences
    sequences = create_sequences(scaled_data, sequence_length)
    
    if sequences is None:
        raise ValueError("❌ Impossible de créer les séquences")
    
    # PRÉPARER LES LABELS - Version robuste
    X = sequences
    
    # Gérer la préparation des labels selon la forme des données
    if len(scaled_data.shape) == 1:
        # Données 1D
        y = scaled_data[sequence_length:]
        print("📊 Utilisation des données 1D pour les labels")
    else:
        # Données 2D - vérifier que l'index est valide
        if col_idx >= scaled_data.shape[1]:
            print(f"⚠️  column_index {col_idx} hors limites (max: {scaled_data.shape[1]-1}), utilisation de 0")
            col_idx = 0
        y = scaled_data[sequence_length:, col_idx]
        print(f"📊 Utilisation de la colonne {col_idx} pour les labels")
    
    print(f"📊 Données préparées - X: {X.shape}, y: {y.shape}")
    
    # Vérifier que X et y ont la même longueur
    if len(X) != len(y):
        print(f"⚠️  Ajustement des dimensions: X({len(X)}) != y({len(y)})")
        min_length = min(len(X), len(y))
        X = X[:min_length]
        y = y[:min_length]
        print(f"✅ Dimensions ajustées - X: {X.shape}, y: {y.shape}")
    
    # Diviser en train/test
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"🎯 Division train/test - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Créer le dossier models
    os.makedirs(model_path, exist_ok=True)
    
    # Entraîner le modèle
    if model_type == 'lstm':
        model, history = _train_lstm(X_train, y_train, X_test, y_test, epochs, batch_size, model_path)
        return model, scaler
    elif model_type == 'random_forest':
        model, accuracy = _train_random_forest(X_train, y_train, X_test, y_test, model_path)
        return model, scaler
    else:
        raise ValueError("❌ model_type doit être 'lstm' ou 'random_forest'")

def _train_with_prepared_data(X_train, y_train, X_test, y_test, model_type, model_path):
    """Entraîne avec des données déjà préparées"""
    print("🔧 Utilisation des données préparées")
    
    # Créer le dossier models
    os.makedirs(model_path, exist_ok=True)
    
    if model_type == 'lstm':
        model, history = _train_lstm(X_train, y_train, X_test, y_test, 10, 32, model_path)
        return model, None
    elif model_type == 'random_forest':
        model, accuracy = _train_random_forest(X_train, y_train, X_test, y_test, model_path)
        return model, None
    else:
        raise ValueError("❌ model_type doit être 'lstm' ou 'random_forest'")

def _train_lstm(X_train, y_train, X_test, y_test, epochs, batch_size, model_path):
    """Entraîne un modèle LSTM"""
    print("🔮 Entraînement du modèle LSTM...")
    
    # S'assurer que les données sont au bon format
    if len(X_train.shape) == 2:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    print(f"🔧 Forme des données LSTM - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Créer le modèle LSTM
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)  # Régression (une valeur de sortie)
    ])
    
    # Compiler le modèle
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"🔧 Début de l'entraînement LSTM - Epochs: {epochs}, Batch size: {batch_size}")
    
    # Entraîner le modèle
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Sauvegarder le modèle
    model_path_full = os.path.join(model_path, 'lstm_model.h5')
    model.save(model_path_full)
    print(f"✅ Modèle LSTM sauvegardé: {model_path_full}")
    
    return model, history

def _train_random_forest(X_train, y_train, X_test, y_test, model_path):
    """Entraîne un modèle Random Forest"""
    print("🌲 Entraînement du modèle Random Forest...")
    
    # Reshape pour Random Forest
    if len(X_train.shape) == 3:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        print(f"🔧 Données reshapées - X_train: {X_train_flat.shape}, X_test: {X_test_flat.shape}")
    else:
        X_train_flat = X_train
        X_test_flat = X_test
    
    # Créer et entraîner le modèle
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    rf_model.fit(X_train_flat, y_train)
    
    # Faire des prédictions
    y_pred = rf_model.predict(X_test_flat)
    
    # Calculer la précision
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Précision Random Forest: {accuracy:.4f}")
    
    # Sauvegarder le modèle
    model_path_full = os.path.join(model_path, 'random_forest_model.pkl')
    joblib.dump(rf_model, model_path_full)
    print(f"✅ Modèle Random Forest sauvegardé: {model_path_full}")
    
    return rf_model, accuracy

def load_model(model_type='lstm', model_path='models/'):
    """Charge un modèle pré-entraîné"""
    try:
        if model_type == 'lstm':
            model_path_full = os.path.join(model_path, 'lstm_model.h5')
            model = tf.keras.models.load_model(model_path_full)
        elif model_type == 'random_forest':
            model_path_full = os.path.join(model_path, 'random_forest_model.pkl')
            model = joblib.load(model_path_full)
        else:
            raise ValueError("❌ model_type doit être 'lstm' ou 'random_forest'")
        
        print(f"✅ Modèle {model_type} chargé avec succès: {model_path_full}")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

# Test du module
if __name__ == "__main__":
    print("🧪 Test du module train_models...")
    
    # Créer un fichier de test simple
    try:
        import csv
        os.makedirs('data', exist_ok=True)
        with open('data/test.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['value'])
            for i in range(100):
                writer.writerow([i * 0.1])
        print("✅ Fichier de test créé: data/test.csv")
        
        # Test avec le fichier
        model, scaler = train_and_save_model(
            "data/test.csv",
            column_index=0,
            sequence_length=10,
            epochs=2,
            batch_size=16,
            model_type='lstm'
        )
        print("✅ Test réussi!")
    except Exception as e:

        print(f"❌ Test échoué: {e}")
