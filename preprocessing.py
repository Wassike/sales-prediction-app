# preprocessing.py - Version corrigée avec gestion robuste des index
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Charge les données depuis un fichier CSV"""
    try:
        data = pd.read_csv(file_path)
        print(f"✅ Données chargées : {data.shape[0]} lignes, {data.shape[1]} colonnes")
        return data
    except Exception as e:
        print(f"❌ Erreur chargement : {e}")
        return None

def scale_data(data):
    """Normalise les données entre 0 et 1"""
    if data is None:
        return None, None
    
    scaler = MinMaxScaler()
    
    # Gérer différents types de données
    if isinstance(data, pd.DataFrame):
        # Sélectionner uniquement les colonnes numériques
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            print("❌ Aucune colonne numérique trouvée")
            return None, None
        scaled_data = scaler.fit_transform(numeric_data)
    else:
        # Tableau numpy
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    print("✅ Données normalisées")
    return scaled_data, scaler

def load_and_scale(file_path_or_data, column_index=None):
    """
    Charge ET normalise les données
    Supporte deux modes:
    - load_and_scale("fichier.csv") 
    - load_and_scale(dataframe, column_index)
    """
    print(f"📂 Chargement et normalisation...")
    
    # Mode 1: Si c'est un string, c'est un chemin de fichier
    if isinstance(file_path_or_data, str):
        data = load_data(file_path_or_data)
        if data is not None:
            scaled_data, scaler = scale_data(data)
            return scaled_data, scaler, data
        else:
            return None, None, None
    
    # Mode 2: Si c'est un DataFrame ou array
    elif isinstance(file_path_or_data, (pd.DataFrame, np.ndarray)):
        data = file_path_or_data
        
        # DEBUG: Afficher les informations sur les données
        print(f"🔍 DEBUG - Type de données: {type(data)}")
        if isinstance(data, pd.DataFrame):
            print(f"🔍 DEBUG - Colonnes: {list(data.columns)}")
            print(f"🔍 DEBUG - Forme: {data.shape}")
        else:
            print(f"🔍 DEBUG - Forme: {data.shape}")
        print(f"🔍 DEBUG - Column_index reçu: {column_index}")
        
        # Si un column_index est spécifié et que c'est un DataFrame
        if column_index is not None and isinstance(data, pd.DataFrame):
            # Vérifier si column_index est un nom de colonne
            if column_index in data.columns:
                print(f"🔍 DEBUG - Utilisation de la colonne par nom: {column_index}")
                data_to_scale = data[[column_index]]
            else:
                # Vérifier si column_index est un index numérique valide
                try:
                    col_idx = int(column_index)
                    if 0 <= col_idx < len(data.columns):
                        print(f"🔍 DEBUG - Utilisation de la colonne par index: {col_idx}")
                        col_name = data.columns[col_idx]
                        data_to_scale = data[[col_name]]
                    else:
                        print(f"⚠️  Index {col_idx} hors limites (0-{len(data.columns)-1}), utilisation de la première colonne")
                        data_to_scale = data[[data.columns[0]]]
                except (ValueError, TypeError):
                    print(f"⚠️  Column_index '{column_index}' invalide, utilisation de la première colonne")
                    data_to_scale = data[[data.columns[0]]]
        else:
            # Utiliser toutes les colonnes numériques
            if isinstance(data, pd.DataFrame):
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    print("❌ Aucune colonne numérique trouvée")
                    return None, None, None
                data_to_scale = data[numeric_cols]
                print(f"🔍 DEBUG - Utilisation de toutes les colonnes numériques: {list(numeric_cols)}")
            else:
                data_to_scale = data
        
        scaled_data, scaler = scale_data(data_to_scale)
        
        # DEBUG: Afficher la forme finale
        if scaled_data is not None:
            print(f"🔍 DEBUG - Forme finale scaled_data: {scaled_data.shape}")
        
        return scaled_data, scaler, data
    
    else:
        print(f"❌ Type de données non supporté: {type(file_path_or_data)}")
        return None, None, None

def create_sequences(data, sequence_length=50):
    """Crée des séquences pour l'entraînement LSTM"""
    if data is None:
        return None
    
    # S'assurer que les données sont 2D pour le traitement
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:(i + sequence_length)])
    
    sequences = np.array(sequences)
    print(f"🔢 Séquences créées : {sequences.shape}")
    return sequences

def split_data(sequences, labels, test_size=0.2):
    """Divise les données en ensembles d'entraînement et de test"""
    if sequences is None or labels is None:
        return None, None, None, None
    
    split_index = int(len(sequences) * (1 - test_size))
    
    X_train = sequences[:split_index]
    X_test = sequences[split_index:]
    y_train = labels[:split_index]
    y_test = labels[split_index:]
    
    print(f"📊 Données divisées - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def prepare_labels_for_sequences(data, sequence_length=50):
    """Prépare les labels pour les séquences"""
    if data is None:
        return None
    
    # S'assurer que les données sont 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    labels = data[sequence_length:]
    print(f"🏷️ Labels préparés : {labels.shape}")
    return labels