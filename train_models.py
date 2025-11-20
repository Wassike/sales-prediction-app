# train_models.py - Version PyTorch CORRIGÉE avec méthode predict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import os
from preprocessing import load_and_scale, create_sequences

# Définition du modèle LSTM avec PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialisation des états cachés
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Passage through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # On prend seulement le dernier output
        out = self.fc(out[:, -1, :])
        return out
    
    def predict(self, X):
        """Méthode predict compatible avec l'interface de scikit-learn"""
        self.eval()  # Mode évaluation
        with torch.no_grad():
            # Conversion en tenseur PyTorch
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X)
            else:
                X_tensor = X
            
            # Prédiction
            predictions = self.forward(X_tensor)
            return predictions.numpy()

def train_and_save_model(file_path, column_index=0, sequence_length=50, epochs=10, batch_size=32, model_type='lstm', model_path='models/'):
    """
    Entraîne et sauvegarde un modèle - Version compatible avec votre app.py
    """
    
    print(f"🔧 Début de l'entraînement avec model_type: {model_type}")
    
    # Détection automatique du mode d'appel
    if isinstance(file_path, (np.ndarray, pd.DataFrame)) and not isinstance(file_path, str):
        return _train_with_prepared_data(file_path, column_index, sequence_length, epochs, batch_size, model_type, model_path)
    else:
        return _train_from_file(file_path, column_index, sequence_length, epochs, batch_size, model_type, model_path)

def _train_from_file(file_path, column_index, sequence_length, epochs, batch_size, model_type, model_path):
    """Entraîne à partir d'un fichier CSV"""
    print(f"📂 Chargement des données depuis: {file_path}")
    
    # Gestion améliorée de column_index
    if isinstance(column_index, str):
        # C'est un nom de colonne, on le garde tel quel
        col_name = column_index
    else:
        try:
            col_name = str(column_index)
        except (ValueError, TypeError):
            col_name = "0"
    
    # Charger et préparer les données
    scaled_data, scaler, original_data = load_and_scale(file_path)
    
    if scaled_data is None:
        raise ValueError("❌ Impossible de charger les données")
    
    print(f"🔍 DEBUG - Forme de scaled_data: {scaled_data.shape}")
    
    # Créer les séquences
    sequences = create_sequences(scaled_data, sequence_length)
    
    if sequences is None:
        raise ValueError("❌ Impossible de créer les séquences")
    
    # PRÉPARER LES LABELS
    X = sequences
    
    if len(scaled_data.shape) == 1:
        y = scaled_data[sequence_length:]
    else:
        # Pour les données 2D, utiliser la première colonne
        y = scaled_data[sequence_length:, 0]
    
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
        model, history = _train_lstm_pytorch(X_train, y_train, X_test, y_test, epochs, batch_size, model_path)
        return model, scaler
    elif model_type == 'random_forest':
        model, accuracy = _train_random_forest(X_train, y_train, X_test, y_test, model_path)
        return model, scaler
    else:
        raise ValueError("❌ model_type doit être 'lstm' ou 'random_forest'")

def _train_lstm_pytorch(X_train, y_train, X_test, y_test, epochs, batch_size, model_path):
    """Entraîne un modèle LSTM avec PyTorch"""
    print("🔮 Entraînement du modèle LSTM (PyTorch)...")
    
    # Conversion en tenseurs PyTorch
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    # Création du modèle
    model = LSTMModel(input_size=X_train.shape[2], hidden_size=50, num_layers=2, output_size=1)
    
    # Définition de la loss et de l'optimiseur
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Entraînement
    train_losses = []
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch training
        for i in range(0, len(X_train_tensor), batch_size):
            # Get mini-batch
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Calcul de la loss sur l'ensemble d'entraînement
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            train_losses.append(train_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'📈 Epoch [{epoch+1}/{epochs}], Loss: {train_loss.item():.4f}')
    
    # Évaluation sur le set de test
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        mae = mean_absolute_error(y_test_tensor.numpy(), test_outputs.numpy())
    
    print(f"✅ Entraînement terminé - Test Loss: {test_loss.item():.4f}, MAE: {mae:.4f}")
    
    # Sauvegarde du modèle
    model_path_full = os.path.join(model_path, 'lstm_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': X_train.shape[2],
        'hidden_size': 50,
        'num_layers': 2
    }, model_path_full)
    
    print(f"✅ Modèle LSTM sauvegardé: {model_path_full}")
    
    return model, train_losses

def _train_random_forest(X_train, y_train, X_test, y_test, model_path):
    """Entraîne un modèle Random Forest"""
    print("🌲 Entraînement du modèle Random Forest...")
    
    # Reshape pour Random Forest
    if len(X_train.shape) == 3:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
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
            model_path_full = os.path.join(model_path, 'lstm_model.pth')
            checkpoint = torch.load(model_path_full, map_location=torch.device('cpu'))
            model = LSTMModel(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
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
    
    # Test avec des données synthétiques
    try:
        # Créer des données de test
        X_test = np.random.random((100, 30, 1))
        y_test = np.random.random((100,))
        
        model, losses = _train_lstm_pytorch(X_test, y_test, X_test, y_test, 5, 16, 'models/')
        
        # Test de la méthode predict
        predictions = model.predict(X_test)
        print(f"✅ Test PyTorch réussi! Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"❌ Test échoué: {e}")
