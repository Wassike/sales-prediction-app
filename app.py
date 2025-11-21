# app.py - APPLICATION DE PRÉDICTION DES VENTES AVEC LSTM - VERSION COMPLÈTE
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from preprocessing import load_and_scale, create_sequences
from train_models import train_and_save_model, load_model

# Configuration de la page
st.set_page_config(
    page_title="Système de Prédiction des Ventes",
    page_icon="📈",
    layout="wide"
)

# Initialisation de tous les états de session
def initialize_session_state():
    if 'current_data' not in st.session_state:
        st.session_state['current_data'] = None
    if 'current_file' not in st.session_state:
        st.session_state['current_file'] = None
    if 'scaled_data' not in st.session_state:
        st.session_state['scaled_data'] = None
    if 'sequences' not in st.session_state:
        st.session_state['sequences'] = None
    if 'labels' not in st.session_state:
        st.session_state['labels'] = None
    if 'scaler' not in st.session_state:
        st.session_state['scaler'] = None
    if 'selected_column' not in st.session_state:
        st.session_state['selected_column'] = None
    if 'sequence_length' not in st.session_state:
        st.session_state['sequence_length'] = 30
    if 'trained_model' not in st.session_state:
        st.session_state['trained_model'] = None
    if 'model_type' not in st.session_state:
        st.session_state['model_type'] = None
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = None
    if 'future_predictions' not in st.session_state:
        st.session_state['future_predictions'] = None

# Initialiser les états
initialize_session_state()

# Titre de l'application
st.title("📈 Système de Prédiction des Ventes avec LSTM")
st.markdown("Prédisez les ventes futures grâce à l'intelligence artificielle (LSTM)")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", [
    "Chargement des Données", 
    "Préprocessing", 
    "Entraînement du Modèle", 
    "Prédictions",
    "Visualisation des Résultats"
])

# Section 1: Chargement des Données
if page == "Chargement des Données":
    st.header("📁 Chargement des Données de Ventes")
    
    st.info("""
    **Format attendu :**
    - Fichier CSV avec une colonne de dates
    - Une ou plusieurs colonnes de ventes/chiffre d'affaires
    - Exemple : date, ventes, chiffre_affaires
    """)
    
    # Option 1: Upload de fichier
    uploaded_file = st.file_uploader("Téléchargez votre fichier CSV de ventes", type=["csv"])
    
    # Option 2: Utiliser un fichier existant
    data_files = []
    if os.path.exists("data"):
        data_files = [f for f in os.listdir("data") if f.endswith('.csv')]
    
    selected_file = None
    
    if uploaded_file is not None:
        # Sauvegarder le fichier uploadé
        os.makedirs("data", exist_ok=True)
        with open(f"data/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        selected_file = f"data/{uploaded_file.name}"
        st.success(f"✅ Fichier {uploaded_file.name} téléchargé avec succès!")
    
    elif data_files:
        selected_file = st.selectbox("Ou choisissez un fichier existant:", 
                                   [f"data/{f}" for f in data_files])
    
    if selected_file:
        try:
            # Charger les données
            df = pd.read_csv(selected_file)
            
            # Vérifier s'il y a une colonne de date
            date_columns = df.select_dtypes(include=['object']).columns
            if len(date_columns) > 0:
                # Essayer de convertir la première colonne texte en date
                try:
                    df[date_columns[0]] = pd.to_datetime(df[date_columns[0]])
                    st.success(f"✅ Colonne de date détectée: {date_columns[0]}")
                except:
                    st.warning("ℹ️  Aucune colonne de date détectée, utilisation de l'index comme temps")
            
            st.success(f"✅ Données chargées: {df.shape[0]} périodes, {df.shape[1]} colonnes")
            
            # Aperçu des données
            st.subheader("Aperçu des Données")
            st.dataframe(df.head(10))
            
            # Informations sur les données
            st.subheader("Analyse des Données")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Colonnes Numériques (Ventes):**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    st.write(f"- {col} (moyenne: {df[col].mean():.2f})")
            
            with col2:
                st.write("**Statistiques Globales:**")
                if len(numeric_cols) > 0:
                    total_sales = df[numeric_cols[0]].sum()
                    st.write(f"- Ventes totales: {total_sales:,.0f} €")
                    st.write(f"- Période couverte: {len(df)} jours/mois")
                    st.write(f"- Ventes moyennes: {df[numeric_cols[0]].mean():.2f} €")
                st.write(f"- Valeurs manquantes: {df.isnull().sum().sum()}")
            
            # Visualisation initiale
            if len(numeric_cols) > 0:
                st.subheader("Évolution des Ventes")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Tracer la première colonne numérique (supposée être les ventes)
                sales_col = numeric_cols[0]
                ax.plot(df[sales_col].values, marker='o', linewidth=2, markersize=4)
                ax.set_title(f'Évolution des {sales_col}')
                ax.set_xlabel('Période')
                ax.set_ylabel('Ventes')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
            
            # Sauvegarder dans session state
            st.session_state['current_file'] = selected_file
            st.session_state['current_data'] = df
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement: {e}")

# Section 2: Préprocessing
elif page == "Préprocessing":
    st.header("🔧 Préparation des Données pour la Prédiction")
    
    if st.session_state['current_data'] is None:
        st.warning("⚠️ Veuillez d'abord charger des données dans l'onglet 'Chargement des Données'")
    else:
        df = st.session_state['current_data']
        
        # Sélection de la colonne des ventes
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("❌ Aucune colonne numérique trouvée dans les données!")
        else:
            col = st.selectbox("Sélectionnez la colonne des ventes à prédire:", numeric_cols)
            
            # Paramètres de séquençage
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sequence_length = st.slider("Fenêtre temporelle (jours):", 
                                          min_value=7, max_value=90, value=30,
                                          help="Nombre de périodes passées utilisées pour prédire la suivante")
            
            with col2:
                test_size = st.slider("Pourcentage de test:", 
                                    min_value=0.1, max_value=0.4, value=0.2,
                                    help="Pourcentage des données utilisées pour tester le modèle")
            
            with col3:
                future_steps = st.slider("Jours à prédire:", 
                                       min_value=1, max_value=30, value=7,
                                       help="Nombre de jours dans le futur à prédire")
            
            if st.button("🔧 Préparer les Données pour l'IA"):
                with st.spinner("Préprocessing des données de ventes..."):
                    try:
                        # Utiliser load_and_scale avec les deux arguments
                        scaled_data, scaler, original_data = load_and_scale(df, col)
                        
                        if scaled_data is not None:
                            # Créer les séquences
                            sequences = create_sequences(scaled_data, sequence_length)
                            
                            if sequences is not None:
                                # Préparer les labels (valeur suivante)
                                if len(scaled_data.shape) == 1:
                                    labels = scaled_data[sequence_length:]
                                else:
                                    labels = scaled_data[sequence_length:, 0]
                                
                                # Sauvegarder dans session state
                                st.session_state['scaled_data'] = scaled_data
                                st.session_state['sequences'] = sequences
                                st.session_state['labels'] = labels
                                st.session_state['scaler'] = scaler
                                st.session_state['selected_column'] = col
                                st.session_state['sequence_length'] = sequence_length
                                st.session_state['future_steps'] = future_steps
                                
                                st.success("✅ Données préparées avec succès pour l'IA!")
                                
                                # Afficher les résultats
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Périodes historiques", f"{df.shape[0]}")
                                with col2:
                                    st.metric("Séquences d'entraînement", f"{sequences.shape[0]}")
                                with col3:
                                    st.metric("Fenêtre temporelle", f"{sequence_length}j")
                                with col4:
                                    st.metric("Jours à prédire", f"{future_steps}j")
                                
                                # Visualisation des données préparées
                                st.subheader("Données Préparées pour l'IA")
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                                
                                # Données originales
                                ax1.plot(df[col].values, label='Ventes réelles', color='blue', alpha=0.7)
                                ax1.set_title('Ventes Originales')
                                ax1.set_xlabel('Période')
                                ax1.set_ylabel('Ventes')
                                ax1.legend()
                                ax1.grid(True, alpha=0.3)
                                
                                # Données normalisées
                                ax2.plot(scaled_data, label='Ventes normalisées', color='green', alpha=0.7)
                                ax2.set_title('Ventes Normalisées (0-1)')
                                ax2.set_xlabel('Période')
                                ax2.set_ylabel('Ventes normalisées')
                                ax2.legend()
                                ax2.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                                
                            else:
                                st.error("❌ Erreur lors de la création des séquences")
                        else:
                            st.error("❌ Erreur lors de la normalisation des données")
                            
                    except Exception as e:
                        st.error(f"❌ Erreur lors du préprocessing: {e}")

# Section 3: Entraînement du Modèle LSTM
elif page == "Entraînement du Modèle":
    st.header("🤖 Entraînement du Modèle LSTM de Prédiction")
    
    if st.session_state['sequences'] is None:
        st.warning("⚠️ Veuillez d'abord préparer les données dans l'onglet 'Préprocessing'")
    else:
        sequences = st.session_state['sequences']
        labels = st.session_state['labels']
        selected_column = st.session_state.get('selected_column', 0)
        
        st.info("""
        **Le modèle LSTM va apprendre les patterns de vos ventes :**
        - Saisonnalité (quotidienne, hebdomadaire, mensuelle)
        - Tendances (croissance, décroissance)
        - Comportements cycliques
        """)
        
        # Paramètres d'entraînement
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Nombre d'époques d'entraînement:", 
                             min_value=10, max_value=200, value=50,
                             help="Plus d'époques = meilleure précision mais plus long")
            
            model_type = st.selectbox("Type de modèle:", ["lstm"], 
                                    help="LSTM est spécialisé pour les séries temporelles")
        
        with col2:
            batch_size = st.slider("Taille du lot:", 
                                 min_value=8, max_value=64, value=32,
                                 help="Nombre de séquences traitées simultanément")
            
            learning_rate = st.selectbox("Vitesse d'apprentissage:", 
                                      [0.001, 0.01, 0.1], 
                                      index=0,
                                      help="Vitesse à laquelle le modèle apprend")
        
        if st.button("🚀 Démarrer l'Entraînement du Modèle"):
            with st.spinner("Entraînement du modèle LSTM en cours... Cela peut prendre quelques minutes"):
                try:
                    # Préparer les données pour l'entraînement
                    split_index = int(len(sequences) * (1 - 0.2))  # 20% pour le test
                    X_train, X_test = sequences[:split_index], sequences[split_index:]
                    y_train, y_test = labels[:split_index], labels[split_index:]
                    
                    # Entraîner le modèle
                    model, scaler = train_and_save_model(
                        st.session_state['current_file'],
                        selected_column,
                        st.session_state['sequence_length'],
                        epochs,
                        batch_size,
                        model_type
                    )
                    
                    if model is not None:
                        st.session_state['trained_model'] = model
                        st.session_state['model_type'] = model_type
                        
                        # Faire des prédictions sur le set de test pour évaluation
                        test_predictions = model.predict(X_test)
                        
                        # Calculer l'erreur
                        from sklearn.metrics import mean_absolute_error, mean_squared_error
                        mae = mean_absolute_error(y_test, test_predictions.flatten())
                        rmse = np.sqrt(mean_squared_error(y_test, test_predictions.flatten()))
                        
                        st.success("✅ Modèle LSTM entraîné avec succès!")
                        
                        # Afficher les métriques
                        st.subheader("📊 Performance du Modèle")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Données d'entraînement", f"{X_train.shape[0]} séq.")
                        with col2:
                            st.metric("Données de test", f"{X_test.shape[0]} séq.")
                        with col3:
                            st.metric("MAE", f"{mae:.4f}")
                        with col4:
                            st.metric("RMSE", f"{rmse:.4f}")
                        
                        # Visualisation des prédictions vs réalité
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Tracer les vraies valeurs et prédictions
                        ax.plot(y_test, label='Ventes Réelles', color='blue', alpha=0.7, linewidth=2)
                        ax.plot(test_predictions.flatten(), label='Prédictions LSTM', color='red', 
                               alpha=0.7, linestyle='--', linewidth=2)
                        
                        ax.set_title('Comparaison Prédictions vs Réalité (Set de Test)')
                        ax.set_xlabel('Période')
                        ax.set_ylabel('Ventes Normalisées')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'entraînement: {e}")

# Section 4: Prédictions Futures
elif page == "Prédictions":
    st.header("🔮 Prédictions des Ventes Futures")
    
    if st.session_state['trained_model'] is None:
        st.warning("⚠️ Veuillez d'abord entraîner un modèle dans l'onglet 'Entraînement du Modèle'")
    else:
        model = st.session_state['trained_model']
        scaled_data = st.session_state['scaled_data']
        scaler = st.session_state['scaler']
        sequence_length = st.session_state['sequence_length']
        future_steps = st.session_state.get('future_steps', 7)
        
        st.info(f"**Préparation de la prédiction des {future_steps} prochains jours**")
        
        if st.button("🎯 Générer les Prédictions"):
            with st.spinner("Génération des prédictions futures..."):
                try:
                    # Utiliser les dernières séquences pour prédire le futur
                    last_sequence = scaled_data[-sequence_length:]
                    
                    # Faire des prédictions pas à pas
                    future_predictions = []
                    current_sequence = last_sequence.copy()
                    
                    for _ in range(future_steps):
                        # Préparer la séquence pour la prédiction
                        seq_reshaped = current_sequence.reshape(1, sequence_length, 1)
                        
                        # Prédire la prochaine valeur
                        next_pred = model.predict(seq_reshaped)[0, 0]
                        future_predictions.append(next_pred)
                        
                        # Mettre à jour la séquence
                        current_sequence = np.append(current_sequence[1:], next_pred)
                    
                    # Convertir les prédictions à l'échelle originale
                    future_predictions = np.array(future_predictions).reshape(-1, 1)
                    future_predictions_original = scaler.inverse_transform(future_predictions)
                    
                    # Sauvegarder les prédictions
                    st.session_state['future_predictions'] = future_predictions_original.flatten()
                    
                    st.success(f"✅ Prédictions générées pour les {future_steps} prochains jours!")
                    
                    # Afficher les prédictions
                    st.subheader("📋 Prédictions Détaillées")
                    
                    # Créer un DataFrame pour les prédictions
                    today = datetime.now()
                    future_dates = [today + timedelta(days=i) for i in range(1, future_steps + 1)]
                    
                    predictions_df = pd.DataFrame({
                        'Date': future_dates,
                        'Ventes Prédites': future_predictions_original.flatten()
                    })
                    
                    st.dataframe(predictions_df.style.format({
                        'Ventes Prédites': '{:,.0f} €'
                    }))
                    
                    # Statistiques des prédictions
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Ventes moyennes prédites", 
                                f"{future_predictions_original.mean():.0f} €")
                    with col2:
                        st.metric("Ventes totales prédites", 
                                f"{future_predictions_original.sum():.0f} €")
                    with col3:
                        st.metric("Période de prédiction", f"{future_steps} jours")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction: {e}")

# Section 5: Visualisation des Résultats
elif page == "Visualisation des Résultats":
    st.header("📊 Analyse et Visualisation des Prédictions")
    
    if st.session_state.get('future_predictions') is None:
        st.warning("⚠️ Veuillez d'abord générer des prédictions dans l'onglet 'Prédictions'")
    else:
        future_predictions = st.session_state['future_predictions']
        original_data = st.session_state['current_data']
        selected_column = st.session_state['selected_column']
        future_steps = st.session_state.get('future_steps', 7)
        
        # Créer les dates pour les prédictions
        last_date = datetime.now()
        future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps + 1)]
        
        st.subheader("📈 Évolution Historique et Prédictions Futures")
        
        # Créer le graphique complet
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Tracer les données historiques
        historical_dates = range(len(original_data))
        ax.plot(historical_dates, original_data[selected_column].values, 
               label='Ventes Historiques', color='blue', linewidth=2, marker='o')
        
        # Tracer les prédictions futures
        future_indices = range(len(original_data), len(original_data) + future_steps)
        ax.plot(future_indices, future_predictions, 
               label='Prédictions Futures', color='red', linewidth=2, marker='s', linestyle='--')
        
        # Zone de confiance (simulée)
        confidence_upper = future_predictions * 1.1  # +10%
        confidence_lower = future_predictions * 0.9  # -10%
        ax.fill_between(future_indices, confidence_lower, confidence_upper, 
                       alpha=0.2, color='red', label='Intervalle de Confiance')
        
        ax.set_title('Prédictions des Ventes - Historique et Futur')
        ax.set_xlabel('Période')
        ax.set_ylabel('Ventes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Améliorer la lisibilité
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Analyse des tendances
        st.subheader("📋 Analyse des Tendances")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Croissance des prédictions
            growth = ((future_predictions[-1] - future_predictions[0]) / future_predictions[0]) * 100
            trend = "📈 Hausse" if growth > 0 else "📉 Baisse"
            st.metric("Tendance générale", f"{trend}", f"{growth:.1f}%")
            
            # Volatilité
            volatility = np.std(future_predictions) / np.mean(future_predictions) * 100
            st.metric("Volatilité prédite", f"{volatility:.1f}%")
        
        with col2:
            # Meilleur jour
            best_day_idx = np.argmax(future_predictions)
            best_day_sales = future_predictions[best_day_idx]
            st.metric("Pic de ventes prédit", f"{best_day_sales:.0f} €", f"Jour {best_day_idx + 1}")
            
            # Ventes moyennes
            avg_sales = np.mean(future_predictions)
            st.metric("Ventes moyennes prédites", f"{avg_sales:.0f} €")
        
        # Recommandations
        st.subheader("💡 Recommandations Commerciales")
        
        if growth > 5:
            st.success("**🎯 Opportunité :** Tendance haussière détectée ! Pensez à augmenter les stocks et le marketing.")
        elif growth < -5:
            st.warning("**⚠️ Alerte :** Tendance baissière. Revoyez votre stratégie commerciale.")
        else:
            st.info("**ℹ️ Stabilité :** Tendance stable. Maintenez votre stratégie actuelle.")
        
        # Téléchargement des prédictions
        st.subheader("📥 Export des Prédictions")
        
        # Créer un DataFrame complet avec historique et prédictions
        historical_df = pd.DataFrame({
            'Date': [f'Période {i+1}' for i in range(len(original_data))],
            'Type': 'Historique',
            'Ventes': original_data[selected_column].values
        })
        
        predictions_df = pd.DataFrame({
            'Date': [f'Jour {i+1}' for i in range(len(future_predictions))],
            'Type': 'Prédiction',
            'Ventes': future_predictions
        })
        
        full_results = pd.concat([historical_df, predictions_df], ignore_index=True)
        
        # Bouton de téléchargement
        csv = full_results.to_csv(index=False)
        st.download_button(
            label="📊 Télécharger toutes les données (CSV)",
            data=csv,
            file_name="predictions_ventes_completes.csv",
            mime="text/csv"
        )
        
        # Bouton pour télécharger seulement les prédictions
        predictions_only = predictions_df.to_csv(index=False)
        st.download_button(
            label="🔮 Télécharger les prédictions seulement (CSV)",
            data=predictions_only,
            file_name="predictions_ventes_futures.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("📈 Système de Prédiction des Ventes - Développé avec Streamlit et LSTM")
st.markdown("*Utilise l'intelligence artificielle pour anticiper vos ventes futures*")


