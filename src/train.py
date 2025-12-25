"""
Script d'entraînement du modèle de reconnaissance d'écriture manuscrite
"""

import os
import time
import matplotlib.pyplot as plt
from data_loader import prepare_mnist_dataset
from model import create_cnn_model, compile_model, get_model_summary, setup_callbacks


def plot_training_history(history, save_path='../examples/training_history.png'):
    """
    Visualise l'évolution de l'accuracy et de la loss pendant l'entraînement
    Permet de voir si le modèle apprend bien ou s'il y a de l'overfitting
    """
    print("\nCréation des graphiques d'entraînement...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Graphique de l'accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    ax1.set_title('Évolution de l\'Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Graphique de la loss
    ax2.plot(history.history['loss'], label='Train Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Val Loss', marker='s')
    ax2.set_title('Évolution de la Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Graphiques sauvegardés dans '{save_path}'")


def train_model(model, X_train, y_train, X_test, y_test, 
                epochs=10, batch_size=128, callbacks=None):
    """
    Entraîne le modèle sur le dataset MNIST
    
    Paramètres:
    - epochs: nombre de passages complets sur le dataset
    - batch_size: nombre d'images traitées en même temps
    """
    print("\n" + "=" * 60)
    print("DÉMARRAGE DE L'ENTRAÎNEMENT")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Images d'entraînement: {len(X_train)}")
    print(f"Images de validation: {len(X_test)}")
    print("=" * 60)
    
    # Démarrer le chronomètre
    start_time = time.time()
    
    # Entraîner le modèle
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Calculer le temps total
    training_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"✓ ENTRAÎNEMENT TERMINÉ en {training_time:.1f} secondes")
    print("=" * 60)
    
    return history


def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances finales du modèle sur le dataset de test
    """
    print("\n" + "=" * 60)
    print("ÉVALUATION FINALE DU MODÈLE")
    print("=" * 60)
    
    # Évaluer sur le dataset de test
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"✓ Loss sur test: {test_loss:.4f}")
    print(f"✓ Accuracy sur test: {test_accuracy*100:.2f}%")
    print("=" * 60)
    
    return test_loss, test_accuracy


def save_final_model(model, save_path='../models/mnist_cnn_final.keras'):
    """
    Sauvegarde le modèle final entraîné
    """
    print(f"\nSauvegarde du modèle final dans '{save_path}'...")
    
    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Sauvegarder le modèle
    model.save(save_path)
    
    print("✓ Modèle sauvegardé avec succès")


# Pipeline complet d'entraînement
def main(use_simple_model=False, epochs=10, batch_size=128):
    """
    Pipeline complet d'entraînement du modèle
    """
    print("=" * 70)
    print("ENTRAÎNEMENT DU MODÈLE DE RECONNAISSANCE D'ÉCRITURE MANUSCRITE")
    print("=" * 70)
    
    # 1. Charger et préparer les données
    X_train, y_train, X_test, y_test = prepare_mnist_dataset(show_samples=True)
    
    # 2. Créer et compiler le modèle
    if use_simple_model:
        from model import create_simple_model
        model = create_simple_model()
        model_name = "simple"
    else:
        model = create_cnn_model()
        model_name = "full"
    
    model = compile_model(model)
    get_model_summary(model)
    
    # 3. Configurer les callbacks
    callbacks = setup_callbacks(f'../models/best_model_{model_name}.keras')
    
    # 4. Entraîner le modèle
    history = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # 5. Visualiser l'entraînement
    plot_training_history(history, f'../examples/training_history_{model_name}.png')
    
    # 6. Évaluer le modèle
    test_loss, test_accuracy = evaluate_model(model, X_test, y_test)
    
    # 7. Sauvegarder le modèle final
    save_final_model(model, f'../models/mnist_cnn_{model_name}_final.keras')
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE D'ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print(f"✅ Accuracy finale: {test_accuracy*100:.2f}%")
    print("=" * 70)
    
    return model, history


# Exécution du script
if __name__ == "__main__":
    import sys
    
    # Option pour entraîner le modèle simple (rapide) ou complet
    use_simple = '--simple' in sys.argv
    
    if use_simple:
        print("Mode: Modèle simple (entraînement rapide)")
        model, history = main(use_simple_model=True, epochs=5)
    else:
        print("Mode: Modèle complet (meilleure performance)")
        model, history = main(use_simple_model=False, epochs=10)
