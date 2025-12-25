# 🔧 Guide d'installation

## ⚠️ Note importante pour Windows

TensorFlow nécessite l'activation des chemins longs sur Windows. Si vous rencontrez des erreurs d'installation, suivez ces étapes :

## 📝 Étapes d'installation

### 1. Activer les chemins longs Windows (Obligatoire)

**Méthode 1 - Via le Registre :**
1. Appuyez sur `Win + R` et tapez `regedit`
2. Naviguez vers : `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Créez ou modifiez la valeur `LongPathsEnabled` (DWORD) à `1`
4. Redémarrez votre ordinateur

**Méthode 2 - Via PowerShell (Admin) :**
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**Méthode 3 - Via Group Policy :**
1. Appuyez sur `Win + R` et tapez `gpedit.msc`
2. Allez à : `Configuration ordinateur > Modèles d'administration > Système > Système de fichiers`
3. Activez "Activer les chemins longs Win32"

### 2. Installer Python et pip

Assurez-vous d'avoir Python 3.8+ installé :
```bash
python --version
```

### 3. Créer un environnement virtuel (Recommandé)

```bash
python -m venv venv
```

**Activer l'environnement :**
- Windows : `venv\Scripts\activate`
- Linux/Mac : `source venv/bin/activate`

### 4. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Si TensorFlow échoue encore, essayez :
```bash
pip install tensorflow-cpu
```

### 5. Vérifier l'installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 🐧 Installation sur Linux

Sur Linux, l'installation est plus simple :

```bash
# Créer environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

## 🍎 Installation sur macOS

```bash
# Créer environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

**Pour Mac M1/M2 (Apple Silicon) :**
```bash
pip install tensorflow-macos
pip install tensorflow-metal  # Accélération GPU
```

## 🔍 Résolution de problèmes

### Erreur "OSError: [Errno 2] No such file or directory"
→ Activez les chemins longs Windows (voir étape 1)

### Erreur "Could not find a version that satisfies the requirement tensorflow"
→ Vérifiez votre version de Python (doit être 3.8-3.11)
→ Essayez `pip install tensorflow-cpu`

### Le modèle ne s'entraîne pas ou est très lent
→ TensorFlow utilisera le CPU par défaut
→ Pour GPU : installez `tensorflow-gpu` et les drivers CUDA

### Erreur d'importation "No module named 'tensorflow'"
→ Vérifiez que vous êtes dans l'environnement virtuel
→ Réinstallez : `pip install --force-reinstall tensorflow`

## 📞 Support

Si vous rencontrez des problèmes :
1. Vérifiez que Python 3.8-3.11 est installé
2. Vérifiez que les chemins longs sont activés (Windows)
3. Essayez dans un nouvel environnement virtuel
4. Consultez la documentation TensorFlow : https://www.tensorflow.org/install

## ✅ Tester l'installation

Une fois tout installé, testez avec :
```bash
cd src
python -c "from data_loader import load_mnist_data; print('OK')"
```

Si cela fonctionne, vous êtes prêt à utiliser le projet !
