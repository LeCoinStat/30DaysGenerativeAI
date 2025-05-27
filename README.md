# 30DaysGenerativeAI
Ce repo est créé pour partager toutes les ressouces des vidéos du challenge 30DaysGenerativeAI sur la chaîne Youtube LeCoinStat.

## Jour21 : Classification Chien/Chat avec CNN

Le dossier `Jour21` contient un exemple complet de réseau de neurones convolutifs pour classifier les images de chiens et de chats à l'aide du jeu de données `cats_vs_dogs` de TensorFlow Datasets. 

Pour entraîner rapidement le modèle :
```bash
python Jour21/cnn_chien_chat.py
```

Le script télécharge automatiquement le jeu de données, construit un CNN simple puis lance l'entraînement et l'évaluation.

## Classification et génération de texte

Le fichier `transformers_text.py` montre comment utiliser la bibliothèque `transformers` de Hugging Face pour faire de la classification de texte et de la génération de texte.

Installez d'abord la bibliothèque :

```bash
pip install transformers
```

Puis lancez le script :

```bash
python transformers_text.py
```

Les modèles pré-entraînés seront téléchargés lors de la première exécution.
