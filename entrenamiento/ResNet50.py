"""
Entrenamiento de un clasificador SVM para el dataset `tomato`
=============================================================

Flujo:
1.  Extraer *embeddings* por imagen con un ResNet-50 pre-entrenado (feature-extractor, sin ajuste).
2.  Entrenar un SVM (RBF) sobre esos vectores.
3.  Evaluar y guardar el modelo.

Requisitos:
torch, torchvision, scikit-learn, numpy, joblib, tqdm
"""

import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm


# ---------- 1. Configuración general ----------
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
directorio_datos = './dataset'          # La estructura debe ser .../train y .../val

# ---------- 2. Transformaciones (sin aumento de datos) ----------
transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], # Media y std de ImageNet
                         [0.229, 0.224, 0.225])
])

# ---------- 3. Datasets y Cargadores de Datos ----------
conjuntos_datos = {division: datasets.ImageFolder(os.path.join(directorio_datos, division), transformaciones)
                   for division in ['train', 'val']}

cargadores_datos = {division: DataLoader(conjuntos_datos[division], batch_size=64,
                                          shuffle=False, num_workers=2)
                    for division in ['train', 'val']}

nombres_clases = conjuntos_datos['train'].classes
print(f"Número de clases: {len(nombres_clases)} → {nombres_clases}")

# ---------- 4. Extractor de Características (ResNet-50 congelado) ----------
red_extractora = models.resnet50(weights='IMAGENET1K_V2')
red_extractora.fc = torch.nn.Identity()   # Quitamos la capa de clasificación final
red_extractora.eval().to(dispositivo)     # Modo evaluación y mover a GPU/CPU

# Congelamos todos los parámetros de la red
for parametro in red_extractora.parameters():
    parametro.requires_grad = False       # ¡Totalmente congelado!

@torch.inference_mode()  # Decorador para eficiencia: desactiva el cálculo de gradientes
def extraer_vectores(cargador):
    """Pasa todos los datos a través de la red y devuelve los vectores y etiquetas."""
    vectores_lista, etiquetas_lista = [], []
    for lote_imgs, lote_etiquetas in tqdm(cargador, desc='Extrayendo vectores'):
        lote_imgs = lote_imgs.to(dispositivo)
        # Extrae características y las mueve a la CPU como arrays de numpy
        vectores_batch = red_extractora(lote_imgs).cpu().numpy()
        vectores_lista.append(vectores_batch)
        etiquetas_lista.append(lote_etiquetas.numpy())
        
    # Concatena los resultados de todos los lotes en un único array
    return np.concatenate(vectores_lista), np.concatenate(etiquetas_lista)

print("\nExtrayendo embeddings de las imágenes...")
X_entrenamiento, y_entrenamiento = extraer_vectores(cargadores_datos['train'])
X_validacion, y_validacion     = extraer_vectores(cargadores_datos['val'])
print(f"Forma de los embeddings de entrenamiento: {X_entrenamiento.shape}") # (N_imágenes, 2048_dimensiones)

# ---------- 5. Entrenamiento del Clasificador SVM ----------
print("\nEntrenando clasificador SVM (kernel RBF)...")
# Usamos un kernel RBF (Función de Base Radial), una elección común y potente
# C=10 es el parámetro de regularización. probability=True para poder obtener probabilidades después.
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm.fit(X_entrenamiento, y_entrenamiento)

# ---------- 6. Evaluación del Modelo ----------
print("\nEvaluando el modelo en el conjunto de validación...")
predicciones_validacion = svm.predict(X_validacion)
precision = accuracy_score(y_validacion, predicciones_validacion)

print(f"\nPrecisión en validación: {precision:.4f}\n")
print("Reporte de Clasificación:")
print(classification_report(y_validacion, predicciones_validacion, target_names=nombres_clases))

# ---------- 7. Guardar el Modelo Entrenado ----------
# Se guarda el objeto SVM entrenado y la lista de nombres de las clases
# Las claves 'svm' y 'classes' se mantienen en inglés por compatibilidad con otras aplicaciones.
joblib.dump({
    'svm': svm,
    'classes': nombres_clases
}, 'svm_tomato.pkl')

print("\n✅ Modelo SVM guardado exitosamente en 'svm_tomato.pkl'")