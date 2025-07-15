import torch 
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os

# --- Configuración Inicial ---
# Define el dispositivo a usar: 'cuda' para GPU si está disponible, si no, 'cpu'
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {dispositivo}")

# Rutas a las carpetas con las imágenes
ruta_entrenamiento = './dataset/train'
ruta_validacion = './dataset/val'


# --- Carga de Datos de Entrenamiento ---
rutas_archivos = []
etiquetas = []

carpetas = os.listdir(ruta_entrenamiento)
print(f"Clases encontradas: {carpetas}")

# Recorre cada carpeta de clase para obtener las rutas de las imágenes y sus etiquetas
for nombre_carpeta in carpetas:
    ruta_carpeta = os.path.join(ruta_entrenamiento, nombre_carpeta)
    for nombre_archivo in os.listdir(ruta_carpeta):
        ruta_completa_archivo = os.path.join(ruta_carpeta, nombre_archivo)
        rutas_archivos.append(ruta_completa_archivo)
        etiquetas.append(nombre_carpeta)

# Crea un DataFrame de pandas con los datos de entrenamiento
serie_archivos = pd.Series(rutas_archivos, name='rutas_archivos')
serie_etiquetas = pd.Series(etiquetas, name='etiquetas')
df_entrenamiento = pd.concat([serie_archivos, serie_etiquetas], axis=1)


# --- Carga de Datos de Validación y Prueba ---
rutas_archivos = []
etiquetas = []

carpetas = os.listdir(ruta_validacion)

for nombre_carpeta in carpetas:
    ruta_carpeta = os.path.join(ruta_validacion, nombre_carpeta)
    lista_archivos = os.listdir(ruta_carpeta)
    for nombre_archivo in lista_archivos:
        ruta_completa_archivo = os.path.join(ruta_carpeta, nombre_archivo)
        etiquetas.append(nombre_carpeta)
        rutas_archivos.append(ruta_completa_archivo)

# Crea un DataFrame temporal para los datos de validación/prueba
serie_archivos = pd.Series(rutas_archivos, name='rutas_archivos')
serie_etiquetas = pd.Series(etiquetas, name='etiquetas')
df_validacion_temporal = pd.concat([serie_archivos, serie_etiquetas], axis=1)

# Divide el DataFrame temporal en un conjunto de validación y otro de prueba (50% para cada uno)
df_validacion, df_prueba = train_test_split(df_validacion_temporal, test_size=0.5, random_state=42)

print(f"\nTamaño del conjunto de entrenamiento: {df_entrenamiento.shape}")
print(f"Tamaño del conjunto de validación: {df_validacion.shape}")
print(f"Tamaño del conjunto de prueba: {df_prueba.shape}")


# --- Preprocesamiento y Creación de Datasets ---
# Codifica las etiquetas (nombres de clases) a números enteros
codificador_etiquetas = LabelEncoder()
codificador_etiquetas.fit(df_entrenamiento["etiquetas"])

# Define la secuencia de transformaciones para las imágenes
transformacion = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensiona las imágenes a 128x128 píxeles
    transforms.ToTensor(),          # Convierte las imágenes a tensores de PyTorch
    transforms.ConvertImageDtype(torch.float) # Convierte los valores de los píxeles a tipo float
])

# Clase personalizada para manejar el dataset
class DatasetDeImagenesPersonalizado(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transformacion = transform
        # Transforma y guarda las etiquetas como tensores en el dispositivo correcto
        self.etiquetas = torch.tensor(codificador_etiquetas.transform(self.dataframe['etiquetas'])).to(dispositivo)

    def __len__(self):
        # Devuelve el número total de imágenes en el dataset
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        # Obtiene una imagen y su etiqueta por su índice
        ruta_imagen = self.dataframe.iloc[idx, 0]
        etiqueta = self.etiquetas[idx]

        # Abre la imagen y la convierte a formato RGB
        imagen = Image.open(ruta_imagen).convert('RGB')
        
        # Aplica las transformaciones si se han definido
        if self.transformacion:
            # Normaliza los valores de los píxeles a [0, 1] y envía la imagen al dispositivo
            imagen = (self.transformacion(imagen) / 255.0).to(dispositivo)

        return imagen, etiqueta

# Crea los objetos Dataset para cada conjunto de datos
dataset_entrenamiento = DatasetDeImagenesPersonalizado(dataframe=df_entrenamiento, transform=transformacion)
dataset_validacion = DatasetDeImagenesPersonalizado(dataframe=df_validacion, transform=transformacion)
dataset_prueba = DatasetDeImagenesPersonalizado(dataframe=df_prueba, transform=transformacion)

# Ejemplo de cómo obtener un elemento del dataset
print(f"\nEjemplo de un dato: {dataset_entrenamiento.__getitem__(2)}")
# Ejemplo de cómo decodificar una etiqueta numérica a su nombre original
print(f"La etiqueta número 2 corresponde a: {codificador_etiquetas.inverse_transform([2])[0]}")


# --- Visualización de Muestras de Imágenes ---
num_filas = 3
num_columnas = 3

fig, ejes = plt.subplots(num_filas, num_columnas, figsize=(8, 8))

for fila in range(num_filas):
    for columna in range(num_columnas):
        # Toma una muestra aleatoria de una imagen del DataFrame de entrenamiento
        imagen = Image.open(df_entrenamiento.sample(n=1)["rutas_archivos"].iloc[0]).convert('RGB')
        ejes[fila, columna].imshow(imagen)
        ejes[fila, columna].axis('off') # Oculta los ejes

plt.tight_layout()
plt.suptitle("Muestras de Imágenes de Entrenamiento", fontsize=16, y=1.02)
plt.show()


# --- Configuración del Modelo y Entrenamiento ---
# Hiperparámetros
TASA_APRENDIZAJE = 1e-3
TAMANO_LOTE = 4
EPOCAS = 50

# DataLoaders: para cargar los datos en lotes de forma eficiente
cargador_entrenamiento = DataLoader(dataset_entrenamiento, batch_size=TAMANO_LOTE, shuffle=True)
cargador_validacion = DataLoader(dataset_validacion, batch_size=TAMANO_LOTE, shuffle=True)
cargador_prueba = DataLoader(dataset_prueba, batch_size=TAMANO_LOTE, shuffle=True)

# Carga el modelo EfficientNet B7 con pesos pre-entrenados en ImageNet
modelo_efficientnet = models.efficientnet_b7(weights='DEFAULT')

# Asegura que todos los parámetros del modelo sean entrenables
for parametro in modelo_efficientnet.parameters():
    parametro.requires_grad = True

# Reemplaza la última capa (clasificador) para que coincida con nuestro número de clases
num_clases = len(df_entrenamiento['etiquetas'].unique())
print(f"\nNúmero de clases para el clasificador: {num_clases}")
modelo_efficientnet.classifier = torch.nn.Linear(in_features=2560, out_features=num_clases)
print("Arquitectura del clasificador final:")
print(modelo_efficientnet.classifier)

# Mueve el modelo al dispositivo (GPU o CPU)
modelo_efficientnet.to(dispositivo)

# Define la función de pérdida y el optimizador
funcion_perdida = nn.CrossEntropyLoss()
optimizador = Adam(modelo_efficientnet.parameters(), lr=TASA_APRENDIZAJE)

# Listas para guardar el historial de entrenamiento para graficar después
historial_perdida_entrenamiento = []
historial_precision_entrenamiento = []


# --- Bucle de Entrenamiento ---
print("\n--- Iniciando Entrenamiento ---")
for epoca in range(EPOCAS):
    precision_total_entrenamiento = 0
    perdida_total_entrenamiento = 0

    # Itera sobre los lotes de datos de entrenamiento
    for entradas, etiquetas_lote in cargador_entrenamiento:
        optimizador.zero_grad()  # Reinicia los gradientes
        
        # Realiza la predicción
        salidas = modelo_efficientnet(entradas)
        
        # Calcula la pérdida
        perdida_entrenamiento = funcion_perdida(salidas, etiquetas_lote)
        perdida_total_entrenamiento += perdida_entrenamiento.item()
        
        # Retropropagación
        perdida_entrenamiento.backward()
        
        # Calcula la precisión del lote
        precision_entrenamiento = (torch.argmax(salidas, axis=1) == etiquetas_lote).sum().item()
        precision_total_entrenamiento += precision_entrenamiento
        
        # Actualiza los pesos del modelo
        optimizador.step()

    # Guarda la pérdida y precisión promedio de la época
    perdida_promedio = round(perdida_total_entrenamiento / len(cargador_entrenamiento), 4)
    precision_promedio = round(precision_total_entrenamiento / len(dataset_entrenamiento) * 100, 4)
    
    historial_perdida_entrenamiento.append(perdida_promedio)
    historial_precision_entrenamiento.append(precision_promedio)

    print(f"Época {epoca+1}/{EPOCAS}, Pérdida de Entrenamiento: {perdida_promedio}, Precisión de Entrenamiento: {precision_promedio} %")

print("--- Entrenamiento Finalizado ---")


# --- Evaluación del Modelo ---
modelo_efficientnet.eval()  # Pone el modelo en modo de evaluación
with torch.no_grad(): # Desactiva el cálculo de gradientes para la evaluación
    precision_total_prueba = 0

    for entradas, etiquetas_lote in cargador_prueba:
        prediccion = modelo_efficientnet(entradas)
        # Suma los aciertos del lote
        precision_lote = (torch.argmax(prediccion, axis=1) == etiquetas_lote).sum().item()
        precision_total_prueba += precision_lote

precision_final = round(precision_total_prueba / len(dataset_prueba) * 100, 2)
print(f"\nPrecisión final en el conjunto de prueba: {precision_final}%")


# --- Guardado del Modelo Entrenado ---
torch.save(modelo_efficientnet.state_dict(), "modelo_enfermedades_plantas.pth")
print("\n✅ Modelo guardado exitosamente como 'modelo_enfermedades_plantas.pth'")
