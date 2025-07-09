import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# --- Configuración Inicial ---
# Define el dispositivo a usar: 'cuda' para GPU si está disponible, si no, 'cpu'
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {dispositivo}")

# Define las transformaciones para los datos de entrenamiento y validación
transformaciones_datos = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),       # Redimensiona las imágenes
        transforms.RandomHorizontalFlip(),   # Aumentación de datos: voltea imágenes horizontalmente
        transforms.ToTensor(),               # Convierte imágenes a tensores de PyTorch
        transforms.Normalize([0.485, 0.456, 0.406], # Normaliza usando la media y std de ImageNet
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Directorio principal del dataset (actualizar la ruta si es necesario)
directorio_datos = './dataset'

# Crear los datasets usando ImageFolder, que asume que cada subdirectorio es una clase
datasets_imagenes = {x: datasets.ImageFolder(os.path.join(directorio_datos, x),
                                           transformaciones_datos[x])
                   for x in ['train', 'val']}

# Crear los cargadores de datos (DataLoaders) para pasar los datos en lotes
cargadores_datos = {x: DataLoader(datasets_imagenes[x], batch_size=32,
                                  shuffle=True, num_workers=2)
                  for x in ['train', 'val']}

# Obtener los nombres de las clases y su cantidad
nombres_clases = datasets_imagenes['train'].classes
num_clases = len(nombres_clases)
print(f"Clases encontradas: {nombres_clases}")

# --- Visualización de Datos ---
# Función para mostrar un tensor de imagen
def mostrar_imagen(tensor_entrada, titulo=None):
    # Invierte la normalización para mostrar la imagen correctamente
    tensor_entrada = tensor_entrada.numpy().transpose((1, 2, 0))
    media = np.array([0.485, 0.456, 0.406])
    desv_est = np.array([0.229, 0.224, 0.225])
    tensor_entrada = np.clip((tensor_entrada * desv_est + media), 0, 1)
    
    plt.imshow(tensor_entrada)
    if titulo is not None:
        plt.title(titulo)
    plt.axis('off')

# Muestra una imagen por cada clase del conjunto de entrenamiento
clases_mostradas = set()
plt.figure(figsize=(15, 10))
plt.suptitle("Muestras de Imágenes por Clase", fontsize=16)
for i, (entradas, etiquetas) in enumerate(cargadores_datos['train']):
    for j in range(entradas.size(0)):
        etiqueta = etiquetas[j].item()
        if etiqueta not in clases_mostradas:
            plt.subplot(3, 4, len(clases_mostradas) + 1)
            mostrar_imagen(entradas[j].cpu(), titulo=nombres_clases[etiqueta])
            clases_mostradas.add(etiqueta)
        if len(clases_mostradas) == num_clases:
            break
    if len(clases_mostradas) == num_clases:
        break
plt.show()

# --- Configuración del Modelo ---
# Cargar el modelo MobileNetV3 pre-entrenado
modelo = models.mobilenet_v3_large(weights='IMAGENET1K_V1')

# Reemplazar la última capa (clasificador) para que coincida con nuestro número de clases
modelo.classifier[3] = nn.Linear(modelo.classifier[3].in_features, num_clases)

# Enviar el modelo al dispositivo (GPU o CPU) y habilitar multi-GPU si está disponible
modelo = modelo.to(dispositivo)
if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs")
    modelo = nn.DataParallel(modelo)

# Definir la función de pérdida y el optimizador
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.0005)

# --- Bucle de Entrenamiento ---
# Parámetros para la detención temprana (Early Stopping)
mejor_perdida = float('inf')
mejores_pesos_modelo = copy.deepcopy(modelo.state_dict())
contador_detencion_temprana = 0
paciencia = 5 # Número de épocas a esperar si no hay mejora

# Listas para guardar el historial de entrenamiento
lista_perdida_entrenamiento, lista_perdida_validacion = [], []
lista_precision_entrenamiento, lista_precision_validacion = [], []

num_epocas = 100

for epoca in range(num_epocas):
    print(f"\nÉpoca {epoca+1}/{num_epocas}")
    print('-' * 20)

    # Cada época tiene una fase de entrenamiento y una de validación
    for fase in ['train', 'val']:
        if fase == 'train':
            modelo.train()  # Poner el modelo en modo de entrenamiento
            desc_fase = "Entrenamiento"
        else:
            modelo.eval()   # Poner el modelo en modo de evaluación
            desc_fase = "Validación"

        perdida_acumulada = 0.0
        aciertos_acumulados = 0

        # Usar tqdm para una barra de progreso
        cargador_actual = cargadores_datos[fase]
        barra_progreso_fase = tqdm(cargador_actual, desc=f"{desc_fase} [{epoca+1}/{num_epocas}]", leave=False)

        for entradas, etiquetas in barra_progreso_fase:
            entradas, etiquetas = entradas.to(dispositivo), etiquetas.to(dispositivo)
            optimizador.zero_grad() # Limpiar gradientes

            # Habilitar gradientes solo en la fase de entrenamiento
            with torch.set_grad_enabled(fase == 'train'):
                salidas = modelo(entradas)
                _, predicciones = torch.max(salidas, 1)
                perdida = criterio(salidas, etiquetas)

                # Si es la fase de entrenamiento, hacer retropropagación
                if fase == 'train':
                    perdida.backward()
                    optimizador.step()

            # Acumular estadísticas
            perdida_acumulada += perdida.item() * entradas.size(0)
            aciertos_acumulados += torch.sum(predicciones == etiquetas.data)

            # Actualizar la barra de progreso con la pérdida y precisión del lote actual
            barra_progreso_fase.set_postfix({
                "Pérdida": perdida.item(),
                "Precisión": (torch.sum(predicciones == etiquetas.data).double() / entradas.size(0)).item()
            })

        # Calcular pérdida y precisión promedio de la época
        perdida_epoca = perdida_acumulada / len(datasets_imagenes[fase])
        precision_epoca = aciertos_acumulados.double() / len(datasets_imagenes[fase])

        # Guardar estadísticas y gestionar la detención temprana
        if fase == 'train':
            lista_perdida_entrenamiento.append(perdida_epoca)
            lista_precision_entrenamiento.append(precision_epoca.item())
        else: # fase == 'val'
            lista_perdida_validacion.append(perdida_epoca)
            lista_precision_validacion.append(precision_epoca.item())
            
            # Comprobar si es el mejor modelo hasta ahora
            if perdida_epoca < mejor_perdida:
                mejor_perdida = perdida_epoca
                mejores_pesos_modelo = copy.deepcopy(modelo.state_dict())
                
                # Guarda el mejor modelo (maneja el caso de DataParallel)
                if isinstance(modelo, nn.DataParallel):
                    torch.save(modelo.module.state_dict(), 'best_model.pth')
                else:
                    torch.save(modelo.state_dict(), 'best_model.pth')
                
                contador_detencion_temprana = 0 # Reiniciar contador
                print(f"✅ Nuevo mejor modelo guardado con pérdida de validación: {mejor_perdida:.4f}")
            else:
                contador_detencion_temprana += 1

        print(f"Resultado {desc_fase}: Pérdida: {perdida_epoca:.4f} | Precisión: {precision_epoca:.4f}")

    # Comprobar si se debe activar la detención temprana
    if contador_detencion_temprana >= paciencia:
        print(f"\nDetención temprana activada en la época {epoca+1} porque la pérdida no mejoró en {paciencia} épocas.")
        break

# --- Finalización y Visualización de Resultados ---
print("\nEntrenamiento finalizado.")
# Cargar los pesos del mejor modelo encontrado
modelo.load_state_dict(mejores_pesos_modelo)

# Graficar el historial de entrenamiento
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(lista_perdida_entrenamiento, label='Pérdida de Entrenamiento')
plt.plot(lista_perdida_validacion, label='Pérdida de Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Pérdida a lo largo de las Épocas')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lista_precision_entrenamiento, label='Precisión de Entrenamiento')
plt.plot(lista_precision_validacion, label='Precisión de Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.title('Precisión a lo largo de las Épocas')
plt.legend()

plt.tight_layout()
plt.show()

# --- Inferencia: Visualizar predicciones en el conjunto de validación ---
modelo.eval()
entradas, etiquetas = next(iter(cargadores_datos['val']))
entradas = entradas.to(dispositivo)
salidas = modelo(entradas)
_, predicciones = torch.max(salidas, 1)

print("\nMostrando predicciones en un lote de validación:")
plt.figure(figsize=(15, 8))
for i in range(8):
    eje = plt.subplot(2, 4, i + 1)
    mostrar_imagen(entradas[i].cpu())
    eje.set_title(f"Pred: {nombres_clases[predicciones[i]]}\nReal: {nombres_clases[etiquetas[i]]}")
plt.tight_layout()
plt.show()