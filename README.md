# 🍅 Sistema de Detección de Enfermedades en Hojas de Tomate

Sistema avanzado de comparación de modelos de Deep Learning para la detección automática de enfermedades en hojas de tomate, con análisis estadístico completo, visualizaciones interactivas y generación de reportes PDF profesionales.

## 🌟 Características Principales

### 1. **Análisis con 3 Modelos de ML**

- **MobileNetV3**: Red neuronal optimizada para dispositivos móviles (5.4M parámetros)
- **EfficientNetB7**: CNN de alta precisión con balance eficiencia/rendimiento (66M parámetros)
- **SVM + ResNet50**: Enfoque híbrido combinando extracción de características profundas con clasificación tradicional (25M parámetros)

### 2. **Pruebas Estadísticas Completas**

- **Análisis Moderno**:
  - Coeficiente Kappa de Cohen (concordancia entre modelos)
  - Análisis de Entropía (medición de incertidumbre)
  - Consenso ponderado entre modelos
  - Intervalos de confianza visuales
- **Pruebas Tradicionales**:
  - T-Test pareado (comparación de precisiones)
  - Prueba Z de proporciones
  - Matriz de confusión con datos simulados

### 3. **Visualizaciones Avanzadas**

- Gráficos de barras con niveles de confianza
- Matriz de calor de probabilidades (heatmap)
- Matriz de acuerdo entre modelos
- Gráficos de consenso con barras de error
- Matriz de confusión interactiva

### 4. **Generación de Reportes PDF**

- Reporte completo de 7+ páginas
- Incluye imagen analizada en alta resolución
- Todos los gráficos estadísticos
- Tablas de resultados formateadas
- Recomendaciones basadas en severidad
- Formato profesional listo para presentar

## 📋 Requisitos del Sistema

### Hardware

- **Mínimo**: 4GB RAM, CPU dual-core
- **Recomendado**: 8GB RAM, GPU CUDA compatible
- **Espacio en disco**: 2GB para modelos y dependencias

### Software

- Python 3.10.0
- pip (gestor de paquetes)
- Git (opcional, para clonar el repositorio)

## 🚀 Instalación Paso a Paso

### 1. **Preparación del Entorno**

```bash
# Clona el repositorio (o descarga el ZIP)
git clone https://github.com/DiegoVJ22/tomato-leaf-disease-detection-models.git
cd tomato-leaf-disease-detection-models

# Crea un entorno virtual
python -m venv venv

# Activa el entorno
# En Linux/Mac:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### 2. **Instalación de Dependencias**

```bash
# Actualiza pip
pip install --upgrade pip

# Instala todas las dependencias
pip install -r requirements.txt
```

### 3. **Verificación de la Instalación**

```bash
# Verifica que todo esté correcto
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

## 💻 Ejecución de la Aplicación

```bash
# Ejecuta la aplicación
streamlit run app.py

# La aplicación se abrirá en: http://localhost:8501
```

## 🎯 Guía de Uso Detallada

### Tab 1: Análisis Individual

1. **Cargar Imagen**

   - Formatos soportados: JPG, JPEG, PNG
   - Tamaño recomendado: 224x224 a 1024x1024 píxeles
   - La imagen debe mostrar claramente la hoja

2. **Analizar**

   - Click en "🔬 Analizar Imagen"
   - Espera 2-5 segundos para el procesamiento
   - Los 3 modelos se ejecutan en paralelo

3. **Interpretar Resultados**
   - Cada tarjeta muestra: diagnóstico, confianza, severidad, tiempo
   - Los gráficos de barras muestran top 5 probabilidades
   - Verde = saludable, Rojo = enfermedad severa

### Tab 2: Comparación de Modelos

- **Tabla Comparativa**: Resume predicciones de todos los modelos
- **Análisis de Consenso**: Combina predicciones para mayor confiabilidad
- **Matriz de Acuerdo**: Visualiza concordancia entre modelos

### Tab 3: Métricas y Estadísticas

- **Gráfico de Radar**: Compara precisión, recall y F1-Score
- **Trade-off**: Velocidad vs precisión para cada modelo
- **Tiempos Reales**: Basados en tu hardware actual

### Tab 4: Pruebas Estadísticas

1. **Análisis de Concordancia**

   - Kappa de Cohen entre pares de modelos
   - Consenso con confianza promedio
   - Gráfico de barras de confianza

2. **Pruebas Tradicionales**

   - T-Test: Compara precisiones históricas
   - Z-Test: Evalúa proporciones de acierto
   - Valores p para significancia estadística

3. **Visualizaciones Avanzadas**

   - Matriz de probabilidades completa
   - Matriz de confusión de ejemplo
   - Análisis de entropía (incertidumbre)

4. **Generar Reporte PDF**
   - Click en "🎯 Generar Reporte PDF"
   - Incluye todos los análisis y gráficos
   - Descarga automática del archivo

## 📊 Interpretación de Resultados

### Códigos de Confianza

```python
if confidence > 0.9:
    return "Muy Alta - Resultado muy confiable"
elif confidence > 0.7:
    return "Alta - Resultado confiable"
elif confidence > 0.5:
    return "Media - Considerar verificación"
else:
    return "Baja - Requiere revisión manual"
```

### Análisis de Entropía

| Entropía | Interpretación         | Acción Recomendada                   |
| -------- | ---------------------- | ------------------------------------ |
| < 0.5    | Muy seguro             | Proceder con confianza               |
| 0.5-1.0  | Seguro                 | Resultado confiable                  |
| 1.0-1.5  | Moderadamente seguro   | Verificar si es crítico              |
| 1.5-2.0  | Incertidumbre moderada | Considerar segunda opinión           |
| > 2.0    | Alta incertidumbre     | Tomar nueva foto o consultar experto |

### Matriz de Consenso

```
3/3 modelos coinciden → Consenso fuerte
2/3 modelos coinciden → Consenso moderado
0/3 modelos coinciden → Sin consenso (revisar)
```

## 🌿 Enfermedades Detectables

| Enfermedad         | Nombre en Español     | Severidad  | Síntomas Clave                       |
| ------------------ | --------------------- | ---------- | ------------------------------------ |
| Bacterial_spot     | Mancha Bacteriana     | 🔴 Alta    | Manchas negras con halo amarillo     |
| Early_blight       | Tizón Temprano        | 🟠 Media   | Manchas concéntricas marrones        |
| Late_blight        | Tizón Tardío          | 🔴 Alta    | Manchas irregulares gris-verde       |
| Leaf_Mold          | Moho de Hoja          | 🟠 Media   | Manchas amarillas, moho gris         |
| Septoria_leaf_spot | Mancha de Septoria    | 🟠 Media   | Pequeñas manchas con centro gris     |
| Spider_mites       | Ácaros Araña          | 🟡 Baja    | Puntos amarillos, telarañas finas    |
| Target_Spot        | Mancha Diana          | 🟠 Media   | Manchas circulares concéntricas      |
| TYLCV              | Virus Rizado Amarillo | 🔴 Alta    | Hojas rizadas y amarillentas         |
| ToMV               | Virus del Mosaico     | 🔴 Alta    | Patrón de mosaico verde claro/oscuro |
| healthy            | Saludable             | 🟢 Ninguna | Hoja verde uniforme sin manchas      |

## 📄 Estructura del Reporte PDF

### Contenido del Reporte (7+ páginas)

1. **Página 1 - Portada**

   - Título profesional
   - Fecha y hora de análisis
   - Imagen analizada (4×4 pulgadas, centrada)

2. **Página 2 - Resultados Principales**

   - Tabla comparativa de modelos
   - Diagnóstico por consenso
   - Nivel de confianza general

3. **Página 3 - Visualizaciones Estadísticas**

   - Gráfico de comparación de confianza
   - Análisis de consenso con barras de error
   - Matriz de acuerdo entre modelos

4. **Página 4 - Análisis Avanzado**

   - Matriz de probabilidades completa
   - Matriz de confusión de ejemplo
   - Tabla de entropía e incertidumbre

5. **Página 5 - Pruebas Estadísticas**

   - Resultados de T-Test pareado
   - Resultados de prueba Z
   - Interpretaciones estadísticas

6. **Página 6 - Detalles por Modelo**

   - Top 5 probabilidades para cada modelo
   - Tablas individuales con porcentajes

7. **Página 7 - Recomendaciones**
   - Acciones sugeridas según severidad
   - Medidas preventivas
   - Nota de responsabilidad

## 🔧 Solución de Problemas

### Error: "No se pudieron cargar los modelos"

```bash
# Verifica que los archivos existan
ls -la models/

# Verifica los permisos
chmod 644 models/*

# Verifica el tamaño (deben ser > 10MB)
du -h models/*
```

### Warning: Versión de scikit-learn

```bash
# Actualiza a la versión correcta
pip uninstall scikit-learn
pip install scikit-learn==1.3.2
```

### Error: CUDA no disponible

```bash
# Verifica CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Si es False, la app usará CPU automáticamente
# Para forzar CPU:
export CUDA_VISIBLE_DEVICES=""
```

### La aplicación es muy lenta

1. **Optimización de memoria**:

   ```python
   # Reduce batch size en el código
   # Usa solo 1-2 modelos si es necesario
   ```

2. **Usar GPU**:
   ```bash
   # Instala PyTorch con CUDA
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Error al generar PDF

```bash
# Reinstala reportlab
pip uninstall reportlab pillow
pip install reportlab pillow

# En Linux, instala dependencias del sistema
sudo apt-get install python3-dev python3-setuptools
sudo apt-get install libtiff5-dev libjpeg8-dev libopenjp2-7-dev
```

## 📈 Métricas de Rendimiento

### Comparación de Modelos

| Métrica             | MobileNetV3 | EfficientNetB7 | SVM + ResNet50 |
| ------------------- | ----------- | -------------- | -------------- |
| **Precisión**       | 95.2%       | 97.8%          | 93.5%          |
| **Recall**          | 94.8%       | 97.5%          | 93.0%          |
| **F1-Score**        | 95.0%       | 97.6%          | 93.2%          |
| **Velocidad (FPS)** | 45          | 12             | 25             |
| **Tiempo/imagen**   | ~22ms       | ~83ms          | ~40ms          |
| **Uso de memoria**  | ~200MB      | ~800MB         | ~400MB         |
| **Tamaño modelo**   | 21.7MB      | 254.6MB        | 97.8MB         |

### Rendimiento por Hardware

| Hardware       | MobileNetV3 | EfficientNetB7 | SVM   |
| -------------- | ----------- | -------------- | ----- |
| CPU (i5)       | 100ms       | 400ms          | 180ms |
| CPU (i7)       | 60ms        | 250ms          | 120ms |
| GPU (GTX 1060) | 15ms        | 50ms           | 30ms  |
| GPU (RTX 3060) | 8ms         | 25ms           | 15ms  |

## 🤝 Contribuciones y Mejoras

### Mejoras Implementadas

- ✅ Pruebas estadísticas completas
- ✅ Generación de reportes PDF
- ✅ Visualizaciones interactivas
- ✅ Análisis de consenso
- ✅ Manejo de errores robusto

### Mejoras Futuras

- [ ] API REST para integración
- [ ] Soporte para video en tiempo real
- [ ] Explicabilidad con Grad-CAM
- [ ] Base de datos para histórico
- [ ] Exportación a Excel
- [ ] Modo offline completo
- [ ] Aplicación móvil

Desarrollado con ❤️ para la agricultura sostenible 🌱
