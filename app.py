import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import os
from pathlib import Path
from sklearn.preprocessing import label_binarize
from sklearn.metrics import cohen_kappa_score, confusion_matrix, matthews_corrcoef, accuracy_score, roc_curve, auc
from itertools import cycle
import base64
from io import BytesIO
import warnings

# Ignorar advertencias de versiones de scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="🍅 Detección de Enfermedades en Hojas de Tomate",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================
# CONFIGURACIÓN MULTILENGUAJE
# ===========================================

# Diccionario de traducciones
TRADUCCIONES = {
    "es": {
        "titulo_app": "🍅 Sistema de Detección de Enfermedades en Tomate",
        "subtitulo_app": "Comparación de modelos de Deep Learning para diagnóstico automático",
        "configuracion": "⚙️ Configuración",
        "modelos_disponibles": "📊 Modelos Disponibles",
        "opciones_visualizacion": "🎨 Opciones de Visualización",
        "mostrar_probabilidades": "Mostrar todas las probabilidades",
        "mostrar_comparacion": "Mostrar gráfico comparativo",
        "umbral_confianza": "Umbral de confianza",
        "analisis_individual": "🔍 Análisis Individual",
        "comparacion_modelos": "📊 Comparación de Modelos",
        "metricas_estadisticas": "📈 Métricas y Estadísticas",
        "pruebas_estadisticas": "🧪 Pruebas Estadísticas",
        "evaluacion_modelos": "🔬 Evaluación de Modelos (Batch)",
        "cargar_imagen": "📤 Cargar Imagen",
        "seleccionar_imagen": "Selecciona una imagen de hoja de tomate",
        "formato_soportado": "Formatos soportados: JPG, JPEG, PNG",
        "analizar_imagen": "🔬 Analizar Imagen",
        "resultados_analisis": "🎯 Resultados del Análisis",
        "diagnostico": "Diagnóstico",
        "confianza": "Confianza",
        "severidad": "Severidad",
        "tiempo": "Tiempo",
        "comparacion_predicciones": "🔄 Comparación de Predicciones",
        "modelo": "Modelo",
        "prediccion": "Predicción",
        "tiempo_s": "Tiempo (s)",
        "analisis_consenso": "📊 Análisis de Consenso",
        "nivel_acuerdo": "🤝 Nivel de Acuerdo entre Modelos",
        "metricas_rendimiento": "📈 Métricas de Rendimiento",
        "comparacion_metricas_rendimiento": "📊 Comparación de Métricas de Rendimiento",
        "tabla_metricas": "📋 Tabla de Métricas Detalladas",
        "modo_analisis": "Selecciona el Modo de Análisis",
        "selecciona_modo": "Selecciona el modo",
        "analisis_completo": "Análisis Completo (1000 imágenes)",
        "analisis_personalizado": "Análisis Personalizado",
        "num_imagenes_clase": "Número de imágenes por clase",
        "help_num_imagenes": "Elige cuántas imágenes de cada clase quieres analizar. El análisis será más rápido con menos imágenes.",
        "tiempo_ms": "Tiempo (ms)",
        "velocidad_imagenes_segundo": "Velocidad (imágenes/segundo)",
        "precision": "Precisión",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "velocidad_fps": "Velocidad",
        "baja_incertidumbre": "Baja Incertidumbre",
        "alta_incertidumbre": "Alta Incertidumbre",
        "perfecto": "Perfecto",
        "haga_clic_descargar_pdf": "Descargar PDF",
        "ver_matriz_confusion": "Ver Matriz de Confusión",
        "cargar_imagenes_clase": "Carga Imágenes de",
        "generar_reporte_analisis": "📄 Generar Reporte de Análisis",
        "concordancia_consenso": "Consenso de concordancia",
        "compromiso_velocidad_precision":   "⚖️ Velocidad vs. Precisión",
        "tiempo_inferencia_modelo": "⏱️ Tiempo de Inferencia por Modelo",
        "analisis_estadistico": "🧪 Análisis Estadístico Detallado",
        "concordancia_modelos": "🤝 Concordancia entre Modelos",
        "analisis_consenso_tit": "🎯 Análisis de Consenso",
        "analisis_confianza": "📈 Análisis de Confianza",
        "pruebas_tradicionales": "📐 Pruebas Estadísticas Tradicionales (Basadas en Datos Históricos)",
        "ttest_pareado": "📊 T-Test Pareado",
        "ztest_proporciones": "📊 Prueba Z de Proporciones",
        "incertidumbre_visualizaciones": "🔬 Análisis de Incertidumbre y Visualizaciones",
        "generar_reporte": "📄 Generar Reporte Completo",
        "descargar_reporte": "📥 Descargar Reporte PDF",
        "evaluacion_real": "🔬 Evaluación de Modelos con un Conjunto de Datos Real",
        "info_evaluacion": "Carga imágenes de prueba para cada clase para obtener métricas de rendimiento reales y comparar los modelos de forma robusta.",
        "carga_imagenes": "Carga de Imágenes de Prueba",
        "iniciar_evaluacion": "🚀 Iniciar Evaluación de Modelos",
        "resultados_evaluacion": "📊 Resultados de la Evaluación",
        "resumen_rendimiento": "Resumen de Rendimiento General",
        "prueba_mcnemar": "Prueba de McNemar (Comparación de Errores)",
        "info_mcnemar": "Esta prueba determina si los modelos cometen tipos de errores diferentes. Un p-value < 0.05 sugiere que la diferencia en los errores es estadísticamente significativa.",
        "mapa_calor_predicciones": "🔥 Mapa de Calor de Predicciones",
        "info_mapa_calor": "Este mapa muestra la distribución de todas las predicciones hechas por cada modelo. Ayuda a identificar si un modelo tiene un sesgo hacia ciertas clases.",
        "matrices_confusion": "📊 Matrices de Confusión Detalladas",
        "generar_reporte_evaluacion": "📄 Generar Reporte de Evaluación",
        "descargar_reporte_evaluacion": "📥 Descargar Reporte de Evaluación en PDF",
        "nota_pie": "💡 **Nota:** Este sistema es una herramienta de apoyo. Para un diagnóstico definitivo, consulte con un experto agrónomo.",
        "desarrollado": "Desarrollado con ❤️ usando PyTorch y Streamlit",
        "cargando_modelos": "Cargando modelos...",
        "procesando": "Procesando...",
        "error_modelos": "❌ No se pudieron cargar los modelos. Verifica que los archivos estén en la carpeta 'models/'.",
        "info_tab1": "👆 Primero carga y analiza una imagen en la pestaña 'Análisis Individual'",
        "info_tab4": "👆 Primero carga y analiza una imagen en la pestaña 'Análisis Individual' para ver las estadísticas.",
        "evaluacion_completada": "¡Evaluación completada con éxito!",
        "error_evaluacion": "La evaluación no pudo completarse. Asegúrate de cargar imágenes.",
        "generando_pdf": "Generando PDF...",
        "reporte_generado": "✅ Reporte generado exitosamente. ¡Haz clic arriba para descargar!",
        "parametros": "Parámetros",
        "precision_validacion": "Precisión (Validación)",
        "velocidad": "Velocidad",
        "rapido": "Rápido",
        "medio": "Medio",
        "lento": "Lento",
        "interpretacion_ttest": "**Interpretación:** p < 0.05 indica una diferencia estadísticamente significativa en la precisión promedio entre los modelos.",
        "interpretacion_ztest": "**Interpretación:** Compara si las proporciones de aciertos son significativamente diferentes.",
        "entropia_info": "La entropía mide la incertidumbre de una predicción. Valores más altos indican que el modelo duda entre varias clases.",
        "reporte_analisis": "Reporte de Análisis de Enfermedades en Tomate",
        "fecha_generacion": "Fecha de generación",
        "imagen_analizada": "Imagen Analizada",
        "error_cargar_imagen": "Error al cargar la imagen",
        "resultados_analisis_modelo": "Resultados del Análisis por Modelo",
        "recomendaciones_titulo": "Recomendaciones",
        "nota_reporte": "Nota: Este reporte es una herramienta de apoyo. Para un diagnóstico definitivo, consulte con un experto agrónomo.",
        "reporte_evaluacion": "Reporte de Evaluación de Modelos",
        "fecha_evaluacion": "Fecha de evaluación",
        "total_imagenes": "Total de imágenes evaluadas",
        "resumen_rendimiento_tit": "Resumen de Rendimiento",
        "prueba_mcnemar_tit": "Prueba de McNemar (Comparación de Errores)",
        "mapa_calor_distribucion": "Mapa de Calor de Distribución de Predicciones",
        "matrices_confusion_tit": "Matrices de Confusión por Modelo",
        "curva_roc_auc": "📊 Curva ROC-AUC y Métrica AUC",
        "info_roc_auc": "La curva ROC (Receiver Operating Characteristic) ilustra la capacidad de diagnóstico de un clasificador binario. Una curva más cercana a la esquina superior izquierda indica un mejor rendimiento. El AUC (Area Under the Curve) representa la medida de rendimiento agregado en todas las clases; un valor de 1.0 es perfecto, mientras que 0.5 es aleatorio.",
        "curva_roc_auc_comparativa": "Curva ROC Comparativa (Micro-Promedio)",
        "distribucion_predicciones": "Distribución de Predicciones por Modelo",
        "clase_predicha": "Clase Predicha",
        "matriz_confusion": "Matriz de Confusión",
        "clase_real": "Clase Real",
        "ejemplo_validacion": "(Ejemplo con datos de validación)",
        "comparacion": "Comparación",
        "estadistico_chi": "Estadístico Chi-cuadrado",
        "p_value": "P-Value",
        "significativo": "Significativo (p < 0.05)",
        "t_statistic": "t-statistic",
        "significativo_ttest": "Significativo",
        "mean_diff": "Diferencia Media",
        "z_statistic": "z-statistic",
        "prop1": "Prop. Modelo 1",
        "prop2": "Prop. Modelo 2",
        "consenso": "Consenso",
        "entropia": "Entropía",
        "interpretacion": "Interpretación",
        "diferencia_significativa": "Diferencia Significativa",
        "acuerdo": "Acuerdo",
        "diagnostico_consenso": "Diagnóstico por Consenso",
        "confianza_promedio": "Confianza Promedio",
        "niveles_confianza": "Niveles de Confianza por Modelo",
        "top_diagnosticos": "Top 5 Diagnósticos por Consenso",
        "probabilidad_promedio": "Probabilidad Promedio",
        "matriz_acuerdo": "Matriz de Acuerdo entre Modelos",
        "matriz_probabilidades": "Matriz de Probabilidades por Modelo",
        "analisis_probabilidades": "Análisis Detallado de Probabilidades",
        "enfermedad": "Enfermedad",
        "probabilidad": "Probabilidad",
        "severidad_detectada": "Severidad detectada",
        "recomendaciones": {
            "Alta": [
                "Consulte inmediatamente con un experto agrónomo",
                "Aísle las plantas afectadas para evitar propagación",
                "Considere tratamiento con fungicidas específicos",
                "Monitoree diariamente la evolución",
                "Documente la evolución con fotografías diarias"
            ],
            "Media": [
                "Aplique medidas preventivas de control",
                "Mejore la ventilación del cultivo",
                "Revise el programa de riego y fertilización",
                "Realice seguimiento semanal",
                "Considere aplicación preventiva de productos orgánicos"
            ],
            "Baja": [
                "Mantenga vigilancia regular",
                "Aplique buenas prácticas agrícolas",
                "Considere tratamientos preventivos naturales",
                "Revise las condiciones ambientales del cultivo"
            ],
            "Ninguna": [
                "Continúe con el mantenimiento regular",
                "Mantenga las buenas prácticas actuales",
                "Realice monitoreo preventivo periódico",
                "Documente el estado saludable para referencia futura"
            ]
        }
    },
    "en": {
        "titulo_app": "🍅 Tomato Leaf Disease Detection System",
        "subtitulo_app": "Deep Learning model comparison for automatic diagnosis",
        "configuracion": "⚙️ Configuration",
        "modelos_disponibles": "📊 Available Models",
        "opciones_visualizacion": "🎨 Visualization Options",
        "mostrar_probabilidades": "Show all probabilities",
        "mostrar_comparacion": "Show comparative chart",
        "umbral_confianza": "Confidence threshold",
        "analisis_individual": "🔍 Individual Analysis",
        "comparacion_modelos": "📊 Model Comparison",
        "metricas_estadisticas": "📈 Metrics and Statistics",
        "pruebas_estadisticas": "🧪 Statistical Tests",
        "evaluacion_modelos": "🔬 Model Evaluation (Batch)",
        "cargar_imagen": "📤 Upload Image",
        "seleccionar_imagen": "Select a tomato leaf image",
        "formato_soportado": "Supported formats: JPG, JPEG, PNG",
        "analizar_imagen": "🔬 Analyze Image",
        "resultados_analisis": "🎯 Analysis Results",
        "diagnostico": "Diagnosis",
        "confianza": "Confidence",
        "severidad": "Severity",
        "tiempo": "Time",
        "comparacion_predicciones": "🔄 Prediction Comparison",
        "modelo": "Model",
        "prediccion": "Prediction",
        "tiempo_s": "Time (s)",
        "analisis_consenso": "📊 Consensus Analysis",
        "nivel_acuerdo": "🤝 Agreement Level Between Models",
        "metricas_rendimiento": "📈 Performance Metrics",
        "comparacion_metricas_rendimiento": "📊 Performance Metrics Comparison",
        "tabla_metricas": "📋 Detailed Metrics Table",
        "modo_analisis": "Select Analysis Mode",
        "selecciona_modo": "Select mode",
        "analisis_completo": "Full Analysis (1000 images)",
        "analisis_personalizado": "Custom Analysis",
        "num_imagenes_clase": "Number of images per class",
        "help_num_imagenes": "Choose how many images from each class you want to analyze. The analysis will be faster with fewer images.",
        "tiempo_ms": "Time (ms)",
        "velocidad_imagenes_segundo": "Speed (images/second)",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "velocidad_fps": "Speed",
        "perfecto": "Perfect",
        "baja_incertidumbre": "Low Uncertainty",
        "alta_incertidumbre": "High Uncertainty",
        "haga_clic_descargar_pdf": "Download PDF",
        "ver_matriz_confusion": "View Confusion Matrix",
        "cargar_imagenes_clase": "Load Class Images",
        "generar_reporte_analisis": "📄 Generate Analysis Report",
        "concordancia_consenso": "Consensus Agreement",
        "compromiso_velocidad_precision":  "⚖️ Speed vs. Precision",
        "tiempo_inferencia_modelo": "⏱️ Inference Time by Model",
        "analisis_estadistico": "🧪 Detailed Statistical Analysis",
        "concordancia_modelos": "🤝 Model Agreement",
        "analisis_consenso_tit": "🎯 Consensus Analysis",
        "curva_roc_auc": "📊 ROC-AUC Curve and AUC Metric",
        "info_roc_auc": "The ROC (Receiver Operating Characteristic) curve illustrates the diagnostic ability of a binary classifier system. A curve closer to the top-left corner indicates better performance. The AUC (Area Under the Curve) represents the aggregate performance measure across all classes; a value of 1.0 is perfect, while 0.5 is random.",
        "curva_roc_auc_comparativa": "Comparative ROC Curve (Micro-Average)",
        "analisis_confianza": "📈 Confidence Analysis",
        "pruebas_tradicionales": "📐 Traditional Statistical Tests (Based on Historical Data)",
        "ttest_pareado": "📊 Paired T-Test",
        "ztest_proporciones": "📊 Z-Test for Proportions",
        "incertidumbre_visualizaciones": "🔬 Uncertainty Analysis and Visualizations",
        "generar_reporte": "📄 Generate Full Report",
        "descargar_reporte": "📥 Download PDF Report",
        "evaluacion_real": "🔬 Model Evaluation with Real Dataset",
        "info_evaluacion": "Upload test images for each class to obtain real performance metrics and robustly compare models.",
        "carga_imagenes": "Test Images Upload",
        "iniciar_evaluacion": "🚀 Start Model Evaluation",
        "resultados_evaluacion": "📊 Evaluation Results",
        "resumen_rendimiento": "General Performance Summary",
        "prueba_mcnemar": "McNemar Test (Error Comparison)",
        "info_mcnemar": "This test determines if models make different types of errors. A p-value < 0.05 suggests the difference in errors is statistically significant.",
        "mapa_calor_predicciones": "🔥 Prediction Heatmap",
        "info_mapa_calor": "This map shows the distribution of all predictions made by each model. Helps identify if a model is biased towards certain classes.",
        "matrices_confusion": "📊 Detailed Confusion Matrices",
        "generar_reporte_evaluacion": "📄 Generate Evaluation Report",
        "descargar_reporte_evaluacion": "📥 Download Evaluation Report PDF",
        "nota_pie": "💡 **Note:** This system is a support tool. For a definitive diagnosis, consult with an agricultural expert.",
        "desarrollado": "Developed with ❤️ using PyTorch and Streamlit",
        "cargando_modelos": "Loading models...",
        "procesando": "Processing...",
        "error_modelos": "❌ Could not load models. Verify that files are in 'models/' folder.",
        "info_tab1": "👆 First upload and analyze an image in the 'Individual Analysis' tab",
        "info_tab4": "👆 First upload and analyze an image in the 'Individual Analysis' tab to see statistics.",
        "evaluacion_completada": "Evaluation completed successfully!",
        "error_evaluacion": "Evaluation could not be completed. Make sure to upload images.",
        "generando_pdf": "Generating PDF...",
        "reporte_generado": "✅ Report generated successfully. Click above to download!",
        "parametros": "Parameters",
        "precision_validacion": "Accuracy (Validation)",
        "velocidad": "Speed",
        "rapido": "Fast",
        "medio": "Medium",
        "lento": "Slow",
        "interpretacion_ttest": "**Interpretation:** p < 0.05 indicates a statistically significant difference in average accuracy between models.",
        "interpretacion_ztest": "**Interpretation:** Compares if success proportions are significantly different.",
        "entropia_info": "Entropy measures prediction uncertainty. Higher values indicate the model is uncertain between several classes.",
        "reporte_analisis": "Tomato Disease Analysis Report",
        "fecha_generacion": "Generation date",
        "imagen_analizada": "Analyzed Image",
        "error_cargar_imagen": "Error loading image",
        "resultados_analisis_modelo": "Analysis Results by Model",
        "recomendaciones_titulo": "Recommendations",
        "nota_reporte": "Note: This report is a support tool. For a definitive diagnosis, consult with an agricultural expert.",
        "reporte_evaluacion": "Model Evaluation Report",
        "fecha_evaluacion": "Evaluation date",
        "total_imagenes": "Total images evaluated",
        "resumen_rendimiento_tit": "Performance Summary",
        "prueba_mcnemar_tit": "McNemar Test (Error Comparison)",
        "mapa_calor_distribucion": "Prediction Distribution Heatmap",
        "matrices_confusion_tit": "Confusion Matrices by Model",
        "distribucion_predicciones": "Prediction Distribution by Model",
        "clase_predicha": "Predicted Class",
        "matriz_confusion": "Confusion Matrix",
        "clase_real": "True Class",
        "ejemplo_validacion": "(Example with validation data)",
        "comparacion": "Comparison",
        "estadistico_chi": "Chi-square Statistic",
        "p_value": "P-Value",
        "significativo": "Significant (p < 0.05)",
        "t_statistic": "t-statistic",
        "significativo_ttest": "Significant",
        "mean_diff": "Mean Difference",
        "z_statistic": "z-statistic",
        "prop1": "Prop. Model 1",
        "prop2": "Prop. Model 2",
        "consenso": "Consensus",
        "entropia": "Entropy",
        "interpretacion": "Interpretation",
        "diferencia_significativa": "Significant Difference",
        "acuerdo": "Agreement",
        "diagnostico_consenso": "Consensus Diagnosis",
        "confianza_promedio": "Average Confidence",
        "niveles_confianza": "Confidence Levels by Model",
        "top_diagnosticos": "Top 5 Consensus Diagnoses",
        "probabilidad_promedio": "Average Probability",
        "matriz_acuerdo": "Model Agreement Matrix",
        "matriz_probabilidades": "Probability Matrix by Model",
        "analisis_probabilidades": "Detailed Probability Analysis",
        "enfermedad": "Disease",
        "probabilidad": "Probability",
        "severidad_detectada": "Detected severity",
        "recomendaciones": {
            "Alta": [
                "Consult immediately with an agricultural expert",
                "Isolate affected plants to prevent spread",
                "Consider treatment with specific fungicides",
                "Monitor evolution daily",
                "Document evolution with daily photographs"
            ],
            "Media": [
                "Apply preventive control measures",
                "Improve crop ventilation",
                "Review irrigation and fertilization program",
                "Perform weekly follow-up",
                "Consider preventive application of organic products"
            ],
            "Baja": [
                "Maintain regular surveillance",
                "Apply good agricultural practices",
                "Consider natural preventive treatments",
                "Review crop environmental conditions"
            ],
            "Ninguna": [
                "Continue with regular maintenance",
                "Maintain current good practices",
                "Perform periodic preventive monitoring",
                "Document healthy state for future reference"
            ]
        }
    },
    "fr": {
        "titulo_app": "🍅 Système de Détection des Maladies des Feuilles de Tomate",
        "subtitulo_app": "Comparaison de modèles de Deep Learning pour le diagnostic automatique",
        "configuracion": "⚙️ Configuration",
        "modelos_disponibles": "📊 Modèles Disponibles",
        "opciones_visualizacion": "🎨 Options de Visualisation",
        "mostrar_probabilidades": "Afficher toutes les probabilités",
        "mostrar_comparacion": "Afficher le graphique comparatif",
        "umbral_confianza": "Seuil de confiance",
        "analisis_individual": "🔍 Analyse Individuelle",
        "comparacion_modelos": "📊 Comparaison de Modèles",
        "metricas_estadisticas": "📈 Métriques et Statistiques",
        "pruebas_estadisticas": "🧪 Tests Statistiques",
        "evaluacion_modelos": "🔬 Évaluation des Modèles (Batch)",
        "cargar_imagen": "📤 Charger l'Image",
        "seleccionar_imagen": "Sélectionnez une image de feuille de tomate",
        "formato_soportado": "Formats pris en charge: JPG, JPEG, PNG",
        "analizar_imagen": "🔬 Analyser l'Image",
        "resultados_analisis": "🎯 Résultats de l'Analyse",
        "diagnostico": "Diagnostic",
        "confianza": "Confiance",
        "severidad": "Sévérité",
        "tiempo": "Temps",
        "comparacion_predicciones": "🔄 Comparaison des Prédictions",
        "modelo": "Modèle",
        "prediccion": "Prédiction",
        "tiempo_s": "Temps (s)",
        "analisis_consenso": "📊 Analyse de Consensus",
        "nivel_acuerdo": "🤝 Niveau d'Accord entre les Modèles",
        "metricas_rendimiento": "📈 Métriques de Performance",
        "comparacion_metricas_rendimiento": "📊 Comparaison des Métriques de Performance",
        "tabla_metricas": "📋 Tableau de Métriques Détaillées",
        "modo_analisis": "Sélectionnez le Mode d'Analyse",
        "selecciona_modo": "Sélectionnez le mode",
        "analisis_completo": "Analyse Complète (1000 images)",
        "analisis_personalizado": "Analyse Personnalisée",
        "num_imagenes_clase": "Nombre d'images par classe",
        "help_num_imagenes": "Choisissez combien d'images de chaque classe vous souhaitez analyser. L'analyse sera plus rapide avec moins d'images.",
        "tiempo_ms": "Temps (ms)",
        "velocidad_imagenes_segundo": "Vitesse (images/seconde)",
        "precision": "Précision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "velocidad_fps": "Vitesse",
        "perfecto": "Parfait",
        "baja_incertidumbre": "Basse Incertitude",
        "alta_incertidumbre": "Haute Incertitude",
        "haga_clic_descargar_pdf": "Télécharger le PDF",
        "ver_matriz_confusion": "Voir la Matrice de Confusion",
        "cargar_imagenes_clase": "Charger des Images de Classe",
        "generar_reporte_analisis": "📄 Générer un Rapport d'Analyse",
        "concordancia_consenso": "Accord de consensus", 
        "compromiso_velocidad_precision":  "⚖️ Vitesse vs. Précision",
        "tiempo_inferencia_modelo": "⏱️ Temps d'Inférence par Modèle",
        "analisis_estadistico": "🧪 Analyse Statistique Détaillée",
        "concordancia_modelos": "🤝 Accord des Modèles",
        "analisis_consenso_tit": "🎯 Analyse de Consensus",
        "analisis_confianza": "📈 Analyse de Confiance",
        "pruebas_tradicionales": "📐 Tests Statistiques Traditionnels (Basés sur des Données Historiques)",
        "ttest_pareado": "📊 Test T Apparié",
        "ztest_proporciones": "📊 Test Z des Proportions",
        "incertidumbre_visualizaciones": "🔬 Analyse d'Incertitude et Visualisations",
        "generar_reporte": "📄 Générer un Rapport Complet",
        "descargar_reporte": "📥 Télécharger le Rapport PDF",
        "evaluacion_real": "🔬 Évaluation des Modèles avec un Jeu de Données Réel",
        "info_evaluacion": "Téléchargez des images de test pour chaque classe pour obtenir des métriques de performance réelles et comparer robustement les modèles.",
        "carga_imagenes": "Téléchargement des Images de Test",
        "iniciar_evaluacion": "🚀 Démarrer l'Évaluation des Modèles",
        "resultados_evaluacion": "📊 Résultats de l'Évaluation",
        "resumen_rendimiento": "Résumé des Performances Générales",
        "prueba_mcnemar": "Test de McNemar (Comparaison des Erreurs)",
        "info_mcnemar": "Ce test détermine si les modèles font différents types d'erreurs. Une valeur p < 0,05 suggère que la différence dans les erreurs est statistiquement significative.",
        "mapa_calor_predicciones": "🔥 Carte Thermique des Prédictions",
        "info_mapa_calor": "Cette carte montre la distribution de toutes les prédictions faites par chaque modèle. Aide à identifier si un modèle est biaisé vers certaines classes.",
        "matrices_confusion": "📊 Matrices de Confusion Détaillées",
        "generar_reporte_evaluacion": "📄 Générer un Rapport d'Évaluation",
        "descargar_reporte_evaluacion": "📥 Télécharger le Rapport d'Évaluation PDF",
        "nota_pie": "💡 **Note :** Ce système est un outil d'aide. Pour un diagnostic définitif, consultez un expert agronome.",
        "desarrollado": "Développé avec ❤️ en utilisant PyTorch et Streamlit",
        "cargando_modelos": "Chargement des modèles...",
        "procesando": "Traitement...",
        "error_modelos": "❌ Impossible de charger les modèles. Vérifiez que les fichiers sont dans le dossier 'models/'.",
        "info_tab1": "👆 Téléchargez et analysez d'abord une image dans l'onglet 'Analyse Individuelle'",
        "info_tab4": "👆 Téléchargez et analysez d'abord une image dans l'onglet 'Analyse Individuelle' pour voir les statistiques.",
        "evaluacion_completada": "Évaluation terminée avec succès !",
        "error_evaluacion": "L'évaluation n'a pas pu être complétée. Assurez-vous de télécharger des images.",
        "generando_pdf": "Génération du PDF...",
        "reporte_generado": "✅ Rapport généré avec succès. Cliquez ci-dessus pour télécharger !",
        "parametros": "Paramètres",
        "precision_validacion": "Précision (Validation)",
        "velocidad": "Vitesse",
        "rapido": "Rapide",
        "medio": "Moyen",
        "lento": "Lent",
        "interpretacion_ttest": "**Interprétation :** p < 0,05 indique une différence statistiquement significative dans la précision moyenne entre les modèles.",
        "interpretacion_ztest": "**Interprétation :** Compare si les proportions de succès sont significativement différentes.",
        "entropia_info": "L'entropie mesure l'incertitude d'une prédiction. Des valeurs plus élevées indiquent que le modèle hésite entre plusieurs classes.",
        "reporte_analisis": "Rapport d'Analyse des Maladies de la Tomate",
        "fecha_generacion": "Date de génération",
        "imagen_analizada": "Image Analysée",
        "error_cargar_imagen": "Erreur lors du chargement de l'image",
        "resultados_analisis_modelo": "Résultats d'Analyse par Modèle",
        "recomendaciones_titulo": "Recommandations",
        "nota_reporte": "Note : Ce rapport est un outil d'aide. Pour un diagnostic définitif, consultez un expert agronome.",
        "reporte_evaluacion": "Rapport d'Évaluation des Modèles",
        "fecha_evaluacion": "Date d'évaluation",
        "total_imagenes": "Total d'images évaluées",
        "resumen_rendimiento_tit": "Résumé des Performances",
        "prueba_mcnemar_tit": "Test de McNemar (Comparaison des Erreurs)",
        "mapa_calor_distribucion": "Carte Thermique de la Distribution des Prédictions",
        "matrices_confusion_tit": "Matrices de Confusion par Modèle",
        "distribucion_predicciones": "Distribution des Prédictions par Modèle",
        "clase_predicha": "Classe Prédite",
        "matriz_confusion": "Matrice de Confusion",
        "curva_roc_auc": "📊 Courbe ROC-AUC et Métrique AUC",
        "info_roc_auc": "La courbe ROC (Receiver Operating Characteristic) illustre la capacité de diagnostic d'un système de classification binaire. Une courbe plus proche du coin supérieur gauche indique une meilleure performance. L'AUC (Area Under the Curve) représente la mesure de performance agrégée pour toutes les classes ; une valeur de 1.0 est parfaite, tandis que 0.5 est aléatoire.",
        "curva_roc_auc_comparativa": "Courbe ROC comparative (micro-moyenne)",
        "clase_real": "Classe Réelle",
        "ejemplo_validacion": "(Exemple avec données de validation)",
        "comparacion": "Comparaison",
        "estadistico_chi": "Statistique Chi-carré",
        "p_value": "P-Valeur",
        "significativo": "Significatif (p < 0,05)",
        "t_statistic": "Statistique t",
        "significativo_ttest": "Significatif",
        "mean_diff": "Différence Moyenne",
        "z_statistic": "Statistique z",
        "prop1": "Prop. Modèle 1",
        "prop2": "Prop. Modèle 2",
        "consenso": "Consensus",
        "entropia": "Entropie",
        "interpretacion": "Interprétation",
        "diferencia_significativa": "Différence Significative",
        "acuerdo": "Accord",
        "diagnostico_consenso": "Diagnostic par Consensus",
        "confianza_promedio": "Confiance Moyenne",
        "niveles_confianza": "Niveaux de Confiance par Modèle",
        "top_diagnosticos": "Top 5 Diagnostics par Consensus",
        "probabilidad_promedio": "Probabilité Moyenne",
        "matriz_acuerdo": "Matrice d'Accord entre Modèles",
        "matriz_probabilidades": "Matrice de Probabilités par Modèle",
        "analisis_probabilidades": "Analyse Détaillée des Probabilités",
        "enfermedad": "Maladie",
        "probabilidad": "Probabilité",
        "severidad_detectada": "Sévérité détectée",
        "recomendaciones": {
            "Alta": [
                "Consultez immédiatement un expert agronome",
                "Isolez les plantes affectées pour éviter la propagation",
                "Envisagez un traitement avec des fongicides spécifiques",
                "Surveillez l'évolution quotidiennement",
                "Documentez l'évolution avec des photographies quotidiennes"
            ],
            "Media": [
                "Appliquez des mesures de contrôle préventives",
                "Améliorez la ventilation de la culture",
                "Revoyez le programme d'irrigation et de fertilisation",
                "Effectuez un suivi hebdomadaire",
                "Envisagez l'application préventive de produits biologiques"
            ],
            "Baja": [
                "Maintenez une surveillance régulière",
                "Appliquez de bonnes pratiques agricoles",
                "Envisagez des traitements préventifs naturels",
                "Revoyez les conditions environnementales de la culture"
            ],
            "Ninguna": [
                "Continuez avec l'entretien régulier",
                "Maintenez les bonnes pratiques actuelles",
                "Effectuez une surveillance préventive périodique",
                "Documentez l'état sain pour référence future"
            ]
        }
    },
    "pt": {
        "titulo_app": "🍅 Sistema de Detecção de Doenças em Folhas de Tomate",
        "subtitulo_app": "Comparação de modelos de Deep Learning para diagnóstico automático",
        "configuracion": "⚙️ Configuração",
        "modelos_disponibles": "📊 Modelos Disponíveis",
        "opciones_visualizacion": "🎨 Opções de Visualização",
        "mostrar_probabilidades": "Mostrar todas as probabilidades",
        "mostrar_comparacion": "Mostrar gráfico comparativo",
        "umbral_confianza": "Limiar de confiança",
        "analisis_individual": "🔍 Análise Individual",
        "comparacion_modelos": "📊 Comparação de Modelos",
        "metricas_estadisticas": "📈 Métricas e Estatísticas",
        "pruebas_estadisticas": "🧪 Testes Estatísticos",
        "evaluacion_modelos": "🔬 Avaliação de Modelos (Batch)",
        "cargar_imagen": "📤 Carregar Imagem",
        "seleccionar_imagen": "Selecione uma imagem de folha de tomate",
        "formato_soportado": "Formatos suportados: JPG, JPEG, PNG",
        "analizar_imagen": "🔬 Analisar Imagem",
        "resultados_analisis": "🎯 Resultados da Análise",
        "diagnostico": "Diagnóstico",
        "confianza": "Confiança",
        "severidad": "Severidade",
        "tiempo": "Tempo",
        "comparacion_predicciones": "🔄 Comparação de Previsões",
        "modelo": "Modelo",
        "prediccion": "Previsão",
        "tiempo_s": "Tempo (s)",
        "analisis_consenso": "📊 Análise de Consenso",
        "nivel_acuerdo": "🤝 Nível de Concordância entre Modelos",
        "metricas_rendimiento": "📈 Métricas de Desempenho",
        "comparacion_metricas_rendimiento": "📊 Comparação de Métricas de Desempenho",
        "tabla_metricas": "📋 Tabela de Métricas Detalhadas",
        "modo_analisis": "Selecione o Modo de Análise",
        "selecciona_modo": "Selecione o modo",
        "analisis_completo": "Análise Completa (1000 imagens)",
        "analisis_personalizado": "Análise Personalizada",
        "num_imagenes_clase": "Número de imagens por classe",
        "help_num_imagenes": "Escolha quantas imagens de cada classe deseja analisar. A análise será mais rápida com menos imagens.",
        "tiempo_ms": "Tempo (ms)",
        "velocidad_imagenes_segundo": "Velocidade (imagens/segundo)",
        "precision": "Precisão",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "velocidad_fps": "Velocidade",
        "perfecto": "Perfeito",
        "baja_incertidumbre": "Baixa Ince&rteza",
        "alta_incertidumbre": "Alta Incerteza",
        "haga_clic_descargar_pdf": "Clique para baixar PDF",
        "ver_matriz_confusion": "Ver Matriz de Confusão",
        "cargar_imagenes_clase": "Carregar Imagens de Classe",
        "generar_reporte_analisis": "📄 Gerar Relatório de Análise",
        "concordancia_consenso": "Concordância de Consenso",
        "compromiso_velocidad_precision":  "⚖️ Velocidade vs. Precisão",
        "tiempo_inferencia_modelo": "⏱️ Tempo de Inferência por Modelo",
        "analisis_estadistico": "🧪 Análise Estatística Detalhada",
        "concordancia_modelos": "🤝 Concordância entre Modelos",
        "analisis_consenso_tit": "🎯 Análise de Consenso",
        "analisis_confianza": "📈 Análise de Confiança",
        "pruebas_tradicionales": "📐 Testes Estatísticos Tradicionais (Baseados em Dados Históricos)",
        "ttest_pareado": "📊 Teste T Pareado",
        "ztest_proporciones": "📊 Teste Z de Proporções",
        "incertidumbre_visualizaciones": "🔬 Análise de Incerteza e Visualizações",
        "generar_reporte": "📄 Gerar Relatório Completo",
        "descargar_reporte": "📥 Baixar Relatório PDF",
        "evaluacion_real": "🔬 Avaliação de Modelos com um Conjunto de Dados Real",
        "info_evaluacion": "Carregue imagens de teste para cada classe para obter métricas de desempenho reais e comparar os modelos de forma robusta.",
        "carga_imagenes": "Carregamento de Imagens de Teste",
        "iniciar_evaluacion": "🚀 Iniciar Avaliação de Modelos",
        "resultados_evaluacion": "📊 Resultados da Avaliação",
        "resumen_rendimiento": "Resumo de Desempenho Geral",
        "prueba_mcnemar": "Teste de McNemar (Comparação de Erros)",
        "info_mcnemar": "Este teste determina se os modelos cometem diferentes tipos de erros. Um p-value < 0,05 sugere que a diferença nos erros é estatisticamente significativa.",
        "mapa_calor_predicciones": "🔥 Mapa de Calor de Previsões",
        "info_mapa_calor": "Este mapa mostra a distribuição de todas as previsões feitas por cada modelo. Ajuda a identificar se um modelo tem viés para determinadas classes.",
        "matrices_confusion": "📊 Matrizes de Confusão Detalhadas",
        "generar_reporte_evaluacion": "📄 Gerar Relatório de Avaliação",
        "descargar_reporte_evaluacion": "📥 Baixar Relatório de Avaliação em PDF",
        "nota_pie": "💡 **Nota:** Este sistema é uma ferramenta de apoio. Para um diagnóstico definitivo, consulte um especialista agrícola.",
        "desarrollado": "Desenvolvido com ❤️ usando PyTorch e Streamlit",
        "cargando_modelos": "Carregando modelos...",
        "procesando": "Processando...",
        "error_modelos": "❌ Não foi possível carregar os modelos. Verifique se os arquivos estão na pasta 'models/'.",
        "info_tab1": "👆 Primeiro carregue e analise uma imagem na aba 'Análise Individual'",
        "info_tab4": "👆 Primeiro carregue e analise uma imagem na aba 'Análise Individual' para ver estatísticas.",
        "evaluacion_completada": "Avaliação concluída com sucesso!",
        "error_evaluacion": "A avaliação não pôde ser concluída. Certifique-se de carregar imagens.",
        "generando_pdf": "Gerando PDF...",
        "reporte_generado": "✅ Relatório gerado com sucesso. Clique acima para baixar!",
        "parametros": "Parâmetros",
        "precision_validacion": "Precisão (Validação)",
        "velocidad_fps": "Velocidade",
        "curva_roc_auc": "📊 Curva ROC-AUC e Métrica AUC",
        "info_roc_auc": "A curva ROC (Receiver Operating Characteristic) ilustra a capacidade de diagnóstico de um sistema classificador binário. Uma curva mais próxima do canto superior esquerdo indica um melhor desempenho. A AUC (Area Under the Curve) representa a medida de desempenho agregada em todas as classes; um valor de 1,0 é perfeito, enquanto 0,5 é aleatório.",
        "curva_roc_auc_comparativa": "Curva ROC Comparativa (Micro-Média)",
        "rapido": "Rápido",
        "medio": "Médio",
        "lento": "Lento",
        "interpretacion_ttest": "**Interpretação:** p < 0,05 indica uma diferença estatisticamente significativa na precisão média entre os modelos.",
        "interpretacion_ztest": "**Interpretação:** Compara se as proporções de acertos são significativamente diferentes.",
        "entropia_info": "A entropia mede a incerteza de uma previsão. Valores mais altos indicam que o modelo está em dúvida entre várias classes.",
        "reporte_analisis": "Relatório de Análise de Doenças do Tomate",
        "fecha_generacion": "Data de geração",
        "imagen_analizada": "Imagem Analisada",
        "error_cargar_imagen": "Erro ao carregar a imagem",
        "resultados_analisis_modelo": "Resultados da Análise por Modelo",
        "recomendaciones_titulo": "Recomendações",
        "nota_reporte": "Nota: Este relatório é uma ferramenta de apoio. Para um diagnóstico definitivo, consulte um especialista agrícola.",
        "reporte_evaluacion": "Relatório de Avaliação de Modelos",
        "fecha_evaluacion": "Data de avaliação",
        "total_imagenes": "Total de imagens avaliadas",
        "resumen_rendimiento_tit": "Resumo de Desempenho",
        "prueba_mcnemar_tit": "Teste de McNemar (Comparação de Erros)",
        "mapa_calor_distribucion": "Mapa de Calor de Distribuição de Previsões",
        "matrices_confusion_tit": "Matrizes de Confusão por Modelo",
        "distribucion_predicciones": "Distribuição de Previsões por Modelo",
        "clase_predicha": "Classe Prevista",
        "matriz_confusion": "Matriz de Confusão",
        "clase_real": "Classe Real",
        "ejemplo_validacion": "(Exemplo com dados de validação)",
        "comparacion": "Comparação",
        "estadistico_chi": "Estatística Qui-quadrado",
        "p_value": "P-Valor",
        "significativo": "Significativo (p < 0,05)",
        "t_statistic": "Estatística t",
        "significativo_ttest": "Significativo",
        "mean_diff": "Diferença Média",
        "z_statistic": "Estatística z",
        "prop1": "Prop. Modelo 1",
        "prop2": "Prop. Modelo 2",
        "consenso": "Consenso",
        "entropia": "Entropia",
        "interpretacion": "Interpretação",
        "diferencia_significativa": "Diferença Significativa",
        "acuerdo": "Concordância",
        "diagnostico_consenso": "Diagnóstico por Consenso",
        "confianza_promedio": "Confiança Média",
        "niveles_confianza": "Níveis de Confiança por Modelo",
        "top_diagnosticos": "Top 5 Diagnósticos por Consenso",
        "probabilidad_promedio": "Probabilidade Média",
        "matriz_acuerdo": "Matriz de Concordância entre Modelos",
        "matriz_probabilidades": "Matriz de Probabilidades por Modelo",
        "analisis_probabilidades": "Análise Detalhada de Probabilidades",
        "enfermedad": "Doença",
        "probabilidad": "Probabilidade",
        "severidad_detectada": "Severidade detectada",
        "recomendaciones": {
            "Alta": [
                "Consulte imediatamente um especialista agrícola",
                "Isole as plantas afetadas para evitar propagação",
                "Considere tratamento com fungicidas específicos",
                "Monitore diariamente a evolução",
                "Documente a evolução com fotografias diárias"
            ],
            "Media": [
                "Aplique medidas preventivas de controle",
                "Melhore a ventilação da cultura",
                "Revise o programa de irrigação e fertilização",
                "Realize acompanhamento semanal",
                "Considere aplicação preventiva de produtos orgânicos"
            ],
            "Baja": [
                "Mantenha vigilância regular",
                "Aplique boas práticas agrícolas",
                "Considere tratamentos preventivos naturais",
                "Revise as condições ambientais da cultura"
            ],
            "Ninguna": [
                "Continue com a manutenção regular",
                "Mantenha as boas práticas atuais",
                "Realize monitoramento preventivo periódico",
                "Documente o estado saudável para referência futura"
            ]
        }
    }
}

# Función para obtener la traducción
def tr(key):
    idioma = st.session_state.idioma
    return TRADUCCIONES[idioma].get(key, key)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .model-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .model-card h4 {
        color: #f39c12;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .prediction-box {
        background-color: rgba(255,255,255,0.1);
        color: #ecf0f1;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
        backdrop-filter: blur(10px);
    }
    .prediction-box strong {
        color: #3498db;
        font-weight: 600;
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Clases de enfermedades (nombres internos, no deben ser traducidos)
CLASES_ENFERMEDADES = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
    'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'healthy'
]

# Información detallada de las enfermedades para la interfaz
INFO_ENFERMEDADES = {
    'Bacterial_spot': {
        'es': 'Mancha Bacteriana', 
        'en': 'Bacterial Spot', 
        'fr': 'Tache bactérienne', 
        'pt': 'Mancha Bacteriana',
        'severity': 'Alta',
        'color': '#FF6B6B'
    },
    'Early_blight': {
        'es': 'Tizón Temprano', 
        'en': 'Early Blight', 
        'fr': 'Mildiou précoce', 
        'pt': 'Míldio Precoce',
        'severity': 'Media',
        'color': '#FFA726'
    },
    'Late_blight': {
        'es': 'Tizón Tardío', 
        'en': 'Late Blight', 
        'fr': 'Mildiou tardif', 
        'pt': 'Míldio Tardio',
        'severity': 'Alta',
        'color': '#FF5252'
    },
    'Leaf_Mold': {
        'es': 'Moho de Hoja', 
        'en': 'Leaf Mold', 
        'fr': 'Moisissure foliaire', 
        'pt': 'Mofo Foliar',
        'severity': 'Media',
        'color': '#FFB74D'
    },
    'Septoria_leaf_spot': {
        'es': 'Mancha de Septoria', 
        'en': 'Septoria Leaf Spot', 
        'fr': 'Tache septorienne', 
        'pt': 'Mancha de Septoria',
        'severity': 'Media',
        'color': '#FF8A65'
    },
    'Spider_mites': {
        'es': 'Ácaros Araña', 
        'en': 'Spider Mites', 
        'fr': 'Acariens', 
        'pt': 'Ácaros-Aranha',
        'severity': 'Baja',
        'color': '#FFCC80'
    },
    'Target_Spot': {
        'es': 'Mancha Diana', 
        'en': 'Target Spot', 
        'fr': 'Tache cible', 
        'pt': 'Mancha Alvo',
        'severity': 'Media',
        'color': '#FF7043'
    },
    'Tomato_Yellow_Leaf_Curl_Virus': {
        'es': 'Virus del Rizado Amarillo', 
        'en': 'Yellow Leaf Curl Virus', 
        'fr': 'Virus de la frisolée jaune', 
        'pt': 'Vírus do Mosaico Amarelo',
        'severity': 'Alta',
        'color': '#FF5722'
    },
    'Tomato_mosaic_virus': {
        'es': 'Virus del Mosaico', 
        'en': 'Mosaic Virus', 
        'fr': 'Virus de la mosaïque', 
        'pt': 'Vírus do Mosaico',
        'severity': 'Alta',
        'color': '#E64A19'
    },
    'healthy': {
        'es': 'Saludable', 
        'en': 'Healthy', 
        'fr': 'Sain', 
        'pt': 'Saudável',
        'severity': 'Ninguna',
        'color': '#4CAF50'
    }
}

# Mapeo de nombres de carpetas a nombres de clases internas
MAPEO_CARPETA_A_CLASE = {
    'Tomato___Bacterial_spot': 'Bacterial_spot',
    'Tomato___Early_blight': 'Early_blight',
    'Tomato___Late_blight': 'Late_blight',
    'Tomato___Leaf_Mold': 'Leaf_Mold',
    'Tomato___Septoria_leaf_spot': 'Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spider_mites',
    'Tomato___Target_Spot': 'Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus': 'Tomato_mosaic_virus',
    'Tomato___healthy': 'healthy'
}

@st.cache_resource
def cargar_modelos():
    """Carga los tres modelos entrenados desde los archivos."""
    diccionario_modelos = {}

    # 1. MobileNetV3
    try:
        mobilenet = models.mobilenet_v3_large()
        mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, len(CLASES_ENFERMEDADES))

        # Cargar pesos - manejo de modelos guardados con DataParallel
        diccionario_estado = torch.load('models/best_model.pth', map_location='cpu')
        # Remover el prefijo 'module.' si existe
        nuevo_diccionario_estado = {}
        for k, v in diccionario_estado.items():
            if k.startswith('module.'):
                nuevo_diccionario_estado[k[7:]] = v
            else:
                nuevo_diccionario_estado[k] = v

        mobilenet.load_state_dict(nuevo_diccionario_estado)
        mobilenet.eval()
        diccionario_modelos['MobileNetV3'] = mobilenet
    except Exception as e:
        st.error(f"{tr('error_modelos')}: {str(e)}")

    # 2. EfficientNetB7
    try:
        efficientnet = models.efficientnet_b7()
        efficientnet.classifier = nn.Linear(2560, len(CLASES_ENFERMEDADES))
        efficientnet.load_state_dict(torch.load('models/plant_disease_model.pth', map_location='cpu'))
        efficientnet.eval()
        diccionario_modelos['EfficientNetB7'] = efficientnet
    except Exception as e:
        st.error(f"{tr('error_modelos')}: {str(e)}")

    # 3. SVM con ResNet50 como extractor de características
    try:
        datos_svm = joblib.load('models/svm_tomato.pkl')
        # Cargar ResNet50 para la extracción de características
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        resnet.fc = nn.Identity() # Se remueve la última capa para obtener el vector de características
        resnet.eval()
        diccionario_modelos['SVM'] = {'svm': datos_svm['svm'], 'feature_extractor': resnet}
    except Exception as e:
        st.error(f"{tr('error_modelos')}: {str(e)}")

    return diccionario_modelos

def preprocesar_imagen(imagen, nombre_modelo):
    """Preprocesa la imagen para que sea compatible con el modelo especificado."""
    if nombre_modelo == 'MobileNetV3':
        transformacion = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif nombre_modelo == 'EfficientNetB7':
        transformacion = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x / 255.0) # Normalización simple
        ])
    else:  # Para SVM, se usa la misma transformación que para MobileNet/ResNet
        transformacion = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return transformacion(imagen).unsqueeze(0)

def predecir_con_modelo(imagen, modelo, nombre_modelo):
    """Realiza una predicción utilizando un modelo específico."""
    tiempo_inicio = time.time()

    with torch.no_grad():
        if nombre_modelo == 'SVM':
            # Extraer características con ResNet50
            tensor_img = preprocesar_imagen(imagen, 'SVM')
            caracteristicas = modelo['feature_extractor'](tensor_img).numpy()
            # Predicción con el clasificador SVM
            prediccion_idx = modelo['svm'].predict(caracteristicas)[0]
            probabilidades = modelo['svm'].predict_proba(caracteristicas)[0]
            # Mapear el índice de predicción de SVM al nombre de la clase
            prediccion = CLASES_ENFERMEDADES[prediccion_idx]
        else:
            # Predicción con redes neuronales
            tensor_img = preprocesar_imagen(imagen, nombre_modelo)
            salidas = modelo(tensor_img)
            probabilidades_tensor = torch.nn.functional.softmax(salidas, dim=1)
            prediccion_idx = torch.argmax(probabilidades_tensor, dim=1).item()
            prediccion = CLASES_ENFERMEDADES[prediccion_idx]
            probabilidades = probabilidades_tensor.numpy()[0]

    tiempo_inferencia = time.time() - tiempo_inicio

    return {
        'prediction': prediccion,
        'probabilities': probabilidades,
        'confidence': float(probabilidades[prediccion_idx]),
        'inference_time': tiempo_inferencia
    }

def realizar_evaluacion_real(diccionario_modelos, imagenes_por_clase=None, base_path='dataset/val'):
    """Evalúa los modelos usando un número específico o todas las imágenes del directorio dataset/val."""
    etiquetas_reales = []
    rutas_imagenes = []
    
    mapa_clase_a_carpeta = {v: k for k, v in MAPEO_CARPETA_A_CLASE.items()}

    for nombre_clase in CLASES_ENFERMEDADES:
        nombre_carpeta = mapa_clase_a_carpeta.get(nombre_clase)
        if not nombre_carpeta:
            continue
        
        ruta_carpeta = Path(base_path) / nombre_carpeta
        if ruta_carpeta.is_dir():
            # Se recolectan todas las rutas válidas primero
            rutas_validas = [
                p for p in ruta_carpeta.iterdir() 
                if p.is_file() and str(p).lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            # Si se especifica un número, se limita la lista de imágenes
            if imagenes_por_clase is not None:
                rutas_a_procesar = rutas_validas[:imagenes_por_clase]
            else:
                rutas_a_procesar = rutas_validas # Se usan todas

            for ruta_imagen in rutas_a_procesar:
                rutas_imagenes.append(ruta_imagen)
                etiquetas_reales.append(nombre_clase)

    if not rutas_imagenes:
        st.warning(f"No se encontraron imágenes en el directorio '{base_path}'. Asegúrate de que la estructura de carpetas es correcta.")
        return None
    
    # El resto de la función permanece exactamente igual...
    predicciones = {nombre_modelo: [] for nombre_modelo in diccionario_modelos.keys()}
    probabilidades = {nombre_modelo: [] for nombre_modelo in diccionario_modelos.keys()}
    
    barra_progreso = st.progress(0, text=tr('procesando'))
    total_imagenes = len(rutas_imagenes)

    for i, ruta_imagen in enumerate(rutas_imagenes):
        imagen = Image.open(ruta_imagen).convert('RGB')
        
        for nombre_modelo, modelo in diccionario_modelos.items():
            resultado = predecir_con_modelo(imagen, modelo, nombre_modelo)
            predicciones[nombre_modelo].append(resultado['prediction'])
            probabilidades[nombre_modelo].append(resultado['probabilities'])
        
        barra_progreso.progress((i + 1) / total_imagenes, text=f"{tr('procesando')} {i+1}/{total_imagenes}...")

    barra_progreso.empty()

    # Calcular métricas... (todo este bloque se mantiene sin cambios)
    resultados = {}
    nombres_modelos = list(diccionario_modelos.keys())

    for nombre_modelo in nombres_modelos:
        preds = predicciones[nombre_modelo]
        acc = accuracy_score(etiquetas_reales, preds)
        mcc = matthews_corrcoef(etiquetas_reales, preds)
        cm = confusion_matrix(etiquetas_reales, preds, labels=CLASES_ENFERMEDADES)
        resultados[nombre_modelo] = {'accuracy': acc, 'mcc': mcc, 'confusion_matrix': cm}

    resultados_mcnemar = {}
    for i in range(len(nombres_modelos)):
        for j in range(i + 1, len(nombres_modelos)):
            modelo1, modelo2 = nombres_modelos[i], nombres_modelos[j]
            preds1, preds2 = np.array(predicciones[modelo1]), np.array(predicciones[modelo2])
            errores1 = (preds1 != np.array(etiquetas_reales))
            errores2 = (preds2 != np.array(etiquetas_reales))
            n01 = np.sum(~errores1 & errores2)
            n10 = np.sum(errores1 & ~errores2)
            numerador = (np.abs(n10 - n01) - 1)**2
            denominador = n10 + n01
            chi2_stat = numerador / denominador if denominador > 0 else 0.0
            p_value = stats.chi2.sf(chi2_stat, 1) if denominador > 0 else 1.0
            resultados_mcnemar[f'{modelo1} vs {modelo2}'] = {'chi2': chi2_stat, 'p_value': p_value}

    resultados['mcnemar_tests'] = resultados_mcnemar
    resultados['true_labels'] = etiquetas_reales
    resultados['predictions'] = predicciones
    resultados['probabilities'] = probabilidades 
    
    return resultados

def crear_grafico_roc(etiquetas_reales, probabilidades, nombres_clases):
    """Crea y devuelve un buffer de imagen con las curvas ROC para todos los modelos."""
    # Binarizar las etiquetas reales
    y_true_bin = label_binarize(etiquetas_reales, classes=nombres_clases)
    n_classes = y_true_bin.shape[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy'])

    for nombre_modelo, probs in probabilidades.items():
        y_score = np.array(probs)
        # Calcular la curva ROC y el área ROC para cada clase
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Calcular micro-promedio de la curva ROC y el área ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Trazar la curva ROC del micro-promedio
        ax.plot(fpr["micro"], tpr["micro"],
                label=f'ROC (micro-avg) {nombre_modelo} (AUC = {roc_auc["micro"]:.2f})',
                linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2) # Línea de azar
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12)
    ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12)
    ax.set_title(tr('curva_roc_auc_comparativa'), fontsize=16)
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

def agregar_pie_de_pagina(lienzo, doc):
    """Agrega un pie de página con una línea y el número de página al PDF."""
    lienzo.saveState()
    lienzo.setFont('Helvetica', 9)
    lienzo.setFillColor(colors.grey)

    # Dibuja el número de página
    texto_numero_pagina = f"Página {lienzo.getPageNumber()}"
    lienzo.drawRightString(A4[0] - inch, 0.5 * inch, texto_numero_pagina)

    # Dibuja la línea horizontal
    lienzo.setStrokeColor(colors.lightgrey)
    lienzo.line(inch, 0.75 * inch, A4[0] - inch, 0.75 * inch)

    lienzo.restoreState()

def generar_reporte_pdf_evaluacion(resultados_evaluacion):
    """
    Genera un PDF con los resultados de la evaluación por lotes,
    incluyendo ahora la curva ROC.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.8*inch, bottomMargin=1*inch)
    elementos_pdf = []
    estilos = getSampleStyleSheet()

    # Estilos personalizados (código existente sin cambios)
    estilo_titulo = ParagraphStyle('CustomTitle', parent=estilos['Title'], fontSize=20, textColor=colors.HexColor('#2c3e50'), spaceAfter=20, alignment=TA_CENTER)
    estilo_h1 = ParagraphStyle('CustomH1', parent=estilos['h1'], fontSize=16, textColor=colors.HexColor('#34495e'), spaceAfter=12, spaceBefore=24, alignment=TA_LEFT, borderPadding=4)
    estilo_h2 = ParagraphStyle('CustomH2', parent=estilos['h2'], fontSize=14, textColor=colors.HexColor('#34495e'), spaceAfter=10, spaceBefore=18, alignment=TA_LEFT)
    estilo_caption = ParagraphStyle('Caption', parent=estilos['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=10)

    # --- Contenido del PDF ---
    elementos_pdf.append(Paragraph(tr("reporte_evaluacion"), estilo_titulo))
    elementos_pdf.append(Paragraph(f"{tr('fecha_evaluacion')}: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", estilos['Normal']))
    elementos_pdf.append(Paragraph(f"{tr('total_imagenes')}: {len(resultados_evaluacion['true_labels'])}", estilos['Normal']))
    elementos_pdf.append(Spacer(1, 0.2*inch))

    # Tabla de resumen de métricas (código existente sin cambios)
    seccion_metricas = [Paragraph(tr("resumen_rendimiento_tit"), estilo_h1)]
    datos_metricas = [[tr('modelo'), tr('precision_validacion'), tr('mcc')]]
    for nombre_modelo, res in resultados_evaluacion.items():
        if isinstance(res, dict) and 'accuracy' in res:
            datos_metricas.append([nombre_modelo, f"{res['accuracy']:.2%}", f"{res['mcc']:.4f}"])
    tabla = Table(datos_metricas, colWidths=[2.5*inch, 2.5*inch, 2.5*inch], hAlign='CENTER')
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3498db')), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.black), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightgrey])
    ]))
    seccion_metricas.append(tabla)
    seccion_metricas.append(Spacer(1, 0.2*inch))
    elementos_pdf.append(KeepTogether(seccion_metricas))

    # Prueba de McNemar (código existente sin cambios)
    seccion_mcnemar = [Paragraph(tr("prueba_mcnemar_tit"), estilo_h2)]
    datos_mcnemar = [[tr('comparacion'), tr('estadistico_chi'), tr('p_value'), tr('significativo')]]
    for comp, res in resultados_evaluacion['mcnemar_tests'].items():
        datos_mcnemar.append([comp, f"{res['chi2']:.4f}", f"{res['p_value']:.4f}", tr('si') if res['p_value'] < 0.05 else tr('no')])
    tabla = Table(datos_mcnemar, hAlign='CENTER')
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#95a5a6')), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.grey), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightgrey])
    ]))
    seccion_mcnemar.append(tabla)
    elementos_pdf.append(KeepTogether(seccion_mcnemar))

    # *** NUEVA SECCIÓN: CURVA ROC-AUC PARA EL PDF ***
    elementos_pdf.append(PageBreak())
    seccion_roc = [
        Paragraph(tr("curva_roc_auc"), estilo_h1),
        Paragraph(tr('info_roc_auc'), estilo_caption)
    ]
    try:
        buffer_roc_pdf = crear_grafico_roc(
            resultados_evaluacion['true_labels'],
            resultados_evaluacion['probabilities'],
            CLASES_ENFERMEDADES
        )
        img_roc = RLImage(buffer_roc_pdf, width=7*inch, height=5.6*inch)
        tabla_img_roc = Table([[img_roc]])
        tabla_img_roc.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        seccion_roc.append(tabla_img_roc)
    except Exception as e:
        seccion_roc.append(Paragraph(f"Error al generar la curva ROC: {e}", estilos['Normal']))
    elementos_pdf.append(KeepTogether(seccion_roc))


    # --- Mapa de Calor de Predicciones --- (código existente sin cambios)
    elementos_pdf.append(PageBreak())
    elementos_pdf.append(Paragraph(tr("mapa_calor_distribucion"), estilo_h1))
    try:
        df_preds = pd.DataFrame(resultados_evaluacion['predictions'])
        nombres_modelos = list(df_preds.columns)
        datos_conteo = {modelo: df_preds[modelo].value_counts() for modelo in nombres_modelos}
        df_conteo = pd.DataFrame(datos_conteo).fillna(0).astype(int)
        df_conteo.index = df_conteo.index.map(lambda x: INFO_ENFERMEDADES[x][st.session_state.idioma])
        df_conteo = df_conteo.reindex([INFO_ENFERMEDADES[c][st.session_state.idioma] for c in CLASES_ENFERMEDADES]).fillna(0)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_conteo, annot=True, fmt='d', cmap='viridis', ax=ax, linewidths=.5)
        ax.set_title(tr('distribucion_predicciones'), fontsize=16)
        ax.set_xlabel(tr('modelo'), fontsize=12)
        ax.set_ylabel(tr('clase_predicha'), fontsize=12)
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150); buf.seek(0); plt.close(fig)
        tabla_img = Table([[RLImage(buf, width=7*inch, height=5.5*inch)]])
        tabla_img.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        elementos_pdf.append(tabla_img)
    except Exception as e:
        elementos_pdf.append(Paragraph(f"{tr('error_mapa_calor')}: {e}", estilos['Normal']))

    # --- Matrices de Confusión --- (código existente sin cambios)
    elementos_pdf.append(PageBreak())
    elementos_pdf.append(Paragraph(tr("matrices_confusion_tit"), estilo_h1))
    etiquetas_clases = [INFO_ENFERMEDADES[c][st.session_state.idioma] for c in CLASES_ENFERMEDADES]
    for nombre_modelo, res in resultados_evaluacion.items():
        if isinstance(res, dict) and 'confusion_matrix' in res:
            titulo = Paragraph(f"<b>{tr('modelo')}: {nombre_modelo}</b>", estilo_h2)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=etiquetas_clases, yticklabels=etiquetas_clases, ax=ax)
            ax.set_title(f'{tr("matriz_confusion")} - {nombre_modelo}', fontsize=14)
            ax.set_xlabel(tr('prediccion')); ax.set_ylabel(tr('clase_real'))
            plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150); buf.seek(0); plt.close(fig)
            tabla_img = Table([[RLImage(buf, width=6*inch, height=5*inch)]])
            tabla_img.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
            elementos_pdf.append(KeepTogether([titulo, tabla_img]))
            elementos_pdf.append(Spacer(1, 0.2*inch))

    doc.build(elementos_pdf, onFirstPage=agregar_pie_de_pagina, onLaterPages=agregar_pie_de_pagina)
    buffer.seek(0)
    return buffer

def realizar_pruebas_estadisticas(predicciones):
    """Realiza pruebas estadísticas para comparar los modelos en una sola predicción."""
    resultados = {}
    nombres_modelos = list(predicciones.keys())

    # 1. Coeficiente Kappa de Cohen para acuerdo entre modelos
    if len(nombres_modelos) >= 2:
        puntuaciones_kappa = {}
        for i in range(len(nombres_modelos)):
            for j in range(i+1, len(nombres_modelos)):
                modelo1, modelo2 = nombres_modelos[i], nombres_modelos[j]
                pred1 = predicciones[modelo1]['prediction']
                pred2 = predicciones[modelo2]['prediction']
                # Para una sola predicción, el kappa será 1 si coinciden, 0 si no.
                puntuaciones_kappa[f"{modelo1} vs {modelo2}"] = 1.0 if pred1 == pred2 else 0.0
        resultados['kappa_scores'] = puntuaciones_kappa

    # 2. Análisis de confianza
    puntuaciones_confianza = {modelo: predicciones[modelo]['confidence'] for modelo in nombres_modelos}
    resultados['confidence_scores'] = puntuaciones_confianza

    # 3. Análisis de consenso (promedio de probabilidades)
    todas_las_predicciones = {}
    for nombre_modelo, resultado in predicciones.items():
        probs = resultado['probabilities']
        for i, enfermedad in enumerate(CLASES_ENFERMEDADES):
            if enfermedad not in todas_las_predicciones:
                todas_las_predicciones[enfermedad] = []
            todas_las_predicciones[enfermedad].append(probs[i])

    probabilidades_consenso = {enfermedad: np.mean(probs) for enfermedad, probs in todas_las_predicciones.items()}
    resultados['consensus'] = max(probabilidades_consenso, key=probabilidades_consenso.get)
    resultados['consensus_confidence'] = probabilidades_consenso[resultados['consensus']]

    return resultados

def realizar_pruebas_estadisticas_tradicionales(historial_precision_modelos=None):
    """
    Realiza pruebas estadísticas tradicionales como t-test y z-test.
    Usa datos simulados basados en las precisiones reportadas de los modelos.
    """
    # Datos simulados basados en precisiones reportadas si no se proveen.
    if historial_precision_modelos is None:
        historial_precision_modelos = {
            'MobileNetV3': np.random.normal(0.952, 0.01, 10),    # Media 95.2%, std 1%
            'EfficientNetB7': np.random.normal(0.978, 0.008, 10), # Media 97.8%, std 0.8%
            'SVM': np.random.normal(0.935, 0.012, 10) # Media 93.5%, std 1.2%
        }

    resultados = {}

    # T-Test pareado entre modelos
    nombres_modelos = list(historial_precision_modelos.keys())
    resultados_test_t = {}

    for i in range(len(nombres_modelos)):
        for j in range(i+1, len(nombres_modelos)):
            modelo1, modelo2 = nombres_modelos[i], nombres_modelos[j]
            prec1 = historial_precision_modelos[modelo1]
            prec2 = historial_precision_modelos[modelo2]

            # T-test pareado
            t_stat, p_value = stats.ttest_rel(prec1, prec2)

            resultados_test_t[f'{modelo1} vs {modelo2}'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'mean_diff': float(np.mean(prec1) - np.mean(prec2))
            }

    resultados['t_tests'] = resultados_test_t

    # Z-test de proporciones (comparando accuracy en un conjunto de validación simulado)
    n_val = 1000 # Número de imágenes de validación simuladas
    resultados_test_z = {}

    for i in range(len(nombres_modelos)):
        for j in range(i+1, len(nombres_modelos)):
            modelo1, modelo2 = nombres_modelos[i], nombres_modelos[j]

            # Calcular éxitos basados en la precisión promedio
            aciertos1 = int(np.mean(historial_precision_modelos[modelo1]) * n_val)
            aciertos2 = int(np.mean(historial_precision_modelos[modelo2]) * n_val)

            p1 = aciertos1 / n_val
            p2 = aciertos2 / n_val

            # Proporción combinada
            p_combinada = (aciertos1 + aciertos2) / (2 * n_val)

            # Error estándar
            ee = np.sqrt(p_combinada * (1 - p_combinada) * (2/n_val))

            # Estadístico Z
            z = (p1 - p2) / ee if ee > 0 else 0

            # P-valor (dos colas)
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            resultados_test_z[f'{modelo1} vs {modelo2}'] = {
                'z_statistic': float(z),
                'p_value': float(p_value),
                'prop1': float(p1),
                'prop2': float(p2),
                'significant': p_value < 0.05
            }

    resultados['z_tests'] = resultados_test_z

    return resultados

def crear_graficos_estadisticos(predicciones):
    """Crea visualizaciones estadísticas para el análisis de una sola imagen."""
    graficos = {}

    # 1. Gráfico de intervalos de confianza
    fig, ax = plt.subplots(figsize=(10, 6))
    modelos = list(predicciones.keys())
    confianzas = [predicciones[m]['confidence'] for m in modelos]
    colores = ['#3498db', '#e74c3c', '#2ecc71']

    barras = ax.bar(modelos, confianzas, color=colores, alpha=0.7)
    ax.axhline(y=0.7, color='red', linestyle='--', label=f"{tr('umbral_confianza')} (70%)")
    ax.set_ylim(0, 1)
    ax.set_ylabel(tr('confianza'), fontsize=12)
    ax.set_title(tr('niveles_confianza'), fontsize=14, fontweight='bold')
    ax.legend()

    # Añadir valores en las barras
    for barra, conf in zip(barras, confianzas):
        altura = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2., altura + 0.01,
                f'{conf:.2%}', ha='center', va='bottom')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    graficos['confidence_comparison'] = buf
    plt.close()

    # 2. Matriz de calor de probabilidades
    fig, ax = plt.subplots(figsize=(12, 8))

    # Crear matriz de probabilidades
    matriz_probabilidades = []
    for modelo in predicciones:
        matriz_probabilidades.append(predicciones[modelo]['probabilities'])

    matriz_probabilidades = np.array(matriz_probabilidades)

    # Crear mapa de calor
    sns.heatmap(matriz_probabilidades,
                xticklabels=[INFO_ENFERMEDADES[d][st.session_state.idioma][:15] for d in CLASES_ENFERMEDADES],
                yticklabels=modelos,
                cmap='YlOrRd',
                annot=True,
                fmt='.2%',
                cbar_kws={'label': tr('probabilidad')},
                ax=ax)

    ax.set_title(tr('matriz_probabilidades'), fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    graficos['probability_heatmap'] = buf
    plt.close()

    # 3. Gráfico de matriz de confusión (ejemplo)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Simular una matriz de confusión para el mejor modelo
    np.random.seed(42)
    num_clases = len(CLASES_ENFERMEDADES)
    mc = np.zeros((num_clases, num_clases), dtype=int)

    # Llenar diagonal principal con valores altos (aciertos)
    for i in range(num_clases):
        mc[i, i] = np.random.randint(85, 98)
        # Distribuir algunos errores
        for j in range(num_clases):
            if i != j:
                mc[i, j] = np.random.randint(0, 5)

    sns.heatmap(mc,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[INFO_ENFERMEDADES[d][st.session_state.idioma][:10] for d in CLASES_ENFERMEDADES],
                yticklabels=[INFO_ENFERMEDADES[d][st.session_state.idioma][:10] for d in CLASES_ENFERMEDADES],
                ax=ax)
    ax.set_title(f"{tr('matriz_confusion')} - EfficientNetB7 ({tr('ejemplo_validacion')})", fontsize=14)
    ax.set_xlabel(tr('prediccion'))
    ax.set_ylabel(tr('clase_real'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    graficos['confusion_matrix'] = buf
    plt.close()

    return graficos

def crear_graficos_adicionales_para_pdf(predicciones, resultados_estadisticos):
    """Crea gráficos adicionales específicos para el reporte PDF."""
    graficos = {}

    # 1. Gráfico de consenso
    fig, ax = plt.subplots(figsize=(10, 6))

    # Recopilar todas las predicciones
    todas_las_predicciones = {}
    for nombre_modelo, resultado in predicciones.items():
        probs = resultado['probabilities']
        for i, enfermedad in enumerate(CLASES_ENFERMEDADES):
            if enfermedad not in todas_las_predicciones:
                todas_las_predicciones[enfermedad] = []
            todas_las_predicciones[enfermedad].append(probs[i])

    # Calcular promedio y desviación estándar
    datos_consenso = []
    for enfermedad, probs in todas_las_predicciones.items():
        datos_consenso.append({
            'disease': INFO_ENFERMEDADES[enfermedad][st.session_state.idioma],
            'mean': np.mean(probs),
            'std': np.std(probs)
        })

    # Ordenar por probabilidad promedio y tomar el top 5
    datos_consenso = sorted(datos_consenso, key=lambda x: x['mean'], reverse=True)[:5]

    # Crear gráfico
    enfermedades = [d['disease'] for d in datos_consenso]
    medias = [d['mean'] for d in datos_consenso]
    desviaciones = [d['std'] for d in datos_consenso]

    barras = ax.bar(enfermedades, medias, yerr=desviaciones, capsize=5, color='skyblue', edgecolor='navy', alpha=0.7)
    ax.set_ylabel(tr('probabilidad_promedio'), fontsize=12)
    ax.set_title(tr('top_diagnosticos'), fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(medias) * 1.2 if medias else 1)

    # Añadir valores
    for barra, media in zip(barras, medias):
        altura = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2., altura + 0.01,
                f'{media:.2%}', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    graficos['consensus_plot'] = buf
    plt.close()

    # 2. Gráfico de matriz de acuerdo entre modelos
    fig, ax = plt.subplots(figsize=(8, 8))

    nombres_modelos = list(predicciones.keys())
    matriz_acuerdo = np.zeros((len(nombres_modelos), len(nombres_modelos)))

    for i, modelo1 in enumerate(nombres_modelos):
        for j, modelo2 in enumerate(nombres_modelos):
            pred1 = predicciones[modelo1]['prediction']
            pred2 = predicciones[modelo2]['prediction']
            matriz_acuerdo[i, j] = 1.0 if pred1 == pred2 else 0.0

    sns.heatmap(matriz_acuerdo,
                xticklabels=nombres_modelos,
                yticklabels=nombres_modelos,
                annot=True,
                fmt='.0f',
                cmap='Blues',
                vmin=0,
                vmax=1,
                cbar_kws={'label': f"{tr('acuerdo')} (1={tr('si')}, 0={tr('no')})"},
                ax=ax)

    ax.set_title(tr('matriz_acuerdo'), fontsize=14, fontweight='bold')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    graficos['agreement_matrix'] = buf
    plt.close()

    return graficos

def generar_reporte_pdf(predicciones, buffer_imagen, resultados_estadisticos, pruebas_tradicionales=None):
    """Genera un reporte PDF completo del análisis de una sola imagen con todos los gráficos."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elementos_pdf = []
    estilos = getSampleStyleSheet()

    # Título personalizado
    estilo_titulo = ParagraphStyle(
        'CustomTitle',
        parent=estilos['Title'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    # Agregar título
    elementos_pdf.append(Paragraph(tr("reporte_analisis"), estilo_titulo))
    elementos_pdf.append(Spacer(1, 0.2*inch))

    # Información del reporte
    estilo_info = ParagraphStyle(
        'InfoStyle',
        parent=estilos['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#7f8c8d')
    )
    elementos_pdf.append(Paragraph(f"{tr('fecha_generacion')}: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", estilo_info))
    elementos_pdf.append(Spacer(1, 0.3*inch))

    # SECCIÓN 1: Imagen analizada
    elementos_pdf.append(Paragraph(f"<b>{tr('imagen_analizada')}:</b>", estilos['Heading2']))
    if buffer_imagen:
        try:
            # Asegurarse de que el buffer esté al inicio
            buffer_imagen.seek(0)
            # Crear imagen más grande para mejor visualización
            img = RLImage(buffer_imagen, width=4*inch, height=4*inch)
            # Centrar la imagen usando una tabla
            tabla_img = Table([[img]], colWidths=[4*inch])
            tabla_img.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
            elementos_pdf.append(tabla_img)
            elementos_pdf.append(Spacer(1, 0.3*inch))
        except Exception as e:
            elementos_pdf.append(Paragraph(f"{tr('error_cargar_imagen')}: {str(e)}", estilos['Normal']))
            elementos_pdf.append(Spacer(1, 0.3*inch))
    else:
        elementos_pdf.append(Paragraph(f"{tr('error_cargar_imagen')}", estilos['Normal']))
        elementos_pdf.append(Spacer(1, 0.3*inch))

    # SECCIÓN 2: Resultados por modelo
    elementos_pdf.append(Paragraph(tr("resultados_analisis_modelo"), estilos['Heading1']))
    elementos_pdf.append(Spacer(1, 0.2*inch))

    # Tabla de resultados principales
    datos = [[tr('modelo'), tr('diagnostico'), tr('confianza'), tr('tiempo_s')]]
    for nombre_modelo, resultado in predicciones.items():
        datos.append([
            nombre_modelo,
            INFO_ENFERMEDADES[resultado['prediction']][st.session_state.idioma],
            f"{resultado['confidence']:.2%}",
            f"{resultado['inference_time']:.3f}"
        ])

    tabla = Table(datos, colWidths=[2.5*inch, 2.5*inch, 1.5*inch, 1.5*inch])
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elementos_pdf.append(tabla)
    elementos_pdf.append(Spacer(1, 0.5*inch))

    # SECCIÓN 3: Gráficos de análisis estadístico
    elementos_pdf.append(PageBreak())
    elementos_pdf.append(Paragraph(tr("analisis_estadistico"), estilos['Heading1']))
    elementos_pdf.append(Spacer(1, 0.2*inch))

    # Generar y agregar gráficos
    graficos = crear_graficos_estadisticos(predicciones)
    graficos_adicionales = crear_graficos_adicionales_para_pdf(predicciones, resultados_estadisticos)

    # Gráfico de comparación de confianza
    elementos_pdf.append(Paragraph(f"<b>{tr('niveles_confianza')}</b>", estilos['Heading3']))
    if 'confidence_comparison' in graficos:
        graficos['confidence_comparison'].seek(0)
        img_conf = RLImage(graficos['confidence_comparison'], width=5*inch, height=3*inch)
        tabla_img = Table([[img_conf]], colWidths=[5*inch])
        tabla_img.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        elementos_pdf.append(tabla_img)
        elementos_pdf.append(Spacer(1, 0.3*inch))

    # Gráfico de consenso
    elementos_pdf.append(Paragraph(f"<b>{tr('analisis_consenso_tit')}</b>", estilos['Heading3']))
    if 'consensus_plot' in graficos_adicionales:
        graficos_adicionales['consensus_plot'].seek(0)
        img_consensus = RLImage(graficos_adicionales['consensus_plot'], width=5*inch, height=3*inch)
        tabla_img = Table([[img_consensus]], colWidths=[5*inch])
        tabla_img.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        elementos_pdf.append(tabla_img)
        elementos_pdf.append(Spacer(1, 0.3*inch))

    # Matriz de acuerdo
    elementos_pdf.append(PageBreak())
    elementos_pdf.append(Paragraph(f"<b>{tr('matriz_acuerdo')}</b>", estilos['Heading3']))
    if 'agreement_matrix' in graficos_adicionales:
        graficos_adicionales['agreement_matrix'].seek(0)
        img_agreement = RLImage(graficos_adicionales['agreement_matrix'], width=4*inch, height=4*inch)
        tabla_img = Table([[img_agreement]], colWidths=[4*inch])
        tabla_img.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        elementos_pdf.append(tabla_img)
        elementos_pdf.append(Spacer(1, 0.3*inch))

    # Matriz de calor de probabilidades
    elementos_pdf.append(Paragraph(f"<b>{tr('matriz_probabilidades')}</b>", estilos['Heading3']))
    if 'probability_heatmap' in graficos:
        graficos['probability_heatmap'].seek(0)
        img_heat = RLImage(graficos['probability_heatmap'], width=6*inch, height=4*inch)
        tabla_img = Table([[img_heat]], colWidths=[6*inch])
        tabla_img.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        elementos_pdf.append(tabla_img)
        elementos_pdf.append(Spacer(1, 0.3*inch))

    # Matriz de confusión
    elementos_pdf.append(PageBreak())
    elementos_pdf.append(Paragraph(f"<b>{tr('matriz_confusion')} {tr('ejemplo_validacion')}</b>", estilos['Heading3']))
    if 'confusion_matrix' in graficos:
        graficos['confusion_matrix'].seek(0)
        img_cm = RLImage(graficos['confusion_matrix'], width=5.5*inch, height=4.5*inch)
        tabla_img = Table([[img_cm]], colWidths=[5.5*inch])
        tabla_img.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        elementos_pdf.append(tabla_img)
        elementos_pdf.append(Spacer(1, 0.3*inch))

    # SECCIÓN 4: Análisis estadístico
    elementos_pdf.append(PageBreak())
    elementos_pdf.append(Paragraph(tr("analisis_estadistico"), estilos['Heading1']))
    elementos_pdf.append(Spacer(1, 0.2*inch))

    # Consenso
    elementos_pdf.append(Paragraph(f"<b>{tr('diagnostico_consenso')}:</b> {INFO_ENFERMEDADES[resultados_estadisticos['consensus']][st.session_state.idioma]} "
                              f"({tr('confianza_promedio')}: {resultados_estadisticos['consensus_confidence']:.2%})",
                              estilos['Normal']))
    elementos_pdf.append(Spacer(1, 0.2*inch))

    # Acuerdo entre modelos
    if 'kappa_scores' in resultados_estadisticos:
        elementos_pdf.append(Paragraph(f"<b>{tr('nivel_acuerdo')}:</b>", estilos['Normal']))
        for comparacion, puntuacion in resultados_estadisticos['kappa_scores'].items():
            texto_acuerdo = tr("acuerdo_perfecto") if puntuacion == 1.0 else tr("desacuerdo")
            elementos_pdf.append(Paragraph(f"• {comparacion}: {texto_acuerdo}", estilos['Normal']))
        elementos_pdf.append(Spacer(1, 0.2*inch))

    # Análisis de entropía (incertidumbre)
    elementos_pdf.append(Paragraph(f"<b>{tr('incertidumbre_entropia')}:</b>", estilos['Normal']))
    datos_entropia = []
    for nombre_modelo, resultado in predicciones.items():
        probs = resultado['probabilities']
        entropia = -np.sum(probs * np.log2(probs + 1e-10))
        datos_entropia.append([
            nombre_modelo,
            f"{entropia:.4f}",
            tr("baja_incertidumbre") if entropia < 1 else tr("alta_incertidumbre")
        ])

    tabla_entropia = Table([[tr('modelo'), tr('entropia'), tr('interpretacion')]] + datos_entropia)
    tabla_entropia.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95a5a6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    elementos_pdf.append(tabla_entropia)
    elementos_pdf.append(Spacer(1, 0.3*inch))

    # SECCIÓN 5: Pruebas estadísticas tradicionales
    if pruebas_tradicionales:
        elementos_pdf.append(PageBreak())
        elementos_pdf.append(Paragraph(tr("pruebas_tradicionales"), estilos['Heading1']))
        elementos_pdf.append(Spacer(1, 0.2*inch))

        # T-Test pareado
        if 't_tests' in pruebas_tradicionales:
            elementos_pdf.append(Paragraph(f"<b>{tr('ttest_pareado')}</b>", estilos['Heading2']))
            elementos_pdf.append(Paragraph(tr("info_ttest"), estilo_info))
            elementos_pdf.append(Spacer(1, 0.1*inch))

            datos_test_t = [[tr('comparacion'), tr('t_statistic'), tr('p_value'), tr('interpretacion')]]
            for comp, resultado in pruebas_tradicionales['t_tests'].items():
                interpretacion = tr("diferencia_significativa") if resultado['significant'] else tr("sin_diferencia_significativa")
                datos_test_t.append([
                    comp,
                    f"{resultado['t_statistic']:.4f}",
                    f"{resultado['p_value']:.4f}",
                    interpretacion
                ])

            tabla_t = Table(datos_test_t, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 2.5*inch])
            tabla_t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95a5a6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            elementos_pdf.append(tabla_t)
            elementos_pdf.append(Spacer(1, 0.3*inch))

        # Z-Test de proporciones
        if 'z_tests' in pruebas_tradicionales:
            elementos_pdf.append(Paragraph(f"<b>{tr('ztest_proporciones')}</b>", estilos['Heading2']))
            elementos_pdf.append(Paragraph(tr("info_ztest"), estilo_info))
            elementos_pdf.append(Spacer(1, 0.1*inch))

            datos_test_z = [[tr('comparacion'), tr('z_statistic'), tr('p_value'), tr('prop1'), tr('prop2')]]
            for comp, resultado in pruebas_tradicionales['z_tests'].items():
                datos_test_z.append([
                    comp,
                    f"{resultado['z_statistic']:.4f}",
                    f"{resultado['p_value']:.4f}",
                    f"{resultado['prop1']:.3f}",
                    f"{resultado['prop2']:.3f}"
                ])

            tabla_z = Table(datos_test_z, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1*inch, 1*inch])
            tabla_z.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95a5a6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            elementos_pdf.append(tabla_z)
            elementos_pdf.append(Spacer(1, 0.3*inch))

    # SECCIÓN 6: Top 5 probabilidades por modelo
    elementos_pdf.append(PageBreak())
    elementos_pdf.append(Paragraph(tr("analisis_probabilidades"), estilos['Heading1']))
    elementos_pdf.append(Spacer(1, 0.2*inch))

    for nombre_modelo, resultado in predicciones.items():
        elementos_pdf.append(Paragraph(f"<b>{nombre_modelo}</b>", estilos['Heading2']))

        # Crear tabla de probabilidades
        probabilidades_con_enfermedades = [(INFO_ENFERMEDADES[CLASES_ENFERMEDADES[i]][st.session_state.idioma], prob)
                                           for i, prob in enumerate(resultado['probabilities'])]
        probabilidades_ordenadas = sorted(probabilidades_con_enfermedades, key=lambda x: x[1], reverse=True)[:5]

        datos_probabilidades = [[tr('enfermedad'), tr('probabilidad')]]
        for enfermedad, prob in probabilidades_ordenadas:
            datos_probabilidades.append([enfermedad, f"{prob:.2%}"])

        tabla_probabilidades = Table(datos_probabilidades, colWidths=[3*inch, 2*inch])
        tabla_probabilidades.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95a5a6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        elementos_pdf.append(tabla_probabilidades)
        elementos_pdf.append(Spacer(1, 0.3*inch))

    # SECCIÓN 7: Recomendaciones
    elementos_pdf.append(PageBreak())
    elementos_pdf.append(Paragraph(tr("recomendaciones_titulo"), estilos['Heading1']))
    elementos_pdf.append(Spacer(1, 0.2*inch))

    enfermedad_consenso = resultados_estadisticos['consensus']
    severidad = INFO_ENFERMEDADES[enfermedad_consenso]['severity']

    # Información sobre la enfermedad detectada
    color_severidad = {
        'Alta': colors.red,
        'Media': colors.orange,
        'Baja': colors.yellow,
        'Ninguna': colors.green
    }

    estilo_severidad = ParagraphStyle(
        'SeverityStyle',
        parent=estilos['Normal'],
        fontSize=12,
        textColor=color_severidad.get(severidad, colors.black),
        fontName='Helvetica-Bold'
    )

    elementos_pdf.append(Paragraph(f"{tr('severidad_detectada')}: {severidad}", estilo_severidad))
    elementos_pdf.append(Spacer(1, 0.2*inch))

    if severidad in tr("recomendaciones"):
        for rec in tr("recomendaciones")[severidad]:
            elementos_pdf.append(Paragraph(f"• {rec}", estilos['Normal']))

    elementos_pdf.append(Spacer(1, 0.5*inch))
    elementos_pdf.append(Paragraph(f"<i>{tr('nota_reporte')}</i>", estilo_info))

    # Construir PDF
    doc.build(elementos_pdf)
    buffer.seek(0)
    return buffer

def principal():
    """Función principal que ejecuta la aplicación Streamlit."""
    # Inicializar estado de sesión para idioma
    if 'idioma' not in st.session_state:
        st.session_state.idioma = "es"
    
    # Selector de idioma en la barra lateral
    with st.sidebar:
        #st.title(tr("configuracion"))
        idioma = st.selectbox(
            "🌐 Idioma / Language / Langue / Idioma",
            options=["es", "en", "fr", "pt"],
            format_func=lambda x: {"es": "Español", "en": "English", "fr": "Français", "pt": "Português"}[x],
            key='idioma_selector'
        )
        st.session_state.idioma = idioma

        # Información de los modelos
        st.markdown(f"### {tr('modelos_disponibles')}")

        info_modelos = {
            'MobileNetV3': {'params': '5.4M', 'accuracy': '95.2%', 'speed': tr('rapido')},
            'EfficientNetB7': {'params': '66M', 'accuracy': '97.8%', 'speed': tr('lento')},
            'SVM': {'params': '25M (ResNet50)', 'accuracy': '93.5%', 'speed': tr('medio')}
        }

        for modelo, info in info_modelos.items():
            with st.expander(f"📱 {modelo}"):
                st.write(f"**{tr('parametros')}:** {info['params']}")
                st.write(f"**{tr('precision_validacion')}:** {info['accuracy']}")
                st.write(f"**{tr('velocidad')}:** {info['speed']}")

        st.markdown("---")

        # Opciones de visualización
        st.markdown(f"### {tr('opciones_visualizacion')}")
        mostrar_probabilidades = st.checkbox(tr("mostrar_probabilidades"), value=True)
        mostrar_comparacion = st.checkbox(tr("mostrar_comparacion"), value=True)
        umbral_confianza = st.slider(tr("umbral_confianza"), 0.0, 1.0, 0.7)

    st.markdown(f"""
    <div class="main-header">
        <h1>{tr("titulo_app")}</h1>
        <p>{tr("subtitulo_app")}</p>
    </div>
    """, unsafe_allow_html=True)

    # Cargar modelos
    diccionario_modelos = cargar_modelos()
    if not diccionario_modelos:
        st.error(tr("error_modelos"))
        return

    # Pestañas principales
    tab1, tab2, tab3, tab4, tab_eval = st.tabs([
        tr("analisis_individual"), 
        tr("comparacion_modelos"), 
        tr("metricas_estadisticas"), 
        tr("pruebas_estadisticas"), 
        tr("evaluacion_modelos")
    ])

    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"### {tr('cargar_imagen')}")
            archivo_cargado = st.file_uploader(
                tr("seleccionar_imagen"),
                type=['jpg', 'jpeg', 'png'],
                help=tr("formato_soportado")
            )

            if archivo_cargado is not None:
                # 1. Leer los bytes de la imagen y guardarlos en el estado de la sesión
                bytes_imagen = archivo_cargado.getvalue()
                st.session_state['uploaded_image_bytes'] = bytes_imagen

                # 2. Abrir la imagen desde los bytes para mostrarla y procesarla
                imagen = Image.open(BytesIO(bytes_imagen)).convert('RGB')
                st.image(imagen, caption=tr("imagen_cargada"), use_column_width=True)

                # 3. Guardar la imagen en el estado para el análisis
                st.session_state['image_to_analyze'] = imagen

                # Botón de análisis
                if st.button(tr("analizar_imagen"), type="primary"):
                    with st.spinner(tr("procesando")):
                        st.session_state['predicciones'] = {}
                        # Usar la imagen guardada en el estado
                        imagen_a_procesar = st.session_state['image_to_analyze']
                        for nombre_modelo, modelo in diccionario_modelos.items():
                            resultado = predecir_con_modelo(imagen_a_procesar, modelo, nombre_modelo)
                            st.session_state['predicciones'][nombre_modelo] = resultado

        with col2:
            if 'predicciones' in st.session_state and st.session_state['predicciones']:
                st.markdown(f"### {tr('resultados_analisis')}")

                for nombre_modelo, resultado in st.session_state['predicciones'].items():
                    enfermedad = resultado['prediction']
                    confianza = resultado['confidence']
                    tiempo_empleado = resultado['inference_time']

                    # Tarjeta de resultados para cada modelo
                    st.markdown(f"""
                    <div class="model-card">
                        <h4>🤖 {nombre_modelo}</h4>
                        <div class="prediction-box">
                            <strong>{tr('diagnostico')}:</strong> {INFO_ENFERMEDADES[enfermedad][st.session_state.idioma]}<br>
                            <strong>{tr('confianza')}:</strong> {confianza:.2%}<br>
                            <strong>{tr('severidad')}:</strong> {INFO_ENFERMEDADES[enfermedad]['severity']}<br>
                            <strong>{tr('tiempo')}:</strong> {tiempo_empleado:.3f}s
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Mostrar todas las probabilidades si está activado
                    if mostrar_probabilidades:
                        df_probabilidades = pd.DataFrame({
                            tr('enfermedad'): [INFO_ENFERMEDADES[cls][st.session_state.idioma] for cls in CLASES_ENFERMEDADES],
                            tr('probabilidad'): resultado['probabilities']
                        }).sort_values(tr('probabilidad'), ascending=False)

                        fig = px.bar(
                            df_probabilidades.head(5),
                            x=tr('probabilidad'),
                            y=tr('enfermedad'),
                            orientation='h',
                            color=tr('probabilidad'),
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if 'predicciones' in st.session_state and st.session_state['predicciones']:
            st.markdown(f"### {tr('comparacion_predicciones')}")

            # Tabla comparativa
            datos_comparacion = []
            for nombre_modelo, resultado in st.session_state['predicciones'].items():
                datos_comparacion.append({
                    tr('modelo'): nombre_modelo,
                    tr('prediccion'): INFO_ENFERMEDADES[resultado['prediction']][st.session_state.idioma],
                    tr('confianza'): f"{resultado['confidence']:.2%}",
                    tr('tiempo_s'): f"{resultado['inference_time']:.3f}"
                })

            df_comparacion = pd.DataFrame(datos_comparacion)
            st.dataframe(df_comparacion, use_container_width=True)

            # Gráfico de consenso
            if mostrar_comparacion:
                st.markdown(f"### {tr('analisis_consenso')}")

                # Recopilar todas las predicciones
                todas_las_predicciones = {}
                for nombre_modelo, resultado in st.session_state['predicciones'].items():
                    probs = resultado['probabilities']
                    for i, enfermedad in enumerate(CLASES_ENFERMEDADES):
                        if enfermedad not in todas_las_predicciones:
                            todas_las_predicciones[enfermedad] = []
                        todas_las_predicciones[enfermedad].append(probs[i])

                # Calcular promedio de probabilidades
                datos_consenso = []
                for enfermedad, probs in todas_las_predicciones.items():
                    datos_consenso.append({
                        tr('enfermedad'): INFO_ENFERMEDADES[enfermedad][st.session_state.idioma],
                        tr('probabilidad_promedio'): np.mean(probs),
                        tr('desviacion_estandar'): np.std(probs)
                    })

                df_consenso = pd.DataFrame(datos_consenso).sort_values(tr('probabilidad_promedio'), ascending=False)

                # Gráfico de barras con error
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_consenso[tr('enfermedad')][:5],
                    y=df_consenso[tr('probabilidad_promedio')][:5],
                    error_y=dict(type='data', array=df_consenso[tr('desviacion_estandar')][:5]),
                    marker_color='lightblue',
                    name=tr('consenso')
                ))
                fig.update_layout(
                    title=tr('top_diagnosticos'),
                    xaxis_title=tr('enfermedad'),
                    yaxis_title=tr('probabilidad_promedio'),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Matriz de acuerdo entre modelos
                st.markdown(f"### {tr('nivel_acuerdo')}")

                nombres_modelos = list(st.session_state['predicciones'].keys())
                matriz_acuerdo = np.zeros((len(nombres_modelos), len(nombres_modelos)))

                for i, modelo1 in enumerate(nombres_modelos):
                    for j, modelo2 in enumerate(nombres_modelos):
                        pred1 = st.session_state['predicciones'][modelo1]['prediction']
                        pred2 = st.session_state['predicciones'][modelo2]['prediction']
                        matriz_acuerdo[i, j] = 1.0 if pred1 == pred2 else 0.0

                fig_mapa_calor = go.Figure(data=go.Heatmap(
                    z=matriz_acuerdo,
                    x=nombres_modelos,
                    y=nombres_modelos,
                    colorscale='Blues',
                    text=matriz_acuerdo,
                    texttemplate='%{text}',
                    textfont={"size": 16}
                ))
                fig_mapa_calor.update_layout(
                    title=tr('matriz_acuerdo'),
                    height=400
                )
                st.plotly_chart(fig_mapa_calor, use_container_width=True)
        else:
            st.info(tr("info_tab1"))

    with tab3:
        st.markdown(f"### {tr('metricas_rendimiento')}")

        # Métricas simuladas (en un entorno real, vendrían de la validación del modelo)
        datos_metricas = {
            tr('modelo'): ['MobileNetV3', 'EfficientNetB7', 'SVM'],
            tr('precision'): [0.952, 0.978, 0.935],
            tr('recall'): [0.948, 0.975, 0.930],
            tr('f1_score'): [0.950, 0.976, 0.932],
            tr('velocidad_fps'): [45, 12, 25]
        }

        df_metricas = pd.DataFrame(datos_metricas)

        # Gráfico de radar
        categorias = [tr('precision'), tr('recall'), tr('f1_score')]

        fig_radar = go.Figure()

        for idx, modelo in enumerate(df_metricas[tr('modelo')]):
            valores = df_metricas.iloc[idx][categorias].tolist()
            valores += valores[:1]  # Cerrar el polígono

            fig_radar.add_trace(go.Scatterpolar(
                r=valores,
                theta=categorias + categorias[:1],
                fill='toself',
                name=modelo
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.9, 1.0]
                )),
            showlegend=True,
            title=tr("comparacion_metricas_rendimiento")
        )

        st.plotly_chart(fig_radar, use_container_width=True)

        # Tabla de métricas detalladas
        st.markdown(f"### {tr('tabla_metricas')}")
        st.dataframe(
            df_metricas.style.highlight_max(axis=0, subset=[tr('precision'), tr('recall'), tr('f1_score'), tr('velocidad_fps')]),
            use_container_width=True
        )

        # Gráfico de trade-off velocidad vs precisión
        col1, col2 = st.columns(2)

        with col1:
            fig_compromiso = px.scatter(
                df_metricas,
                x=tr('velocidad_fps'),
                y=tr('precision'),
                size=tr('f1_score'),
                color=tr('modelo'),
                hover_data=[tr('recall')],
                title=tr('compromiso_velocidad_precision'),
                labels={tr('velocidad_fps'): tr('velocidad_imagenes_segundo')}
            )
            fig_compromiso.update_traces(marker=dict(size=20))
            st.plotly_chart(fig_compromiso, use_container_width=True)

        with col2:
            # Tiempo de inferencia promedio de la última predicción
            if 'predicciones' in st.session_state:
                tiempos_inferencia = []
                for nombre_modelo, resultado in st.session_state['predicciones'].items():
                    tiempos_inferencia.append({
                        tr('modelo'): nombre_modelo,
                        tr('tiempo_ms'): resultado['inference_time'] * 1000
                    })
                df_tiempos = pd.DataFrame(tiempos_inferencia)
                fig_tiempos = px.bar(df_tiempos, x=tr('modelo'), y=tr('tiempo_ms'), title=tr('tiempo_inferencia_modelo'))
                st.plotly_chart(fig_tiempos, use_container_width=True)

    with tab4:
        st.markdown(f"### {tr('analisis_estadistico')}")

        if 'predicciones' in st.session_state and st.session_state['predicciones']:
            # Realizar pruebas estadísticas con los resultados actuales
            resultados_estadisticos = realizar_pruebas_estadisticas(st.session_state['predicciones'])

            # Realizar pruebas estadísticas tradicionales con datos simulados
            resultados_tradicionales = realizar_pruebas_estadisticas_tradicionales()

            # Sección 1: Pruebas sobre la predicción actual
            st.markdown(f"#### {tr('concordancia_consenso')}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"##### {tr('concordancia_modelos')}")

                # Mostrar Kappa de Cohen
                if 'kappa_scores' in resultados_estadisticos:
                    df_kappa = pd.DataFrame([
                        {tr('comparacion'): comp, tr('acuerdo'): tr('perfecto') if score == 1.0 else tr('desacuerdo')}
                        for comp, score in resultados_estadisticos['kappa_scores'].items()
                    ])
                    st.dataframe(df_kappa, use_container_width=True)

                # Análisis de consenso
                st.markdown(f"##### {tr('analisis_consenso_tit')}")
                info_consenso = f"""
                **{tr('diagnostico_consenso')}:** {INFO_ENFERMEDADES[resultados_estadisticos['consensus']][st.session_state.idioma]}
                **{tr('confianza_promedio')}:** {resultados_estadisticos['consensus_confidence']:.2%}
                **{tr('severidad')}:** {INFO_ENFERMEDADES[resultados_estadisticos['consensus']]['severity']}
                """
                st.info(info_consenso)

            with col2:
                st.markdown(f"##### {tr('analisis_confianza')}")

                # Gráfico de confianza
                datos_confianza = pd.DataFrame([
                    {tr('modelo'): modelo, tr('confianza'): conf}
                    for modelo, conf in resultados_estadisticos['confidence_scores'].items()
                ])

                fig_confianza = px.bar(
                    datos_confianza,
                    x=tr('modelo'),
                    y=tr('confianza'),
                    title=tr('niveles_confianza'),
                    color=tr('confianza'),
                    color_continuous_scale='RdYlGn',
                    range_y=[0, 1]
                )
                fig_confianza.add_hline(y=umbral_confianza, line_dash="dash", line_color="red",
                                     annotation_text=f"{tr('umbral')} ({umbral_confianza:.0%})")
                st.plotly_chart(fig_confianza, use_container_width=True)

            # Sección 2: Pruebas estadísticas tradicionales (con datos simulados)
            st.markdown("---")
            st.markdown(f"#### {tr('pruebas_tradicionales')}")

            col3, col4 = st.columns(2)

            with col3:
                st.markdown(f"##### {tr('ttest_pareado')}")
                st.caption(tr("info_ttest"))

                if 't_tests' in resultados_tradicionales:
                    df_test_t = pd.DataFrame([
                        {
                            tr('comparacion'): comp,
                            tr('t_statistic'): f"{resultado['t_statistic']:.4f}",
                            tr('p_value'): f"{resultado['p_value']:.4f}",
                            tr('significativo_ttest'): "✅" if resultado['significant'] else "❌"
                        }
                        for comp, resultado in resultados_tradicionales['t_tests'].items()
                    ])
                    st.dataframe(df_test_t, use_container_width=True)
                    st.caption(tr("interpretacion_ttest"))

            with col4:
                st.markdown(f"##### {tr('ztest_proporciones')}")
                st.caption(tr("info_ztest"))

                if 'z_tests' in resultados_tradicionales:
                    df_test_z = pd.DataFrame([
                        {
                            tr('comparacion'): comp,
                            tr('z_statistic'): f"{resultado['z_statistic']:.4f}",
                            tr('p_value'): f"{resultado['p_value']:.4f}",
                            tr('prop1'): f"{resultado['prop1']:.3f}",
                            tr('prop2'): f"{resultado['prop2']:.3f}"
                        }
                        for comp, resultado in resultados_tradicionales['z_tests'].items()
                    ])
                    st.dataframe(df_test_z, use_container_width=True)
                    st.caption(tr("interpretacion_ztest"))

            # Análisis de Incertidumbre y Visualizaciones
            st.markdown("---")
            st.markdown(f"#### {tr('incertidumbre_visualizaciones')}")

            # Análisis de entropía (incertidumbre)
            datos_entropia = []
            for nombre_modelo, resultado in st.session_state['predicciones'].items():
                probs = resultado['probabilities']
                entropia = -np.sum(probs * np.log2(probs + 1e-10))
                datos_entropia.append({
                    tr('modelo'): nombre_modelo,
                    tr('entropia'): entropia,
                    tr('interpretacion'): tr('baja_incertidumbre') if entropia < 1.5 else tr('alta_incertidumbre')
                })

            df_entropia = pd.DataFrame(datos_entropia)
            st.dataframe(df_entropia, use_container_width=True)
            st.caption(tr("entropia_info"))


            # Botón para generar reporte PDF
            st.markdown("---")
            st.markdown(f"### {tr('generar_reporte')}")

            if st.button(f"🎯 {tr('generar_reporte_analisis')}", type="primary", use_container_width=True):
                with st.spinner(tr("generando_pdf")):
                    # Obtener la imagen si existe
                    bytes_imagen = st.session_state.get('uploaded_image_bytes', None)
                    buffer_imagen = BytesIO(bytes_imagen) if bytes_imagen else None

                    # Generar PDF con todas las pruebas incluidas
                    buffer_pdf = generar_reporte_pdf(
                        st.session_state['predicciones'],
                        buffer_imagen,
                        resultados_estadisticos,
                        resultados_tradicionales
                    )

                    # Crear botón de descarga que aparece después de la generación
                    st.download_button(
                        label=f"📥 {tr('descargar_reporte')}",
                        data=buffer_pdf,
                        file_name=f"reporte_analisis_tomate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success(tr("reporte_generado"))

        else:
            st.info(tr("info_tab4"))

    with tab_eval:
        st.header(tr("evaluacion_real"))
        st.info(tr("info_evaluacion"))
        
        st.markdown("#### " + tr('modo_analisis'))
        
        # Opciones para el usuario
        modo = st.radio(
            label=tr('selecciona_modo'),
            options=[tr('analisis_completo'), tr('analisis_personalizado')],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        num_imagenes = None
        if modo == tr('analisis_personalizado'):
            num_imagenes = st.number_input(
                label=tr('num_imagenes_clase'),
                min_value=1,
                max_value=100, # Suponiendo que hay 100 imágenes por clase
                value=10,
                step=5,
                help=tr('help_num_imagenes')
            )
        else:
            # Para el análisis completo, usamos 100
            num_imagenes = 100
            st.caption(f"Se analizarán {num_imagenes * len(CLASES_ENFERMEDADES)} imágenes en total ({num_imagenes} por cada una de las {len(CLASES_ENFERMEDADES)} clases).")


        if st.button(tr("iniciar_evaluacion"), type="primary", use_container_width=True):
            with st.spinner(f"{tr('procesando')} {tr('evaluacion_real')}"):
                # Se pasa el número de imágenes seleccionado a la función
                resultados_evaluacion = realizar_evaluacion_real(diccionario_modelos, imagenes_por_clase=num_imagenes)
                if resultados_evaluacion:
                    st.session_state.resultados_evaluacion = resultados_evaluacion
                    st.success(tr("evaluacion_completada"))
                else:
                    st.error(tr("error_evaluacion"))

        # El resto de la pestaña para mostrar resultados permanece exactamente igual...
        if 'resultados_evaluacion' in st.session_state:
            st.markdown("---")
            st.header(tr("resultados_evaluacion"))

            resultados = st.session_state.resultados_evaluacion

            # Tabla de resumen (código existente sin cambios)
            st.markdown(f"#### {tr('resumen_rendimiento')}")
            datos_metricas = []
            for nombre_modelo, res in resultados.items():
                if isinstance(res, dict) and 'accuracy' in res:
                    datos_metricas.append({
                        tr('modelo'): nombre_modelo,
                        tr('precision_validacion'): f"{res['accuracy']:.2%}",
                        tr('mcc'): f"{res['mcc']:.4f}"
                    })
            st.dataframe(pd.DataFrame(datos_metricas), use_container_width=True)

            # Prueba de McNemar (código existente sin cambios)
            st.markdown(f"#### {tr('prueba_mcnemar')}")
            st.caption(tr("info_mcnemar"))
            df_mcnemar = pd.DataFrame([
                {tr('comparacion'): comp, tr('estadistico_chi'): f"{res['chi2']:.4f}", tr('p_value'): f"{res['p_value']:.4f}", tr('diferencia_significativa'): "✅ Sí" if res['p_value'] < 0.05 else "❌ No"}
                for comp, res in resultados['mcnemar_tests'].items()
            ])
            st.dataframe(df_mcnemar, use_container_width=True)
            
            # *** NUEVA SECCIÓN: CURVA ROC-AUC ***
            st.markdown(f"#### {tr('curva_roc_auc')}")
            st.caption(tr('info_roc_auc'))
            
            try:
                # Generar y mostrar el gráfico ROC
                buffer_roc = crear_grafico_roc(
                    resultados['true_labels'],
                    resultados['probabilities'],
                    CLASES_ENFERMEDADES
                )
                st.image(buffer_roc)
            except Exception as e:
                st.error(f"Error al generar la curva ROC: {e}")


            # Mapa de Calor de Predicciones (código existente sin cambios)
            st.markdown(f"#### {tr('mapa_calor_predicciones')}")
            st.caption(tr("info_mapa_calor"))

            try:
                df_preds = pd.DataFrame(resultados['predictions'])
                nombres_modelos = list(df_preds.columns)
                etiquetas_clases = [INFO_ENFERMEDADES[c][st.session_state.idioma] for c in CLASES_ENFERMEDADES]
                df_conteo = pd.DataFrame(index=etiquetas_clases)
                for nombre_modelo in nombres_modelos:
                    conteos = df_preds[nombre_modelo].map(lambda x: INFO_ENFERMEDADES[x][st.session_state.idioma]).value_counts()
                    df_conteo[nombre_modelo] = conteos
                df_conteo = df_conteo.reindex(etiquetas_clases).fillna(0).astype(int)

                fig_mapa_calor, ax_mapa_calor = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_conteo, annot=True, fmt='d', cmap='viridis', linewidths=.5, ax=ax_mapa_calor)
                ax_mapa_calor.set_title(tr('distribucion_predicciones'), fontsize=16)
                ax_mapa_calor.set_xlabel(tr('modelo'), fontsize=12)
                ax_mapa_calor.set_ylabel(tr('clase_predicha'), fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig_mapa_calor)
                plt.close(fig_mapa_calor)
            except Exception as e:
                st.error(f"{tr('error_mapa_calor')}: {e}")

            # Matrices de Confusión (código existente sin cambios)
            st.markdown(f"#### {tr('matrices_confusion')}")
            etiquetas_clases_mc = [INFO_ENFERMEDADES[c][st.session_state.idioma] for c in CLASES_ENFERMEDADES]

            for nombre_modelo, res in resultados.items():
                if isinstance(res, dict) and 'confusion_matrix' in res:
                    with st.expander(f"{tr('ver_matriz_confusion')} {nombre_modelo}"):
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                                    xticklabels=etiquetas_clases_mc, yticklabels=etiquetas_clases_mc, ax=ax)
                        ax.set_title(f'{tr("matriz_confusion")} - {nombre_modelo}', fontsize=16)
                        ax.set_xlabel(tr('prediccion'), fontsize=12)
                        ax.set_ylabel(tr('clase_real'), fontsize=12)
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

            # Botón de reporte
            st.markdown("---")
            st.markdown(f"### {tr('generar_reporte_evaluacion')}")
            if st.button(f"📥 {tr('descargar_reporte_evaluacion')}", use_container_width=True):
                with st.spinner(tr("generando_pdf")):
                    pdf_buffer = generar_reporte_pdf_evaluacion(st.session_state.resultados_evaluacion)
                    st.download_button(
                        label=tr("haga_clic_descargar_pdf"),
                        data=pdf_buffer,
                        file_name=f"reporte_evaluacion_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        key="download_eval_pdf"
                    )

    # Pie de página con información adicional
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
        <p>{tr("nota_pie")}</p>
        <p>{tr("desarrollado")} | {datetime.now().strftime("%Y")}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    principal()