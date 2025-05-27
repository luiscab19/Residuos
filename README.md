# 🗑️ Clasificador de Residuos con Redes Neuronales Convolucionales (CNN)

## 🧠 Descripción
Este proyecto implementa un **sistema de clasificación automática de residuos** utilizando una red neuronal convolucional (CNN) con TensorFlow/Keras. La inteligencia artificial puede reconocer y clasificar 6 tipos de materiales:

- Cartón (cardboard)
- Vidrio (glass)
- Metal (metal)
- Papel (paper)
- Plástico (plastic)
- Basura general (trash)

## 🏗️ Estructura del Proyecto
```
waste-classifier/
│
├── data/                   # Imágenes para entrenamiento (organizadas por clase)
│   ├── cardboard/          # Ej: cajas, envases de cartón
│   ├── glass/              # Ej: botellas, frascos
│   ├── metal/              # Ej: latas, tapas
│   ├── paper/              # Ej: periódicos, revistas
│   ├── plastic/            # Ej: botellas, envases
│   └── trash/              # Residuos no reciclables
│
├── prueba/                 # Imágenes para probar el modelo
│   ├── residuo1.jpg
│   ├── residuo2.jpg
│   └── ...
│
├── main.py                 # Programa principal
├── waste_classifier.keras  # Modelo entrenado (se genera automáticamente)
├── confusion_matrix.png    # Resultados del modelo
└── training_history.png    # Evolución del aprendizaje
```

## ⚙️ Requisitos Técnicos
- Python 3.8+
- Bibliotecas esenciales:
  ```bash
  pip install tensorflow numpy pillow opencv-python scikit-learn matplotlib seaborn
  ```

## 🚀 Cómo Usar el Clasificador

### 1. Preparación de Datos
Organiza tus imágenes de entrenamiento en la carpeta `./data/` con subcarpetas para cada categoría.

### 2. Ejecución del Programa
```bash
python main.py
```

### 3. Menú Interactivo
```
--- SISTEMA DE CLASIFICACIÓN INTELIGENTE ---
1. Entrenar el modelo con imágenes de ./data
2. Clasificar imágenes nuevas de ./prueba
3. Salir
```

### 🔧 Entrenamiento del Modelo (Opción 1)
- Procesa automáticamente todas las imágenes
- Crea un modelo de inteligencia artificial
- Genera reportes visuales del rendimiento
- Guarda el modelo entrenado para uso futuro

### 🔎 Clasificación (Opción 2)
- Abre una interfaz gráfica amigable
- Selecciona cualquier imagen para analizar
- Muestra resultados con porcentaje de confianza

## 🧠 Arquitectura de la Red Neuronal
El modelo utiliza una **CNN profunda** con:

| Capa | Tipo | Detalles |
|------|------|----------|
| 1 | Convolucional | 32 filtros (3x3) + ReLU |
| 2 | Max Pooling | Reducción 2x2 |
| 3 | Convolucional | 64 filtros (3x3) + ReLU |
| 4 | Max Pooling | Reducción 2x2 |
| 5 | Convolucional | 128 filtros (3x3) + ReLU |
| 6 | Max Pooling | Reducción 2x2 |
| 7 | Flatten | Aplanamiento |
| 8 | Densa | 128 neuronas + ReLU |
| 9 | Dropout | Regularización (50%) |
| 10 | Densa | 6 neuronas (salida) + Softmax |

## 📊 Evaluación del Modelo
El sistema genera automáticamente:
- 📈 Gráficos de precisión y pérdida
- 🎯 Matriz de confusión detallada
- 📝 Reporte de métricas por clase:
  - Precisión (accuracy)
  - Sensibilidad (recall)
  - Puntuación F1

## 💡 Recomendaciones
- Usa imágenes variadas (diferentes ángulos, iluminación)
- Mínimo 100-200 imágenes por categoría para buen rendimiento
- Formatos soportados: JPG, PNG, etc.
- Tamaño ideal: mínimo 100x100 píxeles

## 🛠️ Tecnologías Clave
- **TensorFlow/Keras** para la red neuronal
- **OpenCV** para procesamiento de imágenes
- **Scikit-learn** para evaluación
- **Matplotlib/Seaborn** para visualización

📌 *Desarrollado por Luis Alvarez para aplicaciones de Inteligencia Artificial*
