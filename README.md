# README - Sistema de Clasificación de Residuos con Redes Neuronales

## 📌 Descripción

Este proyecto implementa un clasificador de imágenes de residuos (cartón, vidrio, metal, papel, plástico y basura general) utilizando una red neuronal convolucional (CNN) con TensorFlow/Keras. El sistema puede entrenarse con un conjunto de imágenes etiquetadas y luego usarse para clasificar nuevas imágenes de residuos.

## 🗂️ Estructura del Proyecto

```
waste-classifier/
│
├── data/                   # Directorio con imágenes de entrenamiento (subcarpetas por clase)
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
│
├── prueba/                 # Directorio con imágenes para clasificar
│   ├── residuo1.jpg
│   ├── residuo2.jpg
│   └── 
│
├── main.py     # Código principal del clasificador
├── waste_classifier.keras  # Modelo entrenado (se genera al ejecutar)
├── confusion_matrix.png    # Matriz de confusión (se genera al entrenar)
└── training_history.png    # Gráficos de entrenamiento (se generan al entrenar)
```

## 📋 Requisitos

- Python 3.8+
- Bibliotecas requeridas (instalar con `pip install -r requirements.txt`):
  ```
  tensorflow>=2.0
  numpy
  pillow
  opencv-python
  scikit-learn
  matplotlib
  seaborn
  ```

## 🚀 Cómo Usar

1. **Preparar los datos**:
   - Organiza tus imágenes de entrenamiento en subcarpetas dentro de `./data/`, una carpeta por cada clase (cardboard, glass, metal, paper, plastic, trash)

2. **Ejecutar el programa**:
   ```
   python waste_classifier.py
   ```

3. **Menú de opciones**:
   ```
   --- Sistema de Clasificación de Residuos ---
   1. Entrenar modelo con imágenes en ./data
   2. Clasificar imágenes en ./prueba
   3. Salir
   ```

4. **Entrenar el modelo (Opción 1)**:
   - El programa cargará las imágenes, entrenará el modelo y guardará:
     - Modelo entrenado (`waste_classifier.keras`)
     - Matriz de confusión (`confusion_matrix.png`)
     - Gráficos de entrenamiento (`training_history.png`)

5. **Clasificar imágenes (Opción 2)**:
   - Coloca las imágenes a clasificar en `./prueba/`
   - El modelo mostrará cada imagen con su predicción y nivel de confianza

## 🧠 Arquitectura del Modelo

El clasificador utiliza una CNN con la siguiente estructura:

1. Capa Conv2D (32 filtros, kernel 3x3, ReLU)
2. MaxPooling2D (2x2)
3. Capa Conv2D (64 filtros, kernel 3x3, ReLU)
4. MaxPooling2D (2x2)
5. Capa Conv2D (128 filtros, kernel 3x3, ReLU)
6. MaxPooling2D (2x2)
7. Flatten
8. Dense (128 neuronas, ReLU)
9. Dropout (0.5 para regularización)
10. Dense (6 neuronas, softmax - una por clase)

## 📊 Métricas de Evaluación

Al finalizar el entrenamiento, el programa muestra:
- Precisión en el conjunto de prueba
- Reporte de clasificación (precision, recall, f1-score por clase)
- Matriz de confusión visual
- Gráficos de precisión y pérdida durante el entrenamiento

## 📝 Notas

- Las imágenes deben ser en color (RGB) y se redimensionarán a 100x100 píxeles
- El modelo normaliza los valores de píxeles al rango [0, 1]
- Para mejores resultados, se recomienda tener al menos 100-200 imágenes por clase

## 📄 El proyecto fue realizado por Luis Alvarez para la asignatura de Inteligencia Artificial 
