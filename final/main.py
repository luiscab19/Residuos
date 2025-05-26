import os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class WasteClassifier:
    def __init__(self):
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.image_size = (100, 100)
        self.model = None
        self.history = None
        
    def load_data(self, data_dir='./data'):
        images = []
        labels = []
        
        for class_id, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize(self.image_size)
                    image = np.array(image) / 255.0
                    images.append(image)
                    labels.append(class_id)
                except Exception as e:
                    print(f"Error al procesar {image_path}: {e}")
        
        if not images:
            raise ValueError("No se encontraron imágenes en el directorio de datos.")
            
        images = np.array(images)
        labels = np.array(labels)
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(self.class_names))
        
        return train_test_split(images, labels, test_size=0.2, random_state=42)
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=20):
        self.model = self.build_model()
        
        print("\nEntrenando el modelo...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            batch_size=32,
            verbose=1
        )
        
        self.model.save('waste_classifier.keras')
        print("\nModelo entrenado y guardado como 'waste_classifier.keras'")
        
        # Evaluación
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f'\nPrecisión en el conjunto de prueba: {test_acc:.4f}')
        
        # Reporte de clasificación
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print("\nReporte de Clasificación:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=self.class_names))
        
        # Matriz de confusión
        self.plot_confusion_matrix(y_true_classes, y_pred_classes)
        
        # Gráfico de precisión y pérdida
        self.plot_training_history()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.savefig('confusion_matrix.png')
        plt.show()
    
    def plot_training_history(self):
        if self.history is None:
            return
            
        plt.figure(figsize=(12, 5))
        
        # Gráfico de precisión
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Precisión Entrenamiento')
        plt.plot(self.history.history['val_accuracy'], label='Precisión Validación')
        plt.title('Precisión durante el Entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Pérdida Entrenamiento')
        plt.plot(self.history.history['val_loss'], label='Pérdida Validación')
        plt.title('Pérdida durante el Entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        plt.savefig('training_history.png')
        plt.show()
    
    def classify_images_in_folder(self, folder_path='./prueba'):
        if not os.path.exists(folder_path):
            print(f"\nError: No se encontró la carpeta '{folder_path}'")
            return
        
        if self.model is None:
            try:
                self.model = tf.keras.models.load_model('waste_classifier.keras')
                print("\nModelo cargado desde 'waste_classifier.keras'")
            except:
                print("\nError: Modelo no encontrado. Entrena el modelo primero (Opción 1).")
                return
        
        print(f"\nClasificando imágenes en '{folder_path}':")
        
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            
            try:
                # Preprocesamiento de la imagen
                image = Image.open(image_path).convert('RGB')
                image = image.resize(self.image_size)
                image_array = np.array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)
                
                # Predicción
                prediction = self.model.predict(image_array)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                class_name = self.class_names[class_id]
                
                # Mostrar resultados en consola
                print(f"\nImagen: {image_name}")
                print(f"Clase predicha: {class_name}")
                print(f"Confianza: {confidence:.2%}")
                
                # Mostrar imagen con la predicción
                img_display = cv2.imread(image_path)
                if img_display is not None:
                    text = f"{class_name} ({confidence:.2%})"
                    cv2.putText(img_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Clasificación de Residuo', img_display)
                    cv2.waitKey(2000)  # Mostrar cada imagen por 2 segundos
                    cv2.destroyAllWindows()
                else:
                    print(f"No se pudo mostrar la imagen {image_name}")
                    
            except Exception as e:
                print(f"\nError al procesar {image_name}: {str(e)}")

def main():
    classifier = WasteClassifier()
    
    while True:
        print("\n--- Sistema de Clasificación de Residuos ---")
        print("1. Entrenar modelo con imágenes en ./data")
        print("2. Clasificar imágenes en ./prueba")
        print("3. Salir")
        
        choice = input("Selecciona una opción: ")
        
        if choice == '1':
            try:
                print("\nCargando datos de entrenamiento...")
                X_train, X_test, y_train, y_test = classifier.load_data()
                classifier.train(X_train, y_train, X_test, y_test)
            except Exception as e:
                print(f"\nError durante el entrenamiento: {e}")
                
        elif choice == '2':
            classifier.classify_images_in_folder()
            
        elif choice == '3':
            print("\nSaliendo del programa...")
            break
            
        else:
            print("\nOpción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()