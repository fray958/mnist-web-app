from flask import Flask, render_template, request, url_for
from keras.layers import TFSMLayer
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os
import time
from scipy.ndimage import center_of_mass

app = Flask(__name__)

# Cargar tu modelo SavedModel
model = TFSMLayer(
    "modelo_web",
    call_endpoint='serve'
)

# Configuración de carpetas
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Historial temporal
historial_predicciones = []

def procesar_imagen(ruta):
    # 1. Abrir imagen y pasar a escala de grises
    img = Image.open(ruta).convert('L')
    img_array = np.array(img, dtype=np.float32)
    
    # 2. SEGMENTACIÓN AVANZADA POR HISTOGRAMA (Detecta cualquier color de fondo automáticamente)
    # Encontramos el valor de gris que más predomina (el fondo de la imagen)
    valores, conteos = np.unique(img_array, return_counts=True)
    color_fondo = valores[np.argmax(conteos)]
    
    # Si el fondo dominante es claro (Blanco/Gris claro de tus pruebas anteriores)
    if color_fondo > 127:
        # Invertimos para que el fondo pase a ser negro y el número blanco
        img_array = 255.0 - img_array
    else:
        # Si el fondo dominante es oscuro o de color (como el rosa, que en grises es oscuro)
        # Restamos el fondo y dejamos el trazo blanco brillante intacto
        img_array = img_array - color_fondo
        img_array = np.clip(img_array, 0, 255)

    # 3. FILTRO DE CONFIRMACIÓN (Elimina ruidos residuales de bordes blancos/decorativos)
    h, w = img_array.shape
    borde_h = int(h * 0.06)
    borde_w = int(w * 0.06)
    img_array[:borde_h, :] = 0
    img_array[-borde_h:, :] = 0
    img_array[:, :borde_w] = 0
    img_array[:, -borde_w:] = 0

    # Binarización estricta: Limpia cualquier rastro del recuadro rosa
    img_array = np.where(img_array > 90, img_array, 0)
    
    # 4. Ajustar caja de contorno (Bounding Box) del número aislado
    img_limpia = Image.fromarray(img_array.astype(np.uint8))
    caja = img_limpia.getbbox()
    if caja:
        img_limpia = img_limpia.crop(caja)
            
    # 5. Redimensionar manteniendo proporciones
    ancho, alto = img_limpia.size
    if ancho > alto:
        nuevo_ancho = 20
        nuevo_alto = int(round((20.0 * alto / ancho)))
    else:
        nuevo_alto = 20
        nuevo_ancho = int(round((20.0 * ancho / alto)))
            
    nuevo_ancho = max(1, nuevo_ancho)
    nuevo_alto = max(1, nuevo_alto)
        
    img_20x20 = img_limpia.resize((nuevo_ancho, nuevo_alto), resample=Image.LANCZOS)
    img_final_array = np.zeros((28, 28), dtype=np.float32)
        
    # 6. Centrar en el lienzo estándar de 28x28
    inicio_x = (28 - nuevo_ancho) // 2
    inicio_y = (28 - nuevo_alto) // 2
    img_final_array[inicio_y:inicio_y+nuevo_alto, inicio_x:inicio_x+nuevo_ancho] = np.array(img_20x20)
        
    # 7. Alineación perfecta usando el Centro de Masa
    cy, cx = center_of_mass(img_final_array)
    if not np.isnan(cx) and not np.isnan(cy):
        shift_x = int(round(14.0 - cx))
        shift_y = int(round(14.0 - cy))
        img_final_array = np.roll(img_final_array, shift_x, axis=1)
        img_final_array = np.roll(img_final_array, shift_y, axis=0)
        
    # 8. Suavizado idéntico al dataset MNIST
    img_pil_final = Image.fromarray(img_final_array.astype(np.uint8))
    img_pil_final = img_pil_final.filter(ImageFilter.SMOOTH_MORE)
        
    # 9. Normalización final para la entrada de la CNN
    img_cnn = np.array(img_pil_final) / 255.0
    img_cnn = img_cnn.reshape(1, 28, 28, 1).astype(np.float32)
        
    return img_cnn

@app.route("/", methods=["GET", "POST"])
def index():
    global historial_predicciones
    resultado = None
    imagen = None
    if request.method == "POST":
        archivo = request.files.get("imagen")
        if archivo and archivo.filename != '':
            nombre_unico = f"{int(time.time())}_{archivo.filename}"
            ruta = os.path.join(app.config['UPLOAD_FOLDER'], nombre_unico)
            archivo.save(ruta)
            
            # Ejecutar el pipeline
            img = procesar_imagen(ruta)
            prediction_output = model(img)
            if isinstance(prediction_output, dict):
                first_key = list(prediction_output.keys())[0]
                prediction = prediction_output[first_key].numpy()
            else:
                prediction = prediction_output.numpy()
                
            resultado = int(np.argmax(prediction))
            imagen = ruta.replace("\\", "/")
            
            # Guardar en el historial
            hora_actual = time.strftime("%H:%M:%S")
            historial_predicciones.insert(0, {
                "imagen_url": imagen,
                "resultado": resultado,
                "hora": hora_actual
            })
                        
            if len(historial_predicciones) > 5:
                historial_predicciones.pop()
                
    return render_template(
        "index.html",
        resultado=resultado,
        imagen=imagen,
        historial=historial_predicciones
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)