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

# Configuración de carpetas para guardar las imágenes subidas
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Memoria temporal para el historial de la interfaz web
historial_predicciones = []

def procesar_imagen(ruta):
    # 1. Abrir la imagen y convertirla a escala de grises
    img = Image.open(ruta).convert('L')
    img_array = np.array(img)
    
    # 2. DETECCIÓN INTELIGENTE AVANZADA (Muestreo de esquinas)
    # Extraemos el color de las 4 esquinas de la imagen para identificar el fondo
    esquinas = [img_array[0,0], img_array[0,-1], img_array[-1,0], img_array[-1,-1]]
    color_fondo_estimado = np.median(esquinas)
    
    # Buscamos los valores máximos y mínimos absolutos de la imagen
    val_min, val_max = img_array.min(), img_array.max()
    
    # Si el fondo estimado está más cerca del valor máximo (fondo claro), invertimos
    if abs(color_fondo_estimado - val_max) < abs(color_fondo_estimado - val_min):
        img = ImageOps.invert(img)
        img_array = np.array(img)
    else:
        # Si el fondo ya es oscuro (como el rosa o azul), limpiamos el ruido del fondo
        # restando su valor para convertirlo en negro absoluto (0) y resaltar el dígito blanco
        if color_fondo_estimado > 0:
            img_array = np.clip(img_array - color_fondo_estimado, 0, 255)

    # 3. Umbralización dinámica adaptativa
    # Si la imagen tiene poco contraste bajamos el umbral para no borrar los trazos del número
    rango = img_array.max() - img_array.min()
    umbral = 50 if rango < 150 else 100
    img_array = np.where(img_array > umbral, img_array, 0)
        
    # 4. Ajustar caja de contorno (Bounding Box) para eliminar bordes vacíos
    img_limpia = Image.fromarray(img_array.astype(np.uint8))
    caja = img_limpia.getbbox()
    if caja:
        img_limpia = img_limpia.crop(caja)
            
    # 5. Redimensionar el dígito manteniendo sus proporciones originales
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
    img_final_array = np.zeros((28, 28))
        
    # 6. Centrar el número en el lienzo estándar de 28x28 píxeles
    inicio_x = (28 - nuevo_ancho) // 2
    inicio_y = (28 - nuevo_alto) // 2
    img_final_array[inicio_y:inicio_y+nuevo_alto, inicio_x:inicio_x+nuevo_ancho] = np.array(img_20x20)
        
    # 7. Alineación perfecta por Centro de Masa (idéntico a la base de datos MNIST)
    cy, cx = center_of_mass(img_final_array)
    if not np.isnan(cx) and not np.isnan(cy):
        shift_x = int(round(14.0 - cx))
        shift_y = int(round(14.0 - cy))
        img_final_array = np.roll(img_final_array, shift_x, axis=1)
        img_final_array = np.roll(img_final_array, shift_y, axis=0)
        
    # 8. Suavizado final de trazo para eliminar pixelado rudo
    img_pil_final = Image.fromarray(img_final_array.astype(np.uint8))
    img_pil_final = img_pil_final.filter(ImageFilter.SMOOTH_MORE)
        
    # 9. Normalización matemática para la entrada de la red CNN ([0, 1])
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
            
            # Ejecutar el procesamiento de la imagen
            img = procesar_imagen(ruta)
            
            # Realizar la predicción con el modelo cargado
            prediction_output = model(img)
            if isinstance(prediction_output, dict):
                first_key = list(prediction_output.keys())[0]
                prediction = prediction_output[first_key].numpy()
            else:
                prediction = prediction_output.numpy()
                
            resultado = int(np.argmax(prediction))
            imagen = ruta.replace("\\", "/")
            
            # Guardar el registro en el historial dinámico
            hora_actual = time.strftime("%H:%M:%S")
            historial_predicciones.insert(0, {
                "imagen_url": imagen,
                "resultado": resultado,
                "hora": hora_actual
            })
                        
            # Mantener únicamente las últimas 5 predicciones visibles
            if len(historial_predicciones) > 5:
                historial_predicciones.pop()
                
    return render_template(
        "index.html",
        resultado=resultado,
        imagen=imagen,
        historial=historial_predicciones
    )

if __name__ == "__main__":
    # Hugging Face Spaces exige leer la variable de entorno PORT (por defecto 7860)
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)