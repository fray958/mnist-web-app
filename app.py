from flask import Flask, render_template, request, url_for
from keras.layers import TFSMLayer
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os
import time
from scipy.ndimage import center_of_mass
import cv2  # Usaremos OpenCV para aislar el número del recuadro rosa

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

# Memoria temporal para el historial
historial_predicciones = []

def procesar_imagen(ruta):
    # 1. Abrir imagen y convertir a escala de grises
    img = Image.open(ruta).convert('L')
    img_array = np.array(img)
    
    # 2. DETECCIÓN DE INVERSIÓN POR ESQUINAS
    esquinas = [img_array[0,0], img_array[0,-1], img_array[-1,0], img_array[-1,-1]]
    color_fondo_estimado = np.median(esquinas)
    val_min, val_max = img_array.min(), img_array.max()
    
    if abs(color_fondo_estimado - val_max) < abs(color_fondo_estimado - val_min):
        img_array = 255 - img_array  # Inversión manual estable
    else:
        if color_fondo_estimado > 0:
            img_array = np.clip(img_array - color_fondo_estimado, 0, 255)

    # 3. Umbralización para binarizar (Blanco y Negro puro)
    _, img_binaria = cv2.threshold(img_array, 100, 255, cv2.THRESH_BINARY)

    # 4. EXTRACCIÓN DEL DÍGITO CENTRAL (Elimina marcos y fondos rosa/blancos)
    contornos, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_digito = np.zeros_like(img_binaria)
    if contornos:
        # Buscamos el contorno más grande o el que esté más centrado
        h, w = img_binaria.shape
        centro_img = (w // 2, h // 2)
        
        mejor_contorno = None
        min_distancia = float('inf')
        
        for c in contornos:
            M = cv2.moments(c)
            if M["m00"] > 10:  # Ignorar ruidos diminutos
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                distancia = ((cx - centro_img[0])**2 + (cy - centro_img[1])**2)**0.5
                if distancia < min_distancia:
                    min_distancia = distancia
                    mejor_contorno = c
        
        if mejor_contorno is not None:
            # Dibujamos SOLO el número ignorando bordes decorativos
            cv2.drawContours(img_digito, [mejor_contorno], -1, 255, -1)
            # Aplicamos máscara para conservar texturas originales del trazo
            img_final_raw = cv2.bitwise_and(img_array, img_array, mask=img_digito)
        else:
            img_final_raw = img_binaria
    else:
        img_final_raw = img_binaria

    # 5. Ajustar caja de contorno (Bounding Box)
    img_limpia = Image.fromarray(img_final_raw)
    caja = img_limpia.getbbox()
    if caja:
        img_limpia = img_limpia.crop(caja)
            
    # 6. Redimensionar manteniendo proporciones
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
        
    # 7. Centrar en el lienzo de 28x28
    inicio_x = (28 - nuevo_ancho) // 2
    inicio_y = (28 - nuevo_alto) // 2
    img_final_array[inicio_y:inicio_y+nuevo_alto, inicio_x:inicio_x+nuevo_ancho] = np.array(img_20x20)
        
    # 8. Alineación por Centro de Masa
    cy, cx = center_of_mass(img_final_array)
    if not np.isnan(cx) and not np.isnan(cy):
        shift_x = int(round(14.0 - cx))
        shift_y = int(round(14.0 - cy))
        img_final_array = np.roll(img_final_array, shift_x, axis=1)
        img_final_array = np.roll(img_final_array, shift_y, axis=0)
        
    # 9. Suavizado y Normalización para la CNN
    img_pil_final = Image.fromarray(img_final_array.astype(np.uint8))
    img_pil_final = img_pil_final.filter(ImageFilter.SMOOTH_MORE)
        
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
            
            # Procesar y predecir
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