from flask import Flask, render_template, request, url_for
from keras.layers import TFSMLayer
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os
import time
from scipy.ndimage import center_of_mass, label

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
    h, w = img_array.shape
    
    # 2. Análisis del centro geométrico
    centro_y, centro_x = h // 2, w // 2
    
    # Muestrear región central para evaluar la presencia del trazo
    region_central = img_array[max(0, centro_y-15):min(h, centro_y+15), max(0, centro_x-15):min(w, centro_x+15)]
    brillo_centro = np.mean(region_central)
    
    # 3. SEGMENTACIÓN DE DOBLE VÍA (Sirve para fondo blanco tradicional y para el 8 rosa en fondo oscuro)
    if brillo_centro > 200:
        # Caso 1: El centro es sumamente brillante (el trazo del 8 rosa)
        # Extraemos lo puramente blanco e ignoramos el recuadro rosa de fondo
        mascara = (img_array > 220).astype(np.uint8)
    else:
        # Caso 2: El centro no es totalmente blanco (Fondo claro con número negro o similar)
        # Tomamos esquinas para validar si el fondo exterior es claro u oscuro
        esquinas = [img_array[0, 0], img_array[0, w-1], img_array[h-1, 0], img_array[h-1, w-1]]
        if np.mean(esquinas) > 127:
            mascara = (img_array < 100).astype(np.uint8)
        else:
            # Respaldo para trazos claros sobre fondos complejos
            mascara = (img_array > 180).astype(np.uint8)

    # 4. FILTRADO POR CONECTIVIDAD (Aislar la isla del número)
    estructuras_conectadas, num_features = label(mascara)
    img_solo_numero = np.zeros_like(img_array)
    
    if num_features > 0:
        mejor_id = 1
        distancia_minima = float('inf')
        
        for i in range(1, num_features + 1):
            componente_mascara = (estructuras_conectadas == i)
            # Descartar si el componente toca los bordes absolutos de la imagen (marcos externos)
            if componente_mascara[0, :].any() or componente_mascara[-1, :].any() or componente_mascara[:, 0].any() or componente_mascara[:, -1].any():
                continue
                
            cy, cx = center_of_mass(componente_mascara)
            if not np.isnan(cx) and not np.isnan(cy):
                dist = (cy - centro_y)**2 + (cx - centro_x)**2
                if dist < distancia_minima:
                    distancia_minima = dist
                    mejor_id = i
                    
        img_solo_numero = np.where(estructuras_conectadas == mejor_id, 255.0, 0.0)
    
    # Si la limpieza extrema borró todo, usamos umbral adaptativo directo del centro
    if np.sum(img_solo_numero) == 0:
        img_solo_numero = np.where((img_array > 215) & (img_array < 255), 255.0, 0.0)

    # 5. Ajustar caja de contorno (Bounding Box)
    img_limpia = Image.fromarray(img_solo_numero.astype(np.uint8))
    caja = img_limpia.getbbox()
    if caja:
        img_limpia = img_limpia.crop(caja)
            
    # 6. Redimensionar manteniendo proporciones (Estándar MNIST de 20x20)
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
        
    # 7. Centrar en el lienzo de 28x28
    inicio_x = (28 - nuevo_ancho) // 2
    inicio_y = (28 - nuevo_alto) // 2
    img_final_array[inicio_y:inicio_y+nuevo_alto, inicio_x:inicio_x+nuevo_ancho] = np.array(img_20x20)
        
    # 8. Corrección fina usando el Centro de Masa
    cy, cx = center_of_mass(img_final_array)
    if not np.isnan(cx) and not np.isnan(cy):
        shift_x = int(round(14.0 - cx))
        shift_y = int(round(14.0 - cy))
        img_final_array = np.roll(img_final_array, shift_x, axis=1)
        img_final_array = np.roll(img_final_array, shift_y, axis=0)
        
    # 9. Suavizado anti-aliasing para la red neuronal
    img_pil_final = Image.fromarray(img_final_array.astype(np.uint8))
    img_pil_final = img_pil_final.filter(ImageFilter.SMOOTH_MORE)
        
    # 10. Normalización final (0.0 - 1.0)
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
            
            img = procesar_imagen(ruta)
            prediction_output = model(img)
            if isinstance(prediction_output, dict):
                first_key = list(prediction_output.keys())[0]
                prediction = prediction_output[first_key].numpy()
            else:
                prediction = prediction_output.numpy()
                
            resultado = int(np.argmax(prediction))
            imagen = ruta.replace("\\", "/")
            
            hora_actual = time.strftime("%H:%M:%S")
            historial_predicciones.insert(0, {
                "imagen_url": imagen,
                "resultado": resultado,
                "hora": clock_time
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