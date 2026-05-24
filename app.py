from flask import Flask, render_template, request
from keras.layers import TFSMLayer
from PIL import Image, ImageFilter
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

def calcular_otsu_nativo(imagen_gris):
    """Calcula el umbral óptimo de Otsu automáticamente"""
    hist, bin_edges = np.histogram(imagen_gris, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    
    variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
    idx = np.argmax(variance_between)
    return bin_centers[idx]

def procesar_imagen(ruta):
    # 1. Abrir imagen y pasar a escala de grises
    img = Image.open(ruta).convert('L')
    
    # 🔥 Filtro de velocidad para capturas de pantalla gigantes
    img.thumbnail((500, 500), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape
    
    # 2. binarización inteligente automática (Detecta si el fondo es claro u oscuro)
    thresh = calcular_otsu_nativo(img_array)
    
    # Determinar si el fondo predominante es claro u oscuro analizando las esquinas
    esquinas = [img_array[0,0], img_array[0,-1], img_array[-1,0], img_array[-1,-1]]
    fondo_claro = np.mean(esquinas) > thresh
    
    if fondo_claro:
        # Si el fondo es claro, el número es lo oscuro
        mascara_binaria = (img_array < thresh).astype(np.float32) * 255.0
    else:
        # Si el fondo es oscuro, el número es lo claro (como tus nuevas imágenes)
        mascara_binaria = (img_array > thresh).astype(np.float32) * 255.0

    # 3. AISLAR EL OBJETO CENTRAL (El dígito)
    estructuras, num_features = label(mascara_binaria > 50)
    img_solo_numero = np.zeros_like(mascara_binaria)
    
    if num_features > 0:
        centro_y, centro_x = h // 2, w // 2
        mejor_id = 1
        dist_min = float('inf')
        
        # Buscar el componente más cercano al centro físico de la imagen
        for i in range(1, num_features + 1):
            componente = (estructuras == i)
            cy, cx = center_of_mass(componente)
            if not np.isnan(cx) and not np.isnan(cy):
                dist = (cy - centro_y)**2 + (cx - centro_x)**2
                if dist < dist_min:
                    dist_min = dist
                    mejor_id = i
                    
        img_solo_numero = np.where(estructuras == mejor_id, 255.0, 0.0)
    else:
        img_solo_numero = mascara_binaria

    # 4. ENCUADRE Estándar MNIST (Centrado perfecto en caja de 28x28)
    img_limpia = Image.fromarray(img_solo_numero.astype(np.uint8))
    caja = img_limpia.getbbox()
    if caja:
        img_limpia = img_limpia.crop(caja)
            
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
        
    inicio_x = (28 - nuevo_ancho) // 2
    inicio_y = (28 - nuevo_alto) // 2
    img_final_array[inicio_y:inicio_y+nuevo_alto, inicio_x:inicio_x+nuevo_ancho] = np.array(img_20x20)
        
    # Ajuste fino por Centro de Masa
    cy, cx = center_of_mass(img_final_array)
    if not np.isnan(cx) and not np.isnan(cy):
        shift_x = int(round(14.0 - cx))
        shift_y = int(round(14.0 - cy))
        img_final_array = np.roll(img_final_array, shift_x, axis=1)
        img_final_array = np.roll(img_final_array, shift_y, axis=0)
        
    # Filtro anti-aliasing idéntico al entrenamiento de MNIST
    img_pil_final = Image.fromarray(img_final_array.astype(np.uint8))
    img_pil_final = img_pil_final.filter(ImageFilter.SMOOTH_MORE)
        
    # Normalizar a rango [0.0, 1.0] para la CNN
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