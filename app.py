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
    
    # 2. LOCALIZAR LA REGIÓN DE INTERÉS (Detectar el recuadro central útil)
    # Creamos una máscara para encontrar cualquier zona que no sea el fondo oscuro del lienzo gigante
    mascara_bloque = (img_array > 30).astype(np.uint8)
    estructuras, num_features = label(mascara_bloque)
    
    # Por defecto, usamos toda la imagen
    ymin, xmin, ymax, xmax = 0, 0, h, w
    
    if num_features > 0:
        centro_y, centro_x = h // 2, w // 2
        mejor_id = 1
        dist_min = float('inf')
        
        # Encontrar el bloque/recuadro más cercano al centro de la captura
        for i in range(1, num_features + 1):
            componente = (estructuras == i)
            cy, cx = center_of_mass(componente)
            if not np.isnan(cx) and not np.isnan(cy):
                dist = (cy - centro_y)**2 + (cx - centro_x)**2
                if dist < dist_min:
                    dist_min = dist
                    mejor_id = i
        
        # Obtener las coordenadas del recuadro del número
        filas, columnas = np.where(estructuras == mejor_id)
        if len(filas) > 0 and len(columnas) > 0:
            ymin, xmin = np.min(filas), np.min(columnas)
            ymax, xmax = np.max(filas), np.max(columnas)
    
    # Recortamos la imagen para quedarnos SOLO con el cuadro del número (adiós lienzo negro gigante)
    recorte = img_array[ymin:ymax, xmin:xmax]
    rh, rw = recorte.shape
    
    # 3. SEGMENTACIÓN ADAPTATIVA DENTRO DEL RECORTE
    # Evaluamos el color de las esquinas del recorte para saber si el fondo interno es claro u oscuro
    esquinas = [recorte[0, 0], recorte[0, rw-1], recorte[rh-1, 0], recorte[rh-1, rw-1]]
    fondo_claro = np.mean(esquinas) > 120
    
    if fondo_claro:
        # Caso para el 2 y el 3 (Trazo oscuro sobre fondo blanco limpio)
        # Evaluamos si el número es rojizo/oscuro. Invertimos para que el trazo sea blanco brillante.
        mascara_num = (recorte < 110).astype(np.float32) * 255.0
    else:
        # Caso para el 8 rosa (El fondo del recorte es el rosa, que en grises es intermedio/oscuro)
        # Buscamos el trazo blanco brillante del centro
        mascara_num = np.where(recorte > 200, 255.0, 0.0)
        
        # Limpieza rápida de pequeños residuos del borde del recuadro rosa
        b_h = max(1, int(rh * 0.05))
        b_w = max(1, int(rw * 0.05))
        mascara_num[:b_h, :] = 0
        mascara_num[-b_h:, :] = 0
        mascara_num[:, :b_w] = 0
        mascara_num[:, -b_w:] = 0

    # 4. AISLAR EL DÍGITO DEFINITIVO
    estructuras_num, num_feats_num = label(mascara_num > 0)
    img_solo_numero = np.zeros_like(mascara_num)
    
    if num_feats_num > 0:
        c_y, c_x = rh // 2, rw // 2
        id_num = 1
        d_min = float('inf')
        
        for i in range(1, num_feats_num + 1):
            comp = (estructuras_num == i)
            cy, cx = center_of_mass(comp)
            if not np.isnan(cx) and not np.isnan(cy):
                dist = (cy - c_y)**2 + (cx - c_x)**2
                if dist < d_min:
                    d_min = dist
                    id_num = i
                    
        img_solo_numero = np.where(estructuras_num == id_num, 255.0, 0.0)
    else:
        img_solo_numero = mascara_num

    # 5. AJUSTAR CUADRANTE Y PROCESAMIENTO Estándar MNIST (28x28)
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
        
    # Centrar usando Centro de Masa
    cy, cx = center_of_mass(img_final_array)
    if not np.isnan(cx) and not np.isnan(cy):
        shift_x = int(round(14.0 - cx))
        shift_y = int(round(14.0 - cy))
        img_final_array = np.roll(img_final_array, shift_x, axis=1)
        img_final_array = np.roll(img_final_array, shift_y, axis=0)
        
    # Suavizado anti-aliasing idéntico al dataset MNIST
    img_pil_final = Image.fromarray(img_final_array.astype(np.uint8))
    img_pil_final = img_pil_final.filter(ImageFilter.SMOOTH_MORE)
        
    # Normalización para la entrada de la CNN (0.0 a 1.0)
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