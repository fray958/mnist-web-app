# Usar una versión oficial de Python estable
FROM python:3.10-slim

# Establecer la carpeta de trabajo dentro del servidor
WORKDIR /app

# Copiar los archivos de requisitos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el resto del proyecto
COPY . .

# Exponer el puerto estándar que usa Hugging Face
EXPOSE 7860

# Comando para arrancar tu servidor Flask en el puerto correcto
CMD ["python", "app.py"]