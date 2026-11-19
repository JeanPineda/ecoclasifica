# EcoClasifica
Clasificador de imágenes de vidrio, metal y plástico. Autor: Jean Paul Pineda Pinzón.

## Estado comprobado
Modelo entrenado, exportado y evaluado. Aplicación y API verificadas localmente.
Consulta reports/metrics.json y reports/verification.json.
Pendientes: despliegue público en tu cuenta Render, video y envío al aula.
El caso de uso considera un objeto por imagen y fondos claros. Otros objetos pueden recibir una de las tres clases.

## 1. Ejecutar en Windows
Extrae la carpeta eco_clasifica. Abre PowerShell dentro de esa carpeta.
Usa Python 3.12 o 3.13 de 64 bits. Verifica la versión disponible con py --version.

    py -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe app.py

Abre http://127.0.0.1:5000. La carpeta sample_images contiene nueve fotos reservadas de prueba.
La aplicación permite hasta diez archivos JPG/PNG, máximo 5 MB por foto, carga por lotes y exportación CSV.
Si usas otra versión de Python y pip no encuentra onnxruntime, instala Python 3.12 de 64 bits y recrea el entorno.

## 2. Publicar sin costo en Render
1. Crea un repositorio en tu cuenta GitHub y carga el CONTENIDO de eco_clasifica en su raíz.
2. Incluye app.py, requirements.txt, render.yaml, models/, static/ y templates/.
3. En Render, crea New > Web Service y conecta el repositorio.
4. Selecciona Python y el plan Free. La configuración propuesta usa un proceso y dos threads.
5. Build Command: pip install -r requirements.txt
6. Start Command: gunicorn --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT app:app
7. Configura PYTHON_VERSION=3.12.8 si Render no lo toma del archivo.
8. Comprueba que el costo seleccionado es $0 antes de crear el servicio. Las condiciones de cuentas y cuotas pueden cambiar.
9. Espera el build y abre la URL pública. Consulta /health y prueba las tres clases.
10. Guarda capturas de la configuración Free, el despliegue, la app y una respuesta real de la API.

El modelo tiene aproximadamente 3,6 MB. El consumo real de memoria en Render y la disponibilidad en tu cuenta todavía deben comprobarse.
El plan gratuito tiene cuotas y puede suspenderse tras inactividad. Abre la app antes de grabar y espera la reactivación.
El modelo va incluido en models/ y se carga del repositorio. Las fotos de usuarios se procesan en memoria.
No hay base de datos ni servicios adicionales de pago. Revisa https://render.com/docs/free

## 3. Probar la API pública desde PowerShell
Reemplaza la URL por la que entregue Render y la ruta por una foto guardada.
No pegues los signos < >.

    $url = "https://TU-SERVICIO.onrender.com/predict"
    curl.exe -X POST $url -F "images=@C:\ruta\glass10.jpg"

Para revisar el estado abre https://TU-SERVICIO.onrender.com/health.
También puedes enviar hasta diez campos images en una solicitud multipart/form-data.

## 4. Reproducir entrenamiento y evaluación
Abre Entrenar_EcoClasifica_Colab.ipynb en Google Colab y sigue las celdas.
Sube el ZIP del proyecto que recibiste. El script descarga automáticamente el archivo original TrashNet.
Entrena la capa lineal sobre características de MobileNetV3 Small preentrenado. La base queda congelada.
Máximo 100 épocas, paciencia 15, selección por F1 macro de validación, semilla 42.
La ejecución original se realizó en CPU. El script usa CPU de forma explícita y también funciona en Colab.
En cada ejecución nueva, la prueba se consulta después de seleccionar la versión. El rendimiento puede variar por versiones o hardware.
No ajustes el modelo mirando los resultados de prueba.
El manifiesto agrupa coincidencias perceptuales extremadamente próximas. No garantiza identificar todas las fotos del mismo objeto.
Comprueba reports/manifest.csv y reports/exclusions.json.

## 5. Qué presentar
- Informe final PDF actualizado con la URL pública y capturas del despliegue.
- Enlace al video de máximo diez minutos, accesible para el profesor.
- Demostración individual de la aplicación, métricas y errores. Confirma con el docente la autorización de entrega individual, porque la consigna original menciona trabajo en pareja.
- El Word editable permite completar el informe. Sustituye la sección de pendientes por la evidencia real después del despliegue.
- Consulta GUION_VIDEO.md y LISTA_ENTREGA.md.

## Archivos de evidencia
reports/metrics.json: métricas reales de prueba y validación.
reports/test_predictions.csv: puntuaciones y etiquetas de las 208 imágenes de prueba.
reports/manifest.csv: grupos, clases y particiones.
reports/exclusions.json: seis exclusiones por etiquetas conflictivas.
reports/confusion.png, roc.png, learning.png: gráficas.
reports/verification.json: verificación API y preprocesamiento.
models/model.onnx y metadata.json: modelo desplegable y configuración.
scripts/train.py: entrenamiento reproducible.
scripts/verify_app.py: verificación técnica opcional (requiere las dependencias de entrenamiento y el dataset).

## Fuentes y atribución
TrashNet: Gary Thung y Mindy Yang. https://github.com/garythung/trashnet
Las imágenes de ejemplo provienen de TrashNet. Se incluye TRASHNET_LICENSE.
MobileNetV3 Small: pesos IMAGENET1K_V1 de torchvision. https://docs.pytorch.org/vision/stable/models/mobilenetv3.html
Colab: https://research.google.com/colaboratory/faq.html
ONNX Runtime: https://onnxruntime.ai/docs/
