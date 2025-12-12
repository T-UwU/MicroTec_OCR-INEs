## INE OCR – Extracción y validación heurística

Script en Python para extraer datos de la credencial INE (frente unicamente hasta ahora) usando OCR y aplicar una heurística de calidad.  

- Taiga #11: recalibrar heurística OCR para reducir falsos positivos de INEs “legibles”.
- Taiga #20: crear script para validar OCR

---

## Qué hace

- Aplica varios preprocesamientos a la imagen y corre Tesseract con distintas configuraciones.
- Selecciona el mejor resultado de OCR según un score interno.
- Limpia y normaliza texto.
- Extrae campos clave:
  - CURP
  - Clave de elector
  - Nombre completo y partes
  - Sexo
  - Fecha de nacimiento
  - Domicilio
  - Año de registro, sección, vigencia, etc.
- Valida la CURP y cruza datos:
  - CURP ↔ fecha de nacimiento
  - CURP ↔ sexo
  - CURP ↔ estado
- Devuelve un JSON con los campos extraídos y un bloque `_validacion` con:
  - `valido`  
  - `score` de 0 a 100  
  - detalles de checks y errores

Este `score` es el que se puede usar como filtro de “INE legible / no legible”.

---

## Requisitos

- Python 3.8+
- Tesseract OCR instalado en el sistema  
  - Windows: normalmente en  
    `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Librerías Python:

```bash
pip install opencv-python numpy pytesseract
````

---

## Uso rápido (CLI)

Procesar una imagen de INE:

```bash
python ine_ocr_optimized.py ruta/a/ine_frente.jpg
```

Guardar el resultado en JSON:

```bash
python ine_ocr_optimized.py ruta/a/ine_frente.jpg -o salida.json
```

Activar modo debug para inspeccionar preprocesos y texto:

```bash
python ine_ocr_optimized.py ruta/a/ine_frente.jpg --debug
python ine_ocr_optimized.py ruta/a/ine_frente.jpg --debug --debug-dir ./debug_output
```

En debug se generan imágenes intermedias y un reporte sencillo para revisar por qué se eligió cierto OCR y qué campos se extrajeron.

---

## Uso como módulo

```python
from ine_ocr_optimized import process_single_image, INEExtractor

# Caso simple
data = process_single_image("ruta/a/ine_frente.jpg")
print(data["curp"], data["_validacion"]["score"])

# Uso avanzado con debug
extractor = INEExtractor(debug=True, debug_dir="./debug_output")
data = extractor.extract_to_dict("ruta/a/ine_frente.jpg")
```

`data` es un dict con los campos de la INE más `_validacion`.

---

## Notas y alcance

* Solo trabaja con el frente de la INE.
* Está pensado para INEs con fotos relativamente planas, los ángulos muy abiertos pueden afectar el recorte y el OCR.
* La heurística valida estructura interna de los datos (CURP, fechas, sexo, etcétera), pero no consulta bases externas ni catálogos oficiales.

