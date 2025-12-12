import cv2
import numpy as np
import pytesseract
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict
import logging
import os
from datetime import datetime
import platform

# Configuración de Tesseract según sistema operativo
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTES DE ESTADOS
# ============================================================

ENTIDADES_RENAPO = {
    'AS', 'BC', 'BS', 'CC', 'CL', 'CM', 'CS', 'CH', 'DF', 'DG',
    'GT', 'GR', 'HG', 'JC', 'MC', 'MN', 'MS', 'NT', 'NL', 'OC',
    'PL', 'QT', 'QR', 'SP', 'SL', 'SR', 'TC', 'TS', 'TL', 'VZ', 'YN', 'ZS', 'NE'
}

# Abreviaturas postales de estados (para domicilios)
ESTADOS_ABREV = {
    'AGS': 'Aguascalientes', 'BC': 'Baja California', 'BCS': 'Baja California Sur',
    'CAMP': 'Campeche', 'CAM': 'Campeche', 'COAH': 'Coahuila', 'COL': 'Colima',
    'CHIS': 'Chiapas', 'CHIH': 'Chihuahua', 'CDMX': 'Ciudad de México',
    'DF': 'Ciudad de México', 'DGO': 'Durango', 'GTO': 'Guanajuato',
    'GRO': 'Guerrero', 'HGO': 'Hidalgo', 'JAL': 'Jalisco',
    'MEX': 'Estado de México', 'EDOMEX': 'Estado de México',
    'MICH': 'Michoacán', 'MOR': 'Morelos', 'NAY': 'Nayarit', 'NL': 'Nuevo León',
    'OAX': 'Oaxaca', 'PUE': 'Puebla', 'QRO': 'Querétaro', 'QROO': 'Quintana Roo',
    'SLP': 'San Luis Potosí', 'SIN': 'Sinaloa', 'SON': 'Sonora',
    'TAB': 'Tabasco', 'TAMPS': 'Tamaulipas', 'TAM': 'Tamaulipas',
    'TLAX': 'Tlaxcala', 'VER': 'Veracruz', 'YUC': 'Yucatán', 'ZAC': 'Zacatecas'
}

# Variantes OCR de abreviaturas (con puntos, dos puntos, etc.)
ESTADOS_VARIANTES = {
    'B.C': 'BC', 'B.C.': 'BC', 'B.C.S': 'BCS', 'B.C.S.': 'BCS', 'B.CS': 'BCS',
    'N.L': 'NL', 'N.L.': 'NL', 'D.F': 'DF', 'D.F.': 'DF',
    'Q.ROO': 'QROO', 'S.L.P': 'SLP', 'S.L.P.': 'SLP',
    'AGS.': 'AGS', 'CAMP.': 'CAMP', 'COAH.': 'COAH', 'COL.': 'COL',
    'CHIS.': 'CHIS', 'CHIH.': 'CHIH', 'DGO.': 'DGO', 'GTO.': 'GTO',
    'GRO.': 'GRO', 'HGO.': 'HGO', 'JAL.': 'JAL', 'MEX.': 'MEX',
    'MICH.': 'MICH', 'MOR.': 'MOR', 'NAY.': 'NAY', 'OAX.': 'OAX',
    'PUE.': 'PUE', 'QRO.': 'QRO', 'SIN.': 'SIN', 'SON.': 'SON',
    'TAB.': 'TAB', 'TAMPS.': 'TAMPS', 'TLAX.': 'TLAX', 'VER.': 'VER',
    'YUC.': 'YUC', 'ZAC.': 'ZAC'
}


@dataclass
class INEData:
    """Estructura de datos para almacenar información extraída de la INE"""
    nombre_completo: str = ""
    apellido_paterno: str = ""
    apellido_materno: str = ""
    nombre: str = ""
    domicilio: str = ""
    curp: str = ""
    clave_elector: str = ""
    fecha_nacimiento: str = ""
    sexo: str = ""
    año_registro: str = ""
    seccion: str = ""
    vigencia: str = ""
    estado: str = ""
    municipio: str = ""
    localidad: str = ""
    emision: str = ""
    raw_text: str = ""


class TextCleaner:
    """Clase con métodos de limpieza y normalización de texto"""
    
    # Conectores típicos que se mantienen aunque sean cortos
    CONECTORES = {'DE', 'LA', 'DEL', 'LOS', 'LAS', 'Y', 'DA', 'DO', 'EL'}
    
    KEYWORDS_DIRECCION = {
        'CALLE', 'C', 'AV', 'AVE', 'AVENIDA', 'BLVD', 'BOULEVARD', 'PRIV', 'PRIVADA',
        'AND', 'ANDADOR', 'CERRADA', 'CDA', 'CALLEJON', 'CALZ', 'CALZADA', 'PASEO',
        'PROL', 'PROLONGACION', 'RET', 'RETORNO', 'CIRCUITO', 'CTO', 'CARR', 'CARRETERA',
        'COL', 'COLONIA', 'FRACC', 'FRACCIONAMIENTO', 'BARRIO', 'BO', 'UNIDAD',
        'NUM', 'NO', 'MZ', 'MANZANA', 'LT', 'LOTE', 'INT', 'INTERIOR', 'EXT', 'EXTERIOR',
        'DEPTO', 'DPTO', 'DEPARTAMENTO', 'EDIFICIO', 'EDIF', 'PISO', 'DOMICILIO', 'LOC'
    }
    
    # Palabras que indican fin de nombre/inicio de otros campos
    KEYWORDS_FIN_NOMBRE = {'CURP', 'CLAVE', 'FECHA', 'AÑO', 'ANO', 'REGISTRO', 'DOMICILIO', 'ESTADO', 'SECCION', 'SECCIÓN'}
    
    @staticmethod
    def normalize_global(text: str) -> str:
        """
        Normalización global: mayúsculas, compacta espacios, elimina líneas vacías
        """
        text = text.upper()
        
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Eliminar espacios al inicio y final de cada línea
        lines = [line.strip() for line in text.split('\n')]
        
        lines = [line for line in lines if line]
        
        return '\n'.join(lines)
    
    @staticmethod
    def clean_code_like(text: str) -> str:
        """
        Corrección visual para códigos (CURP, clave de elector)
        Sustituye confusiones comunes del OCR
        """
        replacements = {
            'O': '0',  # O por 0
            'Q': '0',  # Q por 0
            'I': '1',  # I por 1
            'L': '1',  # L por 1 (en contexto de dígitos)
            'Z': '2',  # Z por 2
            'S': '5',  # S por 5
            'B': '8',  # B por 8
        }
        
        result = []
        for i, char in enumerate(text):
            # Solo aplicar corrección en posiciones donde debería haber dígitos
            if char in replacements:
                result.append(replacements[char])
            else:
                result.append(char)
        
        return ''.join(result)
    
    @staticmethod
    def clean_code_selective(text: str, digit_positions: List[int]) -> str:
        """
        Corrección selectiva de código basada en posiciones esperadas de dígitos
        """
        replacements_to_digit = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'}
        
        result = list(text)
        for pos in digit_positions:
            if pos < len(result) and result[pos] in replacements_to_digit:
                result[pos] = replacements_to_digit[result[pos]]
        
        return ''.join(result)
    
    @staticmethod
    def remove_invalid_chars(text: str, allow_digits: bool = True, extra_chars: str = "") -> str:
        """
        Elimina caracteres inválidos, manteniendo solo letras, espacios y opcionalmente dígitos
        """
        pattern = r'[A-ZÁÉÍÓÚÑÜ\s'
        if allow_digits:
            pattern += r'0-9'
        pattern += re.escape(extra_chars) + r']'
        
        return ''.join(re.findall(pattern, text.upper()))
    
    @staticmethod
    def split_concatenated_words(text: str) -> str:
        """
        Intenta separar palabras pegadas (nombres concatenados)
        Busca patrones de mayúsculas que indican inicio de nueva palabra
        """
        # Patrón para detectar cambios mayúscula-minúscula-mayúscula en texto ya en mayúsculas
        # Esto es difícil en texto todo mayúsculas, así que buscamos palabras muy largas
        words = text.split()
        result = []
        
        for word in words:
            if len(word) > 15 and word.isalpha():
                # Intentar separar en posiciones comunes de nombres
                parts = TextCleaner._try_split_name(word)
                result.extend(parts)
            else:
                result.append(word)
        
        return ' '.join(result)
    
    @staticmethod
    def _try_split_name(word: str) -> List[str]:
        """
        Intenta separar un nombre concatenado usando patrones comunes
        """
        # Patrones comunes de terminaciones de apellidos
        suffixes = ['EZ', 'ES', 'OS', 'AS', 'AN', 'ON', 'IA', 'IO', 'EZ', 'IZ', 'UZ', 'AZ', 'OZ']
        
        for i in range(6, len(word) - 3):
            if word[i-2:i] in suffixes:
                return [word[:i], word[i:]]
        
        return [word]


class INEExtractor:
    """Clase principal para extraer datos de credenciales INE"""
    
    def __init__(self, tesseract_config: str = '--oem 3 --psm 3', debug: bool = False, 
                 debug_dir: str = None):
        """Inicializa el extractor"""
        self.tesseract_config = tesseract_config
        self.cleaner = TextCleaner()
        self.debug = debug
        self.debug_dir = debug_dir
        self.debug_data = {}  # Almacena datos de debug
        
        if self.debug and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
    
    def _save_debug_image(self, name: str, image: np.ndarray):
        """Guarda una imagen de debug"""
        if self.debug and self.debug_dir:
            path = os.path.join(self.debug_dir, f"{name}.jpg")
            cv2.imwrite(path, image)
            logger.info(f"[DEBUG] Imagen guardada: {path}")
    
    def _log_debug(self, key: str, value):
        """Registra información de debug"""
        if self.debug:
            self.debug_data[key] = value
            if isinstance(value, str) and len(value) < 200:
                logger.info(f"[DEBUG] {key}: {value}")
            elif isinstance(value, (int, float)):
                logger.info(f"[DEBUG] {key}: {value}")
    
    def generate_debug_report(self, output_path: str = None) -> str:
        """Genera un reporte HTML simplificado con información de debug"""
        if not self.debug:
            return ""
        
        best_method = self.debug_data.get('best_method', '')
        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>INE OCR Debug</title>
<style>
body{{font-family:Arial,sans-serif;margin:20px;background:#f5f5f5}}
.section{{background:white;padding:15px;margin:15px 0;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}
th{{background:#007bff;color:white}}pre{{background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:4px;font-size:10px;max-height:300px;overflow:auto}}
.best{{background:#d4edda;border:2px solid #28a745}}.score{{padding:3px 8px;border-radius:4px;background:#28a745;color:white}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}}
.grid img{{max-width:100%;border:1px solid #ddd;border-radius:4px}}
</style></head><body>
<h1>INE OCR Debug Report</h1>
<p>Generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Mejor método: <span class="score">{best_method}</span> | Score: {self.debug_data.get('best_score', 'N/A')}</p>
"""
        
        if self.debug_dir and os.path.exists(self.debug_dir):
            html += '<div class="section"><h3>Imágenes</h3><div class="grid">'
            for img in ['00_original.jpg', '00b_cropped.jpg', '05_best_preprocessing.jpg']:
                if os.path.exists(os.path.join(self.debug_dir, img)):
                    html += f'<div><img src="{img}"><p>{img}</p></div>'
            html += '</div></div>'
        
        if 'preprocessing_scores' in self.debug_data:
            html += '<div class="section"><h3>Scores</h3><table><tr><th>Método</th><th>Score</th><th>Keywords</th></tr>'
            sorted_methods = sorted(self.debug_data['preprocessing_scores'].items(), key=lambda x: x[1].get('score', 0), reverse=True)
            for method, data in sorted_methods[:10]:
                score = data.get('score', 0)
                kw = ', '.join(data.get('keywords', [])[:5])
                cls = ' class="best"' if method == best_method else ''
                html += f'<tr{cls}><td>{method}</td><td>{score:.1f}</td><td>{kw}</td></tr>'
            html += '</table></div>'
        
        if 'all_ocr_texts' in self.debug_data and best_method:
            text = self.debug_data['all_ocr_texts'].get(best_method, '')
            html += f'<div class="section"><h3>OCR - {best_method}</h3><pre>{text}</pre></div>'
        
        if 'extraction_result' in self.debug_data:
            html += '<div class="section"><h3>Resultado</h3><table>'
            for k, v in self.debug_data['extraction_result'].items():
                if k != 'raw_text':
                    html += f'<tr><td><b>{k}</b></td><td>{v or "-"}</td></tr>'
            html += '</table></div>'
        
        html += '</body></html>'
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
        
        return html


    def crop_data_region(self, image: np.ndarray) -> np.ndarray:
        """Recorta la imagen de INE para quedarse solo con la región de datos."""
        height, width = image.shape[:2]
        
        # Porcentajes de recorte basados en el diseño estándar de INE
        left_crop_percent = 0.28   # Eliminar foto y firma (lado izquierdo) - menos agresivo
        top_crop_percent = 0.18    # Eliminar encabezado (MEXICO, INE, CREDENCIAL)
        right_crop_percent = 0.03  # Mantener más área derecha
        
        x_start = int(width * left_crop_percent)
        x_end = int(width * (1 - right_crop_percent))
        y_start = int(height * top_crop_percent)
        
        cropped = image[y_start:height, x_start:x_end]
        
        if self.debug:
            logger.debug(f"Imagen recortada: {width}x{height} -> {cropped.shape[1]}x{cropped.shape[0]}")
        
        return cropped
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesa la imagen para mejorar el OCR. Retorna (gris, color)"""
        original = image.copy()
        height, width = image.shape[:2]
        
        # Escalar imágenes pequeñas para mejorar OCR
        target_width = 2400 if width < 1200 else (2200 if width < 1600 else (2000 if width < 2000 else width))
        
        if width < target_width:
            scale = target_width / width
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            original = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), original
    
    def try_multiple_preprocessing(self, image: np.ndarray) -> str:
        """Prueba múltiples configuraciones de preprocesamiento y retorna el mejor resultado"""
        results = []
        all_images = {}
        all_texts = {}
        preprocessing_scores = {}
        
        keywords = ['NOMBRE', 'CURP', 'CLAVE', 'ELECTOR', 'DOMICILIO', 'FECHA', 'NACIMIENTO', 
                   'SEXO', 'SECCION', 'VIGENCIA', 'REGISTRO', 'ESTADO', 'MUNICIPIO', 'LOCALIDAD', 'EMISION']
        high_priority_keywords = ['CURP', 'CLAVE', 'ELECTOR', 'NOMBRE', 'DOMICILIO', 'FECHA']
        
        if self.debug:
            self._save_debug_image('00_original', image)
        
        # Recortar región de datos
        image_to_process = self.crop_data_region(image)
        if self.debug:
            self._save_debug_image('00b_cropped', image_to_process)
        
        # Preprocesar imagen
        gray, color = self.preprocess_image(image_to_process)
        
        if self.debug:
            self._save_debug_image('01_color', color)
            self._save_debug_image('02_gray', gray)
        
        # Configuraciones de preprocesamiento
        configs = []
        configs.append(('gray_psm3', gray, '--oem 3 --psm 3'))
        configs.append(('gray_psm6', gray, '--oem 3 --psm 6'))
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        configs.append(('clahe_psm3', clahe_img, '--oem 3 --psm 3'))
        configs.append(('clahe_psm6', clahe_img, '--oem 3 --psm 6'))
        
        clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        configs.append(('clahe_strong_psm3', clahe_strong.apply(gray), '--oem 3 --psm 3'))
        
        # Normalización de iluminación
        kernel_size = max(gray.shape) // 20
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        kernel_size = max(kernel_size, 15)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        normalized = cv2.normalize(cv2.subtract(background, gray), None, 0, 255, cv2.NORM_MINMAX)
        configs.append(('normalized_psm3', normalized, '--oem 3 --psm 3'))
        
        background_blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        div_normalized = cv2.divide(gray, background_blur, scale=255)
        configs.append(('div_normalized_psm3', div_normalized, '--oem 3 --psm 3'))
        
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        configs.append(('otsu_psm3', otsu, '--oem 3 --psm 3'))
        configs.append(('otsu_psm6', otsu, '--oem 3 --psm 6'))
        
        _, otsu_clahe = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        configs.append(('otsu_clahe_psm3', otsu_clahe, '--oem 3 --psm 3'))
        
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        configs.append(('adaptive_psm3', adaptive, '--oem 3 --psm 3'))
        
        adaptive_clahe = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        configs.append(('adaptive_clahe_psm3', adaptive_clahe, '--oem 3 --psm 3'))
        
        configs.append(('gray_psm4', gray, '--oem 3 --psm 4'))
        configs.append(('gray_psm11', gray, '--oem 3 --psm 11'))
        
        for name, img, config in configs:
            try:
                text = pytesseract.image_to_string(img, config=config)
                results.append((name, text, img))
            except Exception as e:
                logger.warning(f"Error en OCR con {name}: {e}")
        
        best_text = ""
        best_score = -1
        best_name = ""
        best_img = None
        
        for name, text, img in results:
            text_upper = text.upper()
            
            found_keywords = [kw for kw in keywords if kw in text_upper]
            keyword_score = len(found_keywords) * 15
            
            high_priority_found = [kw for kw in high_priority_keywords if kw in text_upper]
            priority_bonus = len(high_priority_found) * 10
            
            # Score por longitud (texto más largo = mejor, hasta cierto punto)
            length_score = min(len(text) / 10, 50)
            
            # Bonus por estructura (si tiene saltos de línea organizados)
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            line_count = len(lines)
            structure_bonus = min(line_count * 2, 30) if line_count > 5 else 0
            
            # Penalización por mucho ruido (caracteres especiales)
            noise_chars = len(re.findall(r'[^A-Za-z0-9\s\.,/\-áéíóúñÁÉÍÓÚÑ:()]', text))
            noise_penalty = min(noise_chars * 1.0, 50)  # Aumentada penalización
            
            # Penalización extra por ruido al inicio (indica mal procesamiento)
            first_50_chars = text[:50] if len(text) > 50 else text
            noise_at_start = len(re.findall(r'[^A-Za-z0-9\s\.,/\-áéíóúñÁÉÍÓÚÑ%]', first_50_chars))
            start_noise_penalty = noise_at_start * 2
            
            # Bonus por comenzar con texto válido (MEXICO, INSTITUTO, etc.)
            start_bonus = 0
            first_line = lines[0].upper() if lines else ""
            valid_starts = ['MEXICO', 'INSTITUTO', 'CREDENCIAL', '%']  # % común en OCR de logo
            if any(vs in first_line for vs in valid_starts):
                start_bonus = 15
            
            # Bonus para métodos que históricamente funcionan mejor
            method_bonus = 0
            if 'psm3' in name:
                method_bonus = 20  # PSM 3 generalmente funciona mejor para INE (aumentado)
            elif 'psm6' in name:
                method_bonus = 15  # PSM 6 también es bueno
            elif 'psm4' in name:
                method_bonus = 5
            elif 'psm11' in name:
                method_bonus = -15
            
            if 'gray' in name and 'clahe' not in name and 'otsu' not in name:
                method_bonus += 10
            
            score = (keyword_score + priority_bonus + length_score + structure_bonus + 
                    method_bonus + start_bonus - noise_penalty - start_noise_penalty)
            
            all_texts[name] = text
            preprocessing_scores[name] = {
                'score': round(score, 1),
                'keywords': found_keywords,
                'high_priority': high_priority_found,
                'keyword_score': keyword_score,
                'priority_bonus': priority_bonus,
                'length_score': round(length_score, 1),
                'structure_bonus': structure_bonus,
                'method_bonus': method_bonus,
                'start_bonus': start_bonus,
                'noise_penalty': round(noise_penalty, 1),
                'start_noise_penalty': start_noise_penalty,
                'text_length': len(text),
                'line_count': line_count
            }
            
            if score > best_score:
                best_score = score
                best_text = text
                best_name = name
                best_img = img
        
        if self.debug:
            self._log_debug('best_method', best_name)
            self._log_debug('best_score', best_score)
            self.debug_data['preprocessing_scores'] = preprocessing_scores
            self.debug_data['all_ocr_texts'] = all_texts
            self.debug_data['best_ocr_text'] = best_text
            
            if best_img is not None:
                self._save_debug_image('05_best_preprocessing', best_img)
            
            for name, text, img in results:
                safe_name = name.replace('_', '-')
                self._save_debug_image(f'prep_{safe_name}', img)
        
        logger.info(f"Mejor preprocesamiento: {best_name} (score: {best_score:.1f})")
        
        self.all_ocr_results = all_texts
        
        return best_text
    
    def build_consensus_data(self) -> Dict:
        """Analiza todos los resultados OCR para encontrar palabras/patrones consistentes."""
        if not hasattr(self, 'all_ocr_results') or not self.all_ocr_results:
            return {}
        
        from collections import Counter
        
        # Contar frecuencia de palabras en todos los resultados
        word_counts = Counter()
        name_candidates = Counter()  # Palabras cerca de NOMBRE
        municipio_candidates = Counter()  # Palabras antes de abreviaturas de estado
        
        estados_abrev_set = set(ESTADOS_ABREV.keys())
        
        # Palabras que NO son nombres (keywords de INE, ruido común, teclas de teclado)
        non_name_words = {
            'NOMBRE', 'DOMICILIO', 'CURP', 'CLAVE', 'ELECTOR', 'FECHA', 'NACIMIENTO',
            'SEXO', 'SECCION', 'VIGENCIA', 'INSTITUTO', 'NACIONAL', 'ELECTORAL', 
            'MEXICO', 'CREDENCIAL', 'VOTAR', 'REGISTRO', 'ESTADO', 'MUNICIPIO', 
            'LOCALIDAD', 'EMISION', 'COL', 'COLONIA', 'CALLE', 'AV', 'AVENIDA',
            'NUM', 'NUMERO', 'CP', 'ANO', 'DE', 'LA', 'EL', 'LOS', 'LAS', 'DEL',
            'PARA', 'POR', 'CON', 'SIN', 'CLAVEDEELECTOR', 'ANODEREGISTRO',
            'FECHADENACIMIENTO', 'FECHADE', 'AÑODE', 'AÑODEREGISTRO',
            'PAGE', 'UP', 'DOWN', 'CE', 'LE', 'AF', 'SS', 'END', 'HOME', 'INSERT',
            'DELETE', 'ENTER', 'SHIFT', 'CTRL', 'ALT', 'TAB', 'ESC', 'CAPS',
            'HASS', 'VIK', 'ND', 'PIE', 'LATION', 'FATH', 'SNG', 'IO', 'ORACH',
            'EAD', 'OI', 'JAEN', 'HIA', 'URYELE', 'AOE', 'GF', 'EY', 'OM', 'EA',
            'AE', 'ET', 'II', 'AIME', 'ORE', 'BAAS', 'CAN', 'BEAR', 'SEES', 
            'VIERO', 'PERE', 'PER', 'SATHALRALL', 'YATHAIMALEKA', 'IEKA',
            'TTT', 'III', 'OOO', 'AAA', 'EEE', 'REN', 'AOF', 'SE', 'IM', 'EY',
            'CENTRO', 'EJIDAL', 'INSURGENTES', 'NACIONAL', 'CHAMIZAL', 'OPODEPE'
        }
        
        def is_valid_name(word: str) -> bool:
            """Verifica si una palabra parece un nombre válido"""
            if len(word) < 3:
                return False
            if word in non_name_words:
                return False
            vowels = set('AEIOU')
            if not any(c in vowels for c in word):
                return False
            if len(set(word)) <= 2 and len(word) > 4:
                return False
            consonants = 'BCDFGHJKLMNÑPQRSTVWXYZ'
            cons_count = 0
            for c in word:
                if c in consonants:
                    cons_count += 1
                    if cons_count > 3:
                        return False
                else:
                    cons_count = 0
            if word.startswith(('SATH', 'YATH', 'HASS', 'IMAL', 'RALL')):
                return False
            return True
        
        for method_name, text in self.all_ocr_results.items():
            text_upper = text.upper()
            lines = text_upper.split('\n')
            
            words = re.findall(r'\b[A-ZÁÉÍÓÚÑ]{3,}\b', text_upper)
            for word in words:
                if is_valid_name(word):
                    word_counts[word] += 1
            
            for i, line in enumerate(lines):
                if 'NOMBRE' in line:
                    for j in range(i, min(i+5, len(lines))):
                        line_words = re.findall(r'\b[A-ZÁÉÍÓÚÑ]{3,}\b', lines[j])
                        for word in line_words:
                            if (is_valid_name(word) and 
                                not word.isdigit() and
                                not re.match(r'^[A-Z]{4}\d', word)):  # No CURP
                                name_candidates[word] += 1
            
            # Incluye soporte para municipios compuestos como "LOS CABOS", "SAN LUIS", etc.
            for line in lines:
                for estado in estados_abrev_set:
                    # Patrón para municipios compuestos (LOS CABOS, SAN LUIS, PUERTO VALLARTA, etc.)
                    compound_pattern = rf'((?:LOS|LAS|SAN|SANTA|PUERTO|VILLA|CIUDAD|NUEVO|NUEVA|GENERAL)\s+[A-ZÁÉÍÓÚÑ]{{3,}}),?\s*{estado}\.?(?:\s|,|$)'
                    match = re.search(compound_pattern, line)
                    if match:
                        municipio = match.group(1).strip()
                        if municipio not in non_name_words:
                            municipio_candidates[municipio] += 2  # Bonus por ser compuesto
                    
                    simple_patterns = [
                        rf'([A-ZÁÉÍÓÚÑ]{{4,}}),?\s*{estado}\.?\s*$',
                        rf'([A-ZÁÉÍÓÚÑ]{{4,}}),?\s*{estado}\.?(?:\s|,|$)',
                    ]
                    for pattern in simple_patterns:
                        match = re.search(pattern, line)
                        if match:
                            municipio = match.group(1)
                            if municipio not in non_name_words and len(municipio) >= 4:
                                municipio_candidates[municipio] += 1
        
        # Filtrar por frecuencia mínima (debe aparecer en al menos 3 métodos)
        min_frequency = 3
        
        filtered_name_candidates = {word: count for word, count in name_candidates.items() 
                                   if count >= min_frequency and is_valid_name(word)}
        
        consensus = {
            'frequent_words': {word: count for word, count in word_counts.items() if count >= min_frequency},
            'name_candidates': filtered_name_candidates,
            'municipio_candidates': {word: count for word, count in municipio_candidates.items() if count >= 2},
            'total_methods': len(self.all_ocr_results)
        }
        
        if consensus['municipio_candidates']:
            best_municipio = max(consensus['municipio_candidates'].items(), key=lambda x: x[1])
            consensus['best_municipio'] = best_municipio[0]
            consensus['best_municipio_confidence'] = best_municipio[1] / len(self.all_ocr_results)
        
        if consensus['name_candidates']:
            sorted_names = sorted(consensus['name_candidates'].items(), key=lambda x: x[1], reverse=True)
            consensus['probable_names'] = [name for name, count in sorted_names[:6]]  # Top 6
        
        if self.debug:
            self.debug_data['consensus'] = consensus
        
        return consensus
    
    def extract_nombre_with_consensus(self, text: str, consensus: Dict) -> Tuple[str, str, str, str]:
        """Extrae nombre usando consenso de múltiples OCR."""
        probable_names = consensus.get('probable_names', [])
        frequent_words = consensus.get('frequent_words', {})
        
        if not probable_names:
            nombre_completo, _ = self.extract_nombre_domicilio(text)
            apellido_paterno, apellido_materno, nombre = self.parse_nombre_components(nombre_completo)
            return nombre_completo, apellido_paterno, apellido_materno, nombre
        
        # Filtrar nombres probables que realmente parecen nombres
        common_first_names = {
            # Nombres que ya tenías
            'JOSE', 'JUAN', 'LUIS', 'CARLOS', 'MIGUEL', 'ANTONIO', 'FRANCISCO', 'JESUS',
            'PEDRO', 'MANUEL', 'RAFAEL', 'FERNANDO', 'RICARDO', 'ROBERTO', 'ALBERTO',
            'MARIA', 'ANA', 'ROSA', 'CARMEN', 'PATRICIA', 'ELIZABETH', 'GUADALUPE',
            'MARTHA', 'ADRIANA', 'ALEJANDRA', 'VERONICA', 'SANDRA', 'DIANA', 'MARIANA',
            'JONATHAN', 'ALEXANDER', 'ADRIAN', 'JAIRO', 'DAVID', 'EDGAR', 'OSCAR',
            'DANIEL', 'SERGIO', 'JORGE', 'RAUL', 'ARTURO', 'EDUARDO', 'ENRIQUE',
            'GABRIELA', 'CLAUDIA', 'LETICIA', 'NORMA', 'SILVIA', 'LAURA', 'TERESA',
            'ISMAEL', 'IVAN', 'IRENE', 'ISABEL', 'BRENDA', 'LIZETH', 'RODOLFO',
            'SALVADOR', 'HECTOR', 'HUGO', 'GABRIEL', 'GUILLERMO', 'GERARDO', 'ANGEL',
            'CRISTIAN', 'CHRISTIAN', 'FABIAN', 'FELIPE', 'ERNESTO', 'OMAR', 'PABLO',
            'PAOLA', 'TOMAS', 'VALERIA', 'VICTOR', 'VIRGINIA', 'SOFIA', 'MONICA',
            'JULIO', 'RUBEN', 'MARTIN', 'MARCO', 'ANDRES',

            'ALAN', 'ALVARO', 'ABRAHAM', 'ABEL', 'AGUSTIN', 'ALFONSO', 'ALFREDO',
            'ARMANDO', 'AXEL', 'BENJAMIN', 'BRAULIO', 'BRAYAN', 'BRYAN',
            'CESAR', 'CRISTOBAL', 'DAMIAN', 'DIEGO', 'DOMINGO', 'ELIAS',
            'EMANUEL', 'EMMANUEL', 'EMILIO', 'ERICK', 'ERIK', 'ERIC',
            'ESTEBAN', 'EZEQUIEL', 'FEDERICO', 'FRANCO', 'FROILAN',
            'GAEL', 'GILBERTO', 'GONZALO', 'GUADALUPE',
            'HORACIO', 'IGNACIO', 'ISAIAS', 'ISRAEL', 'IVAN', 'JAIME',
            'JAVIER', 'JOAQUIN', 'JOEL', 'JONAS', 'JONAS', 'JORDAN',
            'JOSUE', 'JULIAN', 'KEVIN', 'LEONARDO', 'LEONEL', 'LEOPOLDO',
            'LORENZO', 'MARIO', 'MATEO', 'MATEO', 'MAURICIO', 'MAXIMILIANO',
            'MELCHOR', 'MOISES', 'NAHUM', 'NICOLAS', 'NOE',
            'OCTAVIO', 'ORLANDO', 'PATRICIO', 'PEDRO', 'RAFAEL',
            'RAMIRO', 'RAMON', 'RENATO', 'RENE', 'RICARDO', 'ROGELIO',
            'SAID', 'SAMUEL', 'SEBASTIAN', 'TIRSO', 'ULISES', 'ULISES',
            'URIEL', 'VICENTE', 'WILLIAM', 'XAVIER', 'YAHIR', 'YAIR',
            'ZACARIAS',

            'ABIGAIL', 'ALEJANDRA', 'ALICIA', 'ALMA', 'AMALIA', 'AMERICA',
            'ANDREA', 'ANGELA', 'ANGELICA', 'ARACELI', 'ARIADNA', 'ARIANA',
            'AURORA', 'BEATRIZ', 'BERENICE', 'BERENICE', 'BETTY', 'BLANCA',
            'CAMILA', 'CARLA', 'CAROLINA', 'CECILIA', 'CELIA', 'CINTIA',
            'CYNTHIA', 'CONCEPCION', 'CRISTINA',
            'DANIELA', 'DELIA', 'DORA', 'DULCE',
            'EDITH', 'ELENA', 'ELSA', 'ELVIA', 'ELVIRA',
            'ERIKA', 'ESMERALDA', 'ESTELA', 'ESTHER', 'EVA', 'EVELIA', 'EVELYN',
            'FABIOLA', 'FATIMA', 'FLOR', 'FLORINDA',
            'GEORGINA', 'GLADYS', 'GLORIA', 'GRACIELA',
            'INES', 'IRMA', 'ITZEL', 'IVONNE',
            'JACQUELINE', 'JACKELINE', 'JAZMIN', 'JESSICA', 'JIMENA', 'JOHANNA',
            'JOSEFINA', 'JUANA',
            'KAREN', 'KARINA', 'KARLA', 'KENIA',
            'LAURA', 'LEONOR', 'LIDIA', 'LILIANA', 'LISBETH', 'LIZBETH', 'LOURDES',
            'LUCERO', 'LUCIA', 'LUPITA', 'LUZ',
            'MAGDALENA', 'MALENA', 'MANUELA', 'MARCELA', 'MARGARITA',
            'MARIA FERNANDA', 'MARIA GUADALUPE', 'MARIA ELENA',  # si quieres puedes luego limpiar compuestos
            'MARIBEL', 'MARICELA', 'MARIELA', 'MARISOL', 'MAYRA',
            'MELINA', 'MELISSA', 'MIREYA', 'MIRNA',
            'NANCY', 'NATALIA', 'NAYELI', 'NEREIDA', 'NOEMI', 'NORA',
            'OLGA', 'OLIVIA',
            'PALOMA', 'PAOLA', 'PAULINA', 'PERLA', 'PILAR',
            'REBECA', 'ROCIO', 'ROSALBA', 'ROSALIA', 'ROSARIO',
            'SELENE', 'SOLEDAD', 'SOCORRO', 'SUSANA',
            'TANIA', 'THALIA',
            'URBANA',
            'VANESSA', 'VERA', 'VERONICA', 'VIOLETA',
            'XOCHITL',
            'YADIRA', 'YAMILETH', 'YARELI', 'YARELY', 'YAZMIN', 'YESENIA',
            'YOLANDA', 'YULIANA', 'YURIDIA',

            'ALEX', 'ANDY', 'CRUZ', 'LUZ', 'TRINIDAD'
        }

        
        # Separar en apellidos (primero) y nombres (después)
        apellidos = []
        nombres = []
        
        for name in probable_names:
            if name in common_first_names:
                nombres.append(name)
            else:
                apellidos.append(name)
        
        apellido_paterno = apellidos[0] if len(apellidos) >= 1 else ""
        apellido_materno = apellidos[1] if len(apellidos) >= 2 else ""
        
        if not apellido_paterno and nombres:
            original_nombre, _ = self.extract_nombre_domicilio(text)
            orig_ap, orig_am, _ = self.parse_nombre_components(original_nombre)
            apellido_paterno = orig_ap
            apellido_materno = orig_am
        
        nombre = ' '.join(nombres) if nombres else ""
        
        if not nombre and len(apellidos) > 2:
            nombre = apellidos[2]
        
        parts = [p for p in [apellido_paterno, apellido_materno, nombre] if p]
        nombre_completo = ' '.join(parts)
        
        return nombre_completo, apellido_paterno, apellido_materno, nombre
    
    def extract_municipio_with_consensus(self, domicilio: str, consensus: Dict, estado: str) -> str:
        """Extrae municipio usando consenso y validación mejorada."""
        # Palabras que pueden ser parte de nombres de municipios compuestos
        compound_prefixes = {'LOS', 'LAS', 'SAN', 'SANTA', 'PUERTO', 'VILLA', 'CIUDAD', 
                           'NUEVO', 'NUEVA', 'GENERAL', 'EL', 'LA', 'DEL'}
        
        best_municipio = consensus.get('best_municipio', '')
        municipio_confidence = consensus.get('best_municipio_confidence', 0)
        
        if best_municipio and municipio_confidence >= 0.3:
            return best_municipio
        
        if not domicilio or not estado:
            return best_municipio if best_municipio else ""
        
        domicilio_upper = domicilio.upper().strip()
        
        estado_pattern = rf',?\s*{re.escape(estado)}\.?\s*$'
        match = re.search(estado_pattern, domicilio_upper)
        
        if match:
            before_estado = domicilio_upper[:match.start()].strip().rstrip(',').strip()
            
            cp_match = re.search(r'(\d{5})\s+(.+)$', before_estado)
            if cp_match:
                after_cp = cp_match.group(2).strip()
                
                words = after_cp.split()
                clean_words = []
                for word in words:
                    # Mantener si es prefijo de municipio compuesto o palabra larga
                    if word in compound_prefixes or len(word) >= 3:
                        clean_words.append(word)
                
                if clean_words:
                    return ' '.join(clean_words)
            
            words = before_estado.split()
            if words:
                municipio_words = []
                for word in reversed(words):
                    if re.match(r'^\d+$', word):
                        break
                    # Incluir prefijos de municipios compuestos aunque sean cortos
                    if word in compound_prefixes or (len(word) >= 3 and word not in {'COL', 'NUM', 'AV', 'CP'}):
                        municipio_words.insert(0, word)
                        if len(municipio_words) >= 2 and municipio_words[0] not in compound_prefixes:
                            break
                        if len(municipio_words) >= 3:
                            break
                
                if municipio_words:
                    return ' '.join(municipio_words)
        
        return best_municipio if best_municipio else ""
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extrae texto de la imagen usando Tesseract OCR con múltiples preprocesamientos
        """
        # Probar múltiples configuraciones de preprocesamiento
        text = self.try_multiple_preprocessing(image)
        
        text = self.cleaner.normalize_global(text)
        
        return text
    
    def extract_curp(self, text: str) -> str:
        """Extrae el CURP aplicando heurística avanzada"""
        text_upper = text.upper()
        
        # Patrones de CURP con flexibilidad para errores OCR comunes:
        # - Posiciones 4-9 (fecha): dígitos o letras que parecen dígitos (O→0, I→1, etc.)
        # - Posiciones 16-17: alfanuméricos (el último suele ser dígito pero OCR puede confundir)
        
        # Patrón más flexible que acepta confusiones OCR comunes
        # [0-9OI]{6} = 6 caracteres que son dígitos o confusiones comunes (O→0, I→1)
        # [A-Z0-9OI]{2} = 2 caracteres alfanum (OCR puede poner O en vez de 0)
        pattern_flexible = r'[A-Z]{4}[0-9OIL]{6}[HM][A-Z]{5}[A-Z0-9]{2}'
        
        curp_variants = [
            r'CURP',      # Correcto
            r'CURE',      # P → E (muy común)
            r'CURI',      # P → I
            r'CURR',      # P → R
            r'CURF',      # P → F
            r'CVRP',      # U → V
            r'C0RP',      # U → 0
            r'CORP',      # U → O
            r'CUR[PEFRIB]',  # Patrón general
            r'C[UV0O]R[PEFRIB]',  # Patrón más general
        ]
        
        # Primero buscar cerca de cualquier variante de CURP (más confiable)
        for variant in curp_variants:
            curp_match = re.search(rf'{variant}[:\s]*(.{{18,25}})', text_upper)
            if curp_match:
                raw_curp = curp_match.group(1)
                cleaned = re.sub(r'[^A-Z0-9]', '', raw_curp)[:18]
                
                if len(cleaned) >= 18:
                    candidate = cleaned[:18]
                    corrected = self._correct_curp(candidate)
                    if self._validate_curp_structure(corrected):
                        return corrected
        
        match = re.search(pattern_flexible, text_upper)
        if match:
            candidate = match.group()
            corrected = self._correct_curp(candidate)
            if self._validate_curp_structure(corrected):
                return corrected
        
        # Ejemplo: "SAGJ941001 HCSRRROS" -> "SAGJ941001HCSRRRO8"
        pattern_with_space = r'([A-Z]{4}[0-9OIL]{6})\s*([HM][A-Z]{5}[A-Z0-9]{2})'
        match = re.search(pattern_with_space, text_upper)
        if match:
            candidate = match.group(1) + match.group(2)
            if len(candidate) == 18:
                corrected = self._correct_curp(candidate)
                if self._validate_curp_structure(corrected):
                    return corrected
        
        for variant in ['CURP', 'CURE', 'CURI', 'CURR', 'CURF']:
            curp_section = self._find_section(text_upper, variant, window=80)
            
            if curp_section:
                cleaned = re.sub(r'[^A-Z0-9]', '', curp_section)
                
                for i in range(len(cleaned) - 17):
                    candidate = cleaned[i:i+18]
                    
                    corrected = self._correct_curp(candidate)
                    
                    if self._validate_curp_structure(corrected):
                        return corrected
        
        # Último intento: buscar cualquier secuencia que parezca CURP en todo el texto
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text_upper)
        for i in range(len(cleaned_text) - 17):
            candidate = cleaned_text[i:i+18]
            # Permitir O/I en posiciones de dígitos (serán corregidos)
            date_part = candidate[4:10]
            date_valid = all(c.isdigit() or c in 'OIL' for c in date_part)
            
            if (candidate[:4].isalpha() and 
                date_valid and 
                candidate[10] in 'HM' and
                candidate[11:16].isalpha()):
                corrected = self._correct_curp(candidate)
                if self._validate_curp_structure(corrected):
                    return corrected
        
        return ""
    
    def _validate_curp_structure(self, candidate: str, strict: bool = True) -> bool:
        """Valida la estructura básica de un candidato a CURP según reglas oficiales RENAPO"""
        if len(candidate) != 18:
            return False
        
        if not candidate[:4].isalpha():
            return False
        
        # Validar que posición 1 sea vocal (primera vocal interna del primer apellido)
        vowels = 'AEIOUX'  # X se usa para apellidos sin vocal
        if candidate[1] not in vowels:
            # Algunos apellidos especiales usan X, pero si no es vocal ni X, rechazar
            pass  # Ser permisivo aquí, algunos CURP tienen excepciones
        
        date_part = candidate[4:10]
        if not date_part.isdigit():
            if strict:
                return False
        
        try:
            yy = int(candidate[4:6])
            mm = int(candidate[6:8])
            dd = int(candidate[8:10])
            
            if mm < 1 or mm > 12:
                return False
            if dd < 1 or dd > 31:
                return False
            
            days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if dd > days_in_month[mm - 1]:
                return False
                
        except ValueError:
            if strict:
                return False
        
        if candidate[10] not in ['H', 'M']:
            return False
        
        estado_code = candidate[11:13]
        if estado_code not in ENTIDADES_RENAPO:
            if strict:
                corrected_codes = {
                    'C5': 'CS', '5S': 'CS',  # Chiapas
                    'B5': 'BS',              # BCS
                    'N1': 'NL',              # Nuevo León
                    '5L': 'SL',              # Sinaloa
                    '5P': 'SP',              # San Luis Potosí
                    '5R': 'SR',              # Sonora
                    'T5': 'TS',              # Tamaulipas
                }
                if estado_code not in corrected_codes:
                    return False
        
        # Posiciones 13-15 deben ser letras (consonantes internas)
        if not candidate[13:16].isalpha():
            return False
        
        # Posición 16: puede ser letra (nacidos 2000+) o dígito (nacidos hasta 1999)
        try:
            yy = int(candidate[4:6])
            if yy <= 25:  # Nacidos 2000-2025
                pass
            else:  # Nacidos 1926-1999
                if not candidate[16].isdigit():
                    pass
        except ValueError:
            pass
        
        last_char = candidate[-1]
        if not last_char.isdigit():
            correctable_chars = {'O', 'S', 'B', 'I', 'L', 'Z', 'Q', 'D'}
            if strict and last_char not in correctable_chars:
                return False
        
        return True
    
    def _correct_curp(self, curp: str) -> str:
        """Corrige un CURP aplicando reglas específicas por posición"""
        if len(curp) != 18:
            return curp
        
        result = list(curp)
        
        # Correcciones para posiciones que deben ser letras (0-3)
        letter_replacements = {
            '0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z',
        }
        for pos in [0, 1, 2, 3]:
            if result[pos] in letter_replacements:
                result[pos] = letter_replacements[result[pos]]
        
        # Correcciones para posiciones que deben ser dígitos (4-9, fecha)
        digit_replacements = {
            'O': '0', 'Q': '0', 'D': '0',  # Formas de 0
            'I': '1', 'L': '1', 'T': '1',  # Formas de 1
            'Z': '2',                       # Forma de 2
            'E': '3',                       # Forma de 3
            'A': '4',                       # Forma de 4 (4 confundido con A)
            'S': '5',                       # Forma de 5
            'G': '6', 'b': '6',             # Forma de 6
            'B': '8',                       # Forma de 8
            'g': '9',                       # Forma de 9
        }
        
        for pos in [4, 5, 6, 7, 8, 9]:
            if result[pos] in digit_replacements:
                result[pos] = digit_replacements[result[pos]]
        
        try:
            mm = int(result[6] + result[7])  # Mes
            dd = int(result[8] + result[9])  # Día
            
            if mm > 12:
                if result[6] == '1' and result[7] == '9':
                    result[7] = '0'
                elif result[6] == '4':
                    result[6] = '0'
            
            if dd > 31:
                if result[8] == '4':
                    result[8] = '1'
                elif result[8] == '9':
                    result[8] = '0'
                elif result[8] == '7':
                    result[8] = '1'
        except ValueError:
            pass
        
        if result[10] not in ['H', 'M']:
            if result[10] in ['N', 'W']:
                result[10] = 'H'  # N/W confundido con H
            elif result[10] in ['W', 'N']:
                result[10] = 'M'  # Menos común
        
        # Posiciones 11-12: código entidad federativa RENAPO
        estado_corrections = {
            'C5': 'CS', '5S': 'CS', 'C5': 'CS',  # Chiapas
            'B5': 'BS', '85': 'BS',               # BCS
            'N1': 'NL', 'NI': 'NL',               # Nuevo León
            '5L': 'SL', 'SI': 'SL',               # Sinaloa
            '5P': 'SP',                            # San Luis Potosí
            '5R': 'SR',                            # Sonora
            'T5': 'TS',                            # Tamaulipas
            'V2': 'VZ', 'VL': 'VZ',               # Veracruz
            '2S': 'ZS',                            # Zacatecas
            '0C': 'OC',                            # Oaxaca
            'G7': 'GT', 'G1': 'GT',               # Guanajuato
        }
        estado_code = result[11] + result[12]
        if estado_code in estado_corrections:
            corrected = estado_corrections[estado_code]
            result[11] = corrected[0]
            result[12] = corrected[1]
        
        for pos in [13, 14, 15]:
            if result[pos] in letter_replacements:
                result[pos] = letter_replacements[result[pos]]
        
        # Posición 16: homoclave parte 1 - puede ser letra o dígito
        pos_16_replacements = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'Z': '2'}
        if result[16] in pos_16_replacements and result[16].isalpha():
            try:
                yy = int(result[4] + result[5])
                if yy > 25:  # Nacidos antes de 2000, debería ser dígito
                    result[16] = pos_16_replacements[result[16]]
            except ValueError:
                pass
        
        # Posición 17: homoclave parte 2 - típicamente dígito
        last_char_replacements = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1', 
            'Z': '2',
            'A': '4',  # A se confunde con 4
            'S': '8',  # S se confunde con 8
            'B': '8',
        }
        if result[17] in last_char_replacements:
            result[17] = last_char_replacements[result[17]]
        
        return ''.join(result)
    
    def extract_clave_elector(self, text: str) -> str:
        """
        Extrae la clave de elector
        Clave de elector: 18 caracteres alfanuméricos
        """
        pattern_strict = r'[A-Z]{6}\d{8}[A-Z]\d{3}'
        
        match = re.search(pattern_strict, text)
        if match:
            return match.group()
        
        clave_section = self._find_section(text, 'CLAVE DE ELECTOR', window=100)
        if not clave_section:
            clave_section = self._find_section(text, 'CLAVE', window=100)
        
        candidates = []
        
        if clave_section:
            pattern_flex = r'[A-Z0-9]{18}'
            matches = re.findall(pattern_flex, re.sub(r'[^A-Z0-9]', '', clave_section))
            
            for m in matches:
                score = 0
                
                score += sum(c.isalpha() for c in m[:6])
                
                score += sum(c.isdigit() for c in m[6:14])
                
                if m[14].isalpha():
                    score += 2
                
                score += sum(c.isdigit() for c in m[15:18])
                
                candidates.append((m, score))
        
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            return self._correct_clave_elector(best[0])
        
        match = re.search(r'CLAVE\s*(?:DE\s*)?ELECTOR[:\s]*([A-Z0-9]{18})', text)
        if match:
            return self._correct_clave_elector(match.group(1))
        
        return ""
    
    def _correct_clave_elector(self, clave: str) -> str:
        """
        Corrige una clave de elector aplicando reglas específicas por posición
        """
        if len(clave) != 18:
            return clave
        
        result = list(clave)
        
        # Posiciones donde se esperan dígitos: 6-13 y 15-17
        digit_positions = list(range(6, 14)) + list(range(15, 18))
        
        replacements = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'}
        
        for pos in digit_positions:
            if pos < len(result) and result[pos] in replacements:
                result[pos] = replacements[result[pos]]
        
        return ''.join(result)
    
    def extract_nombre_domicilio(self, text: str) -> Tuple[str, str]:
        """
        Extrae nombre y domicilio del texto OCR usando múltiples estrategias
        """
        nombre = ""
        domicilio = ""
        
        lines = text.split('\n')
        
        
        nombre_idx = -1
        domicilio_idx = -1
        clave_idx = -1
        curp_idx = -1
        
        for i, line in enumerate(lines):
            line_upper = line.upper()
            if 'NOMBRE' in line_upper and nombre_idx == -1:
                nombre_idx = i
            if 'DOMICILIO' in line_upper and domicilio_idx == -1:
                domicilio_idx = i
            if 'CLAVE' in line_upper and 'ELECTOR' in line_upper and clave_idx == -1:
                clave_idx = i
            if 'CURP' in line_upper and curp_idx == -1:
                curp_idx = i
        
        nombre_lines = []
        if nombre_idx != -1:
            end_idx = domicilio_idx if domicilio_idx > nombre_idx else (clave_idx if clave_idx > nombre_idx else curp_idx)
            if end_idx == -1:
                end_idx = min(nombre_idx + 6, len(lines))
            
            for i in range(nombre_idx, min(end_idx, len(lines))):
                line = lines[i]
                line = re.sub(r'\bNOMBRE\b', '', line, flags=re.IGNORECASE).strip()
                
                if line and self._looks_like_name(line):
                    nombre_lines.append(line)
        
        if not nombre_lines:
            nombre_lines = self._find_name_by_pattern(lines)
        
        domicilio_lines = []
        if domicilio_idx != -1:
            end_idx = clave_idx if clave_idx > domicilio_idx else curp_idx
            if end_idx == -1:
                end_idx = min(domicilio_idx + 5, len(lines))
            
            for i in range(domicilio_idx, min(end_idx, len(lines))):
                line = lines[i]
                line = re.sub(r'\bDOMICILIO\b', '', line, flags=re.IGNORECASE).strip()
                
                if line and self._looks_like_address(line):
                    domicilio_lines.append(line)
        
        if not domicilio_lines:
            domicilio_lines = self._find_address_by_pattern(lines)
        
        nombre = ' '.join(nombre_lines)
        nombre = self._clean_nombre(nombre)
        
        domicilio = ' '.join(domicilio_lines)
        domicilio = self._clean_domicilio(domicilio)
        
        return nombre, domicilio
    
    def _looks_like_name(self, text: str) -> bool:
        """Determina si el texto parece ser un nombre"""
        # Primero limpiar el texto de ruido OCR común
        # Eliminar caracteres de ruido pero mantener letras y espacios
        cleaned = re.sub(r'[^A-ZÁÉÍÓÚÑÜ\s]', ' ', text.upper())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if not cleaned or len(cleaned) < 3:
            return False
        
        # Extraer solo palabras válidas (al menos 3 letras)
        words = [w for w in cleaned.split() if len(w) >= 3 and w.isalpha()]
        
        if not words:
            return False
        
        # Calcular ratio basado en el texto limpio, no el original
        valid_chars = sum(len(w) for w in words)
        if valid_chars < 3:
            return False
        
        keywords_to_avoid = {'CURP', 'CLAVE', 'ELECTOR', 'FECHA', 'NACIMIENTO', 
                           'SEXO', 'SECCION', 'SECCIÓN', 'VIGENCIA', 'ESTADO', 'MUNICIPIO',
                           'REGISTRO', 'ELECTORAL', 'INSTITUTO', 'NACIONAL', 'CREDENCIAL',
                           'VOTAR', 'MEXICO', 'DOMICILIO', 'NOMBRE', 'ANO', 'AÑO',
                           'IDENACIMIENTO', 'LOCALIDAD', 'EMISION', 'EMISIÓN', 'PARA'}
        
        # Verificar si alguna palabra válida es una keyword a evitar
        for word in words:
            if word in keywords_to_avoid:
                return False
        
        return True
    
    def _looks_like_address(self, text: str) -> bool:
        """
        Determina si el texto parece ser una dirección
        """
        text_upper = text.upper()
        
        text_normalized = text_upper.replace("'", ".").replace("`", ".").replace("'", ".")
        
        address_keywords = ['CALLE', 'AV', 'AVE', 'AVENIDA', 'COL', 'COLONIA', 
                          'NUM', 'NO.', 'MZ', 'LT', 'FRACC', 'C.P.', 'C P',
                          'PRIV', 'PRIVADA', 'BLVD', 'CARR', 'LOC', 'CAMP', 'SIN',
                          'CABOS', 'TUXPAN', 'CALKINI']  # Añadir ciudades comunes
        
        for kw in address_keywords:
            if kw in text_upper or kw in text_normalized:
                return True
        
        if re.search(r'\d', text):
            return True
        
        # Parece ser nombre de estado o ciudad (terminaciones comunes)
        # Usar texto normalizado para detectar B.C.S., etc.
        state_patterns = [
            r',\s*[A-Z]{2,5}\.?$',  # , ESTADO.
            r'CAMP\.?$', r'SON\.?$', r'VER\.?$', 
            r'B\.?C\.?S\.?$', r'B\.C\.S\.?$',  # Baja California Sur
            r'CDMX\.?$', r'JAL\.?$', r'N\.?L\.?$',
            r'GTO\.?$', r'HGO\.?$', r'MEX\.?$',
            r'[A-Z]{2,5},?\s*$'  # Cualquier abreviatura al final
        ]
        for pattern in state_patterns:
            if re.search(pattern, text_normalized):
                return True
        
        return False
    
    def _find_name_by_pattern(self, lines: List[str]) -> List[str]:
        """
        Busca el nombre usando patrones heurísticos
        """
        results = []
        
        keywords_exclude = {'INSTITUTO', 'NACIONAL', 'ELECTORAL', 'CREDENCIAL', 'VOTAR', 
                          'MEXICO', 'CURP', 'CLAVE', 'ELECTOR', 'FECHA', 'NACIMIENTO',
                          'SEXO', 'SECCION', 'SECCIÓN', 'VIGENCIA', 'REGISTRO', 'DOMICILIO',
                          'NOMBRE', 'ESTADO', 'MUNICIPIO', 'LOCALIDAD', 'ANO', 'AÑO',
                          'IDENACIMIENTO', 'EMISION', 'EMISIÓN', 'PARA', 'TATA', 'EEE',
                          'TEE', 'SAUUAAS', 'CECOTADAL', 'SNM', 'EES', 'PT', 'ANODE'}
        
        name_candidates = []
        
        for i, line in enumerate(lines):
            cleaned = re.sub(r'[^A-ZÁÉÍÓÚÑÜ\s]', '', line.upper()).strip()
            
            if len(cleaned) < 4:
                continue
            
            words = cleaned.split()
            if not words:
                continue
            
            # Filtrar palabras inválidas (keywords o muy cortas)
            valid_words = [w for w in words if w not in keywords_exclude and len(w) >= 3 and w.isalpha()]
            
            if not valid_words:
                continue
            
            # Nombre típico: palabras de 3-15 caracteres
            is_name_like = all(3 <= len(w) <= 15 for w in valid_words)
            
            if is_name_like:
                proximity_bonus = 0
                for j in range(max(0, i-3), min(len(lines), i+3)):
                    line_check = lines[j].upper()
                    if 'SEXO' in line_check:
                        proximity_bonus = 15
                        break
                
                score = len(' '.join(valid_words)) + proximity_bonus
                name_candidates.append((i, ' '.join(valid_words), score))
        
        # Ordenar por score y tomar los mejores candidatos consecutivos
        if name_candidates:
            name_candidates.sort(key=lambda x: x[2], reverse=True)
            
            for _, text, _ in name_candidates[:3]:
                if text not in results:
                    results.append(text)
        
        return results
    
    def _find_address_by_pattern(self, lines: List[str]) -> List[str]:
        """
        Busca la dirección usando patrones heurísticos
        """
        results = []
        
        for line in lines:
            if self._looks_like_address(line):
                cleaned = re.sub(r'[^A-ZÁÉÍÓÚÑÜ0-9\s\.,#\-/]', '', line.upper())
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                
                if len(cleaned) >= 5:
                    results.append(cleaned)
        
        return results[:3]  # Máximo 3 líneas de dirección
    
    def _clean_nombre(self, nombre: str) -> str:
        """
        Limpia el nombre aplicando heurística
        """
        nombre = re.sub(r'[^A-ZÁÉÍÓÚÑÜ\s]', '', nombre.upper())
        
        nombre = re.sub(r'\s+', ' ', nombre).strip()
        
        # Palabras de ruido común del OCR que no son nombres
        ruido_ocr = {
            'EXE', 'RTE', 'EAE', 'OM', 'AM', 'SS', 'ER', 'ES', 'ON', 'BUE', 'ALR',
            'PAGE', 'UP', 'DOWN', 'CE', 'LE', 'AF', 'ND', 'IO', 'II', 'OI',
            'CLAVEDEELECTOR', 'ANODEREGISTRO', 'FECHADENACIMIENTO', 'IEXICO',
            'ORSTJSH', 'ORSTJS', 'MNPRJN', 'SRGRJR', 'BTRDMR', 'SAGJ',
            'ERRY', 'SANA', 'OSCABOS', 'WAAR', 'CAG', 'APA', 'PIE', 'VIK',
            'HASS', 'LATION', 'FATH', 'SNG', 'ORACH', 'EAD', 'JAEN', 'HIA',
            'URYELE', 'AOE', 'GF', 'EY', 'TTT', 'WWW', 'REN', 'AOF', 'IM',
            'SEERS', 'NITE', 'PARE', 'VIERO', 'SEES', 'BEAR', 'CAN',
            'SATHALRALL', 'YATHAIMALEKA', 'IEKA', 'THUN', 'BAAS', 'ORE'
        }
        
        # Eliminar tokens muy cortos que no son conectores, y palabras de ruido
        words = nombre.split()
        cleaned_words = []
        
        for word in words:
            if len(word) <= 2 and word not in TextCleaner.CONECTORES:
                continue
            if word in ruido_ocr:
                continue
            # Filtrar palabras que parecen códigos (solo consonantes o patrones raros)
            if len(word) >= 4 and not any(c in 'AEIOU' for c in word):
                continue
            cleaned_words.append(word)
        
        nombre = ' '.join(cleaned_words)
        nombre = TextCleaner.split_concatenated_words(nombre)
        
        return nombre
    
    def _clean_domicilio(self, domicilio: str) -> str:
        """
        Limpia el domicilio aplicando heurística
        """
        domicilio = re.sub(r'[^A-ZÁÉÍÓÚÑÜ0-9\s\.,#\-/]', '', domicilio.upper())
        
        domicilio = re.sub(r'\s+', ' ', domicilio).strip()
        
        return domicilio
    
    def extract_fecha_nacimiento(self, text: str, curp: str = "") -> str:
        """
        Extrae la fecha de nacimiento con validación lógica y corrección de errores OCR
        """
        def validate_and_fix_date(date_str: str) -> str:
            """Valida y corrige una fecha, retorna string vacío si no es válida"""
            parts = re.split(r'[/\-]', date_str)
            if len(parts) != 3:
                return ""
            
            day_str, month_str, year_str = parts
            
            try:
                day = int(day_str)
                month = int(month_str)
                year = int(year_str)
            except ValueError:
                return ""
            
            if day > 31:
                # 49 → 19, 41 → 11, 42 → 12, etc. (4 confundido con 1)
                if day_str[0] == '4':
                    day = int('1' + day_str[1])
                elif day_str[0] == '9' and day > 31:
                    day = int('0' + day_str[1])
                elif day_str[0] == '7' and day > 31:
                    day = int('1' + day_str[1])
            
            if day > 31:
                if day_str[1] in ['8', '9']:
                    day = int(day_str[0] + '0')
            
            if month > 12:
                if month_str[0] == '1' and month > 12:
                    if month_str[1] in ['4', '9']:
                        month = 10 + (1 if month_str[1] == '4' else 0)
                elif month_str[0] == '4':
                    month = int('0' + month_str[1])
                elif month_str[0] == '9':
                    month = int('0' + month_str[1])
            
            if day < 1 or day > 31:
                return ""
            if month < 1 or month > 12:
                return ""
            if year < 1900 or year > 2025:
                return ""
            
            days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if day > days_in_month[month - 1]:
                if month <= 31 and day <= 12:
                    day, month = month, day
                else:
                    return ""
            
            return f"{day:02d}/{month:02d}/{year}"
        
        patterns = [
            r'FECHA\s*(?:DE\s*)?NACIMIENTO[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})',
            r'FECHA\s*DE\s*NACIMIENTO[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})',
            r'FECHADE\s*NACIMIENTO[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})',  # Sin espacio
            r'FECH[A4]\s*(?:DE\s*)?N[A4]CIMIENTO[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})',  # A confundida con 4
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fecha = validate_and_fix_date(match.group(1))
                if fecha:
                    return fecha
        
        fecha_matches = re.findall(r'(\d{2}[/\-]\d{2}[/\-]\d{4})', text)
        for fecha_str in fecha_matches:
            fecha = validate_and_fix_date(fecha_str)
            if fecha:
                year = int(fecha.split('/')[-1])
                if 1930 <= year <= 2010:
                    return fecha
        
        # Fallback: extraer del CURP (posiciones 4-9 son YYMMDD)
        if curp and len(curp) >= 10:
            try:
                yy = curp[4:6]
                mm = curp[6:8]
                dd = curp[8:10]
                
                # Determinar siglo (asumimos 1900s si YY > 25, 2000s si YY <= 25)
                year = int(yy)
                if year > 25:
                    year += 1900
                else:
                    year += 2000
                
                fecha_from_curp = f"{dd}/{mm}/{year}"
                fecha = validate_and_fix_date(fecha_from_curp)
                if fecha:
                    return fecha
            except (ValueError, IndexError):
                pass
        
        curp_match = re.search(r'[A-Z]{4}(\d{2})(\d{2})(\d{2})[HM][A-Z]{5}', text.upper())
        if curp_match:
            yy, mm, dd = curp_match.groups()
            year = int(yy)
            if year > 25:
                year += 1900
            else:
                year += 2000
            fecha = validate_and_fix_date(f"{dd}/{mm}/{year}")
            if fecha:
                return fecha
        
        return ""
    
    def extract_sexo(self, text: str, curp: str = "") -> str:
        """Extrae el sexo (H o M)"""
        text_upper = text.upper()
        
        patterns = [
            r'SEXO\s*[:\s]?\s*([HMF])\b',
            r'SEXO\s+([HM])\b',
            r':\s*SEXO\s+([HM])\b',
            r'SEXO\s*([HM])\s',
            r'SEXO\s*[:\s]*([HM])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                sexo = match.group(1).upper()
                return 'M' if sexo == 'F' else sexo
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_upper = line.upper()
            if 'SEXO' in line_upper:
                sexo_pos = line_upper.find('SEXO')
                after_sexo = line_upper[sexo_pos+4:].strip()
                
                for j, char in enumerate(after_sexo):
                    if char in ['H', 'M']:
                        if j == 0 or not after_sexo[j-1].isalpha():
                            if j + 1 >= len(after_sexo) or not after_sexo[j+1].isalpha():
                                return char
                
                for word in after_sexo.split():
                    if word in ['H', 'M']:
                        return word
                
                if i + 1 < len(lines):
                    next_line = lines[i+1].upper().strip()
                    if next_line in ['H', 'M']:
                        return next_line
                    if next_line and next_line[0] in ['H', 'M']:
                        if len(next_line) == 1 or not next_line[1].isalpha():
                            return next_line[0]
        
        for line in lines:
            line_upper = line.upper()
            if any(kw in line_upper for kw in ['FECHA', 'NACIMIENTO', 'SECCI']):
                words = line_upper.split()
                for word in words:
                    if word in ['H', 'M']:
                        return word
        
        # Fallback: extraer del CURP (posición 10 es el sexo: H o M)
        if curp and len(curp) >= 11:
            sexo_curp = curp[10].upper()
            if sexo_curp in ['H', 'M']:
                return sexo_curp
        
        curp_match = re.search(r'[A-Z]{4}\d{6}([HM])[A-Z]{5}', text_upper)
        if curp_match:
            return curp_match.group(1)
        
        return ""
    
    def extract_seccion_vigencia_from_line(self, text: str) -> Tuple[str, str]:
        """Extrae sección y vigencia que suelen estar en la misma línea que fecha de nacimiento"""
        seccion = ""
        vigencia = ""
        
        lines = text.split('\n')
        
        # Primero extraer vigencia usando el método mejorado
        vigencia = self.extract_vigencia(text)
        
        for i, line in enumerate(lines):
            line_upper = line.upper()
            
            if 'SECCI' in line_upper:
                match = re.search(r'SECCI[OÓ0]N\s*[:\s]*(\d{4})', line, re.IGNORECASE)
                if match:
                    seccion = match.group(1)
                    break
        
        if not seccion:
            found_header = False
            for i, line in enumerate(lines):
                line_upper = line.upper()
                
                if ('FECHA' in line_upper or 'NACIMIENTO' in line_upper) and ('SECCION' in line_upper or 'SECCI' in line_upper or 'VIGENCIA' in line_upper):
                    found_header = True
                    continue
                
                if found_header and line.strip():
                    
                    fecha_match = re.search(r'\d{2}/\d{2}/\d{4}', line)
                    if fecha_match:
                        after_fecha = line[fecha_match.end():].strip()
                        
                        all_4digits = re.findall(r'(\d{4})', after_fecha)
                        
                        if all_4digits:
                            for d in all_4digits:
                                d_int = int(d)
                                if d.startswith('0') or d_int < 2000:
                                    seccion = d
                                    break
                            
                            # tomar el primer número que no sea año de vigencia
                            if not seccion:
                                vig_years = set()
                                if vigencia:
                                    vig_years = set(re.findall(r'(\d{4})', vigencia))
                                
                                for d in all_4digits:
                                    if d not in vig_years:
                                        d_int = int(d)
                                        if d_int < 1940 or d_int > 2050:
                                            seccion = d
                                            break
                    
                    break
        
        if not seccion:
            for line in lines:
                fecha_match = re.search(r'(\d{2}/\d{2}/\d{4})', line)
                if fecha_match:
                    after_fecha = line[fecha_match.end():]
                    nums = re.findall(r'\b(\d{4})\b', after_fecha)
                    
                    vig_years = set()
                    if vigencia:
                        vig_years = set(re.findall(r'(\d{4})', vigencia))
                    
                    for num in nums:
                        num_int = int(num)
                        if num not in vig_years:
                            if num.startswith('0') or num_int < 1940 or (num_int > 2010 and num_int < 2020):
                                seccion = num
                                break
                    
                    if seccion:
                        break
        
        return seccion, vigencia
    
    def extract_field(self, text: str, field_name: str, pattern: str = r'\d+') -> str:
        """
        Extrae un campo genérico basado en su nombre
        """
        regex = rf'{field_name}[:\s]*({pattern})'
        match = re.search(regex, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""
    
    def extract_año_registro(self, text: str) -> str:
        """Extrae el año de registro"""
        # Patrones para año de registro (incluyendo errores OCR)
        patterns = [
            r'A[ÑN]O\s*DE\s*REGISTRO\s*[:\s]*(\d{4})\s*(\d{2})?',  # ANO DE REGISTRO 2013 05
            r'A[ÑN]ODE\s*REGISTRO\s*[:\s]*(\d{4})\s*(\d{2})?',     # ANODE REGISTRO (sin espacio)
            r'REGISTRO\s*[:\s]*(\d{4})\s*(\d{2})?',                 # Solo REGISTRO
            r'A[ÑN]O\s*REGISTRO\s*[:\s]*(\d{4})\s*(\d{2})?',        # ANO REGISTRO (sin DE)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                year = match.group(1)
                month = match.group(2) if match.group(2) else ""
                if 1990 <= int(year) <= 2030:
                    if month:
                        return f"{year} {month}"
                    return year
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'CURP' in line.upper() or 'REGISTRO' in line.upper():
                match = re.search(r'[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]{2}\s*(\d{4})\s*(\d{2})?', line.upper())
                if match:
                    year = match.group(1)
                    month = match.group(2) if match.group(2) else ""
                    if 1990 <= int(year) <= 2030:
                        if month:
                            return f"{year} {month}"
                        return year
        
        return ""
    
    def extract_vigencia(self, text: str) -> str:
        """Extrae el período de vigencia"""
        # Patrones para VIGENCIA con rango YYYY-YYYY (incluyendo errores OCR comunes)
        vigencia_range_patterns = [
            r'VIGENCIA[:\s]*(\d{4})\s*[-–~]\s*(\d{4})',
            r'V[I1]GENCIA[:\s]*(\d{4})\s*[-–~]\s*(\d{4})',
            r'W[I1]GENCIA[:\s]*(\d{4})\s*[-–~]\s*(\d{4})',  # W por V
            r'V[I1]CENCIA[:\s]*(\d{4})\s*[-–~]\s*(\d{4})',  # C por G
            r'W[I1]?CENCIA[:\s]*(\d{4})\s*[-–~]\s*(\d{4})', # W por V y C por G
            r'[VW][I1]?[GC]ENCIA[:\s]*(\d{4})\s*[-–~]\s*(\d{4})',  # Patrón general
        ]
        
        for pattern in vigencia_range_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)}-{match.group(2)}"
        
        # Patrón más flexible para capturar "2024 - 2034", "2024-2034", "2024 ~2034"
        flexible_patterns = [
            r'(20\d{2})\s*[-–~]\s*(20\d{2})',  # Con guión/tilde
            r'(20\d{2})\s+(20\d{2})',  # Solo espacios (dos años consecutivos)
        ]
        
        for pattern in flexible_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                year1, year2 = int(match[0]), int(match[1])
                # Validar que sea un rango válido de vigencia (8-12 años típico)
                if 8 <= year2 - year1 <= 12:
                    return f"{match[0]}-{match[1]}"
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_upper = line.upper()
            if 'VIGENCIA' in line_upper or 'VICENCIA' in line_upper or 'WICENCIA' in line_upper:
                search_text = line
                if i + 1 < len(lines):
                    search_text += ' ' + lines[i + 1]
                
                years = re.findall(r'(20\d{2})', search_text)
                if len(years) >= 2:
                    year1, year2 = int(years[0]), int(years[1])
                    if 8 <= year2 - year1 <= 12:
                        return f"{years[0]}-{years[1]}"
                elif len(years) == 1:
                    year = int(years[0])
                    if 2015 <= year <= 2035:
                        if year >= 2020:
                            return f"{year}-{year + 10}"
                        else:
                            return f"{year - 10}-{year}"
        
        for line in lines:
            if re.search(r'\d{2}/\d{2}/\d{4}', line):
                years = re.findall(r'(20\d{2})', line)
                if len(years) >= 2:
                    for i in range(len(years) - 1):
                        year1, year2 = int(years[i]), int(years[i+1])
                        if 8 <= year2 - year1 <= 12:
                            return f"{years[i]}-{years[i+1]}"
                elif len(years) == 1:
                    year = int(years[0])
                    if 2020 <= year <= 2035:
                        return f"{year}-{year + 10}"
        
        single_year_patterns = [
            r'VIGENCIA\s*(\d{4})',
            r'V[I1]GENCIA\s*(\d{4})',
            r'W[I1]GENCIA\s*(\d{4})',
            r'V[I1]CENCIA\s*(\d{4})',
            r'W[I1]?CENCIA\s*(\d{4})',
            r'[VW][I1]?[GC]ENCIA\s*(\d{4})',
        ]
        
        for pattern in single_year_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                if 2015 <= year <= 2050:
                    start_year = year - 10
                    return f"{start_year}-{year}"
        
        return ""
    
    def extract_emision(self, text: str, vigencia: str) -> str:
        """
        Extrae el año de emisión
        Si no está mencionado directamente, es el primer año de vigencia
        """
        match = re.search(r'EMISI[OÓ]N[:\s]*(\d{4})', text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        if vigencia:
            match = re.search(r'(\d{4})', vigencia)
            if match:
                return match.group(1)
        
        return ""
    
    def extract_estado_municipio(self, domicilio: str, full_text: str = "") -> Tuple[str, str]:
        """Extrae el estado (abreviatura) y municipio del domicilio"""
        estados_todos = {**ESTADOS_ABREV, **{v: ESTADOS_ABREV.get(v, v) for v in ESTADOS_VARIANTES}}
        
        estado = ""
        municipio = ""
        
        domicilio_normalized = domicilio.replace("'", ".").replace("`", ".").replace("'", ".")
        domicilio_normalized = re.sub(r'([A-Z]):', r'\1.', domicilio_normalized)
        domicilio_upper = domicilio_normalized.upper().strip()
        
        for ruido in ['EL', 'LA', 'ET', 'CE', 'LE', 'SE', 'DE', 'EA', 'AE', 'ND', 'SS', 'AF', 'IO']:
            domicilio_upper = re.sub(rf'\s+{ruido}\s*$', '', domicilio_upper)
        
        for abrev in sorted(estados_todos.keys(), key=len, reverse=True):
            # Patrón: la abreviatura al final, posiblemente con punto
            # Evitar que COL sea detectado como Colima cuando es COLONIA
            pattern = rf',?\s*{re.escape(abrev)}\.?\s*$'
            match = re.search(pattern, domicilio_upper)
            if match:
                if abrev in ['COL', 'COL.']:
                    before_match = domicilio_upper[:match.start()].strip()
                    if before_match.endswith('ONIA') or before_match.endswith(','):
                        continue
                
                estado = abrev.replace('.', '')
                
                before_state = domicilio_upper[:match.start()].strip().rstrip(',').strip()
                
                cp_match = re.search(r'\d{5}\s+(.+)$', before_state)
                if cp_match:
                    municipio = cp_match.group(1).strip().rstrip(',').strip()
                else:
                    parts = before_state.split(',')
                    if len(parts) > 1:
                        municipio = parts[-1].strip()
                    else:
                        # Tomar las últimas palabras que no sean números ni códigos postales
                        words = before_state.split()
                        mun_words = []
                        for w in reversed(words):
                            if not re.match(r'^\d+$', w) and len(w) > 1:
                                mun_words.insert(0, w)
                                if len(mun_words) >= 3:
                                    break
                            elif re.match(r'^\d+$', w):
                                break
                        municipio = ' '.join(mun_words)
                break
        
        if not estado and full_text:
            full_upper = full_text.upper()
            
            for abrev in sorted(estados_todos.keys(), key=len, reverse=True):
                if abrev in ['COL', 'COL.']:
                    continue  # Saltar COL para evitar falsos positivos
                pattern = rf'([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+),?\s*{re.escape(abrev)}\.?(?:\s|$|\n)'
                match = re.search(pattern, full_upper)
                if match:
                    estado = abrev.replace('.', '')
                    if not municipio:
                        municipio = match.group(1).strip()
                    break
        
        if municipio:
            municipio = re.sub(r'[^\w\sÁÉÍÓÚÑáéíóúñ]', '', municipio).strip()
        
        return estado, municipio
    
    def _find_section(self, text: str, keyword: str, window: int = 50) -> Optional[str]:
        """
        Encuentra una sección de texto alrededor de una palabra clave
        """
        pos = text.find(keyword)
        if pos == -1:
            return None
        
        start = max(0, pos)
        end = min(len(text), pos + len(keyword) + window)
        
        return text[start:end]
    
    def parse_nombre_components(self, nombre_completo: str) -> Tuple[str, str, str]:
        """Separa el nombre completo en apellido paterno, materno y nombre(s)"""
        nombres_comunes = {
            'JUAN', 'JOSE', 'MARIA', 'JESUS', 'MIGUEL', 'ANTONIO', 'FRANCISCO', 'PEDRO',
            'CARLOS', 'MANUEL', 'LUIS', 'JORGE', 'RAFAEL', 'FERNANDO', 'ALBERTO', 'RICARDO',
            'ALEJANDRO', 'DANIEL', 'DAVID', 'GABRIEL', 'EDUARDO', 'ROBERTO', 'JAVIER', 'ADRIAN',
            'VICTOR', 'OSCAR', 'SERGIO', 'ANDRES', 'MARTIN', 'HUGO', 'RAUL', 'JAIME',
            'ROSA', 'GUADALUPE', 'PATRICIA', 'MARTHA', 'LETICIA', 'LAURA', 'ANA', 'ELIZABETH',
            'CARMEN', 'SANDRA', 'VERONICA', 'MONICA', 'SILVIA', 'ADRIANA', 'ALEJANDRA', 'DIANA',
            'CLAUDIA', 'MARIANA', 'GABRIELA', 'LUCIA', 'ELENA', 'TERESA', 'JULIA', 'BEATRIZ',
            'JONATHAN', 'ALEXANDER', 'JAIRO', 'RICARDO', 'JESUS', 'CRISTIAN', 'CHRISTIAN',
            'DIEGO', 'IVAN', 'ANGEL', 'MARCO', 'CESAR', 'ARTURO', 'GERARDO', 'ENRIQUE',
            'LAURA', 'PAULA', 'SOFIA', 'ISABELLA', 'VALENTINA', 'CAMILA', 'VALERIA', 'NATALIA'
        }
        
        # Palabras de ruido (keywords de INE y ruido OCR común)
        noise_words = {
            'SEXO', 'SEXGAN', 'DOMICILIO', 'NOMBRE', 'CURP', 'CLAVE', 'ELECTOR', 
            'FECHA', 'NACIMIENTO', 'SECCION', 'VIGENCIA', 'REGISTRO', 'INSTITUTO',
            'NACIONAL', 'ELECTORAL', 'CREDENCIAL', 'VOTAR', 'MEXICO', 'CAMP', 'VER',
            'BCS', 'NUM', 'COL', 'EEE', 'OER', 'EES', 'OEE', 'SEES', 'PONE', 'PEER',
            'EOS', 'SNM', 'RAA', 'PT', 'IG'
        }
        
        words = nombre_completo.split()
        filtered_words = []
        
        for w in words:
            w_upper = w.upper()
            
            if len(w) < 3:
                continue
            
            if w_upper in noise_words or w_upper in ESTADOS_ABREV:
                continue
            
            # Filtrar palabras que son solo consonantes (ruido OCR)
            vowels = set('AEIOU')
            has_vowel = any(c in vowels for c in w_upper)
            if not has_vowel and len(w) > 2:
                continue
            
            # Filtrar palabras con patrones de ruido (repetición de letras)
            if re.match(r'^(.)\1{2,}$', w_upper):  # EEE, OOO, etc.
                continue
            
            # Filtrar palabras que parecen códigos (mezcla rara de letras)
            if re.match(r'^[BCDFGHJKLMNPQRSTVWXYZ]{3,}$', w_upper):  # Solo consonantes
                continue
            
            filtered_words.append(w)
        
        words = filtered_words
        
        if len(words) >= 3:
            name_indices = [i for i, w in enumerate(words) if w.upper() in nombres_comunes]
            
            if name_indices:
                # Encontrar el primer nombre común y asumir que todo antes son apellidos
                first_name_idx = min(name_indices)
                
                if first_name_idx >= 2:
                    apellido_paterno = words[0]
                    apellido_materno = words[1]
                    nombre = ' '.join(words[2:])
                elif first_name_idx == 1:
                    apellido_paterno = words[0]
                    apellido_materno = ""
                    nombre = ' '.join(words[1:])
                else:
                    non_names = [w for w in words if w.upper() not in nombres_comunes]
                    names_only = [w for w in words if w.upper() in nombres_comunes]
                    
                    if len(non_names) >= 2:
                        apellido_paterno = non_names[0]
                        apellido_materno = non_names[1]
                        nombre = ' '.join(names_only)
                    elif len(non_names) == 1:
                        apellido_paterno = non_names[0]
                        apellido_materno = ""
                        nombre = ' '.join(names_only)
                    else:
                        apellido_paterno = words[0]
                        apellido_materno = words[1] if len(words) > 1 else ""
                        nombre = ' '.join(words[2:]) if len(words) > 2 else ""
            else:
                apellido_paterno = words[0]
                apellido_materno = words[1]
                nombre = ' '.join(words[2:])
        elif len(words) == 2:
            apellido_paterno = words[0]
            apellido_materno = ""
            nombre = words[1]
        elif len(words) == 1:
            apellido_paterno = words[0]
            apellido_materno = ""
            nombre = ""
        else:
            apellido_paterno = ""
            apellido_materno = ""
            nombre = ""
        
        return apellido_paterno, apellido_materno, nombre
    
    def extract(self, image_path: str) -> INEData:
        """
        Método principal para extraer todos los datos de una imagen de INE
        Usa análisis de consenso de múltiples OCR para mejorar precisión
        """
        self.debug_data = {}
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        logger.info(f"Procesando imagen: {image_path}")
        self._log_debug('image_path', image_path)
        self._log_debug('image_size', f"{image.shape[1]}x{image.shape[0]}")
        
        text = self.extract_text(image)
        logger.debug(f"Texto extraído:\n{text}")
        
        # Construir datos de consenso de todos los métodos de preprocesamiento
        consensus = self.build_consensus_data()
        
        data = INEData()
        data.raw_text = text
        
        data.curp = self.extract_curp(text)
        data.clave_elector = self.extract_clave_elector(text)
        data.fecha_nacimiento = self.extract_fecha_nacimiento(text, data.curp)
        data.sexo = self.extract_sexo(text, data.curp)
        
        nombre_con_consenso, ap, am, nombre = self.extract_nombre_with_consensus(text, consensus)
        
        # También extraer con método original para comparar
        nombre_original, domicilio_original = self.extract_nombre_domicilio(text)
        ap_orig, am_orig, nombre_orig = self.parse_nombre_components(nombre_original)
        
        # Decidir entre consenso y original
        # Preferir consenso si:
        # 1. Tiene apellido materno y el original no
        # 2. O el original tiene palabras de ruido obvio
        ruido_obvio = {'SES', 'FAE', 'SIE', 'REON', 'LOC', 'OER', 'EXE', 'BUE', 'OW', 'OE', 
                       'SE', 'BAE', 'POETS', 'EARS', 'AAA', 'EEE', 'III', 'OOO', 'UUU'}
        
        original_tiene_ruido = any(word in ruido_obvio for word in nombre_original.upper().split())
        consenso_tiene_apellido_materno = bool(am)
        original_tiene_apellido_materno = bool(am_orig) and am_orig not in ruido_obvio
        
        usar_consenso = False
        if ap:  # Consenso tiene apellido paterno válido
            if consenso_tiene_apellido_materno and not original_tiene_apellido_materno:
                usar_consenso = True  # Consenso tiene más info útil
            elif original_tiene_ruido:
                usar_consenso = True  # Original tiene ruido
            elif len(nombre_con_consenso) >= len(nombre_original):
                usar_consenso = True  # Consenso igual o más largo
        
        if usar_consenso:
            data.nombre_completo = nombre_con_consenso
            data.apellido_paterno = ap
            data.apellido_materno = am
            data.nombre = nombre
        else:
            data.nombre_completo = nombre_original
            data.apellido_paterno = ap_orig
            data.apellido_materno = am_orig
            data.nombre = nombre_orig
        
        # Limpiar nombre usando CURP como referencia
        if data.curp:
            data.nombre_completo, data.apellido_paterno, data.apellido_materno, data.nombre = \
                clean_nombre_with_curp(data.nombre_completo, data.apellido_paterno, 
                                       data.apellido_materno, data.nombre, data.curp, data.raw_text)
        
        data.domicilio = domicilio_original
        
        seccion_extracted, vigencia_extracted = self.extract_seccion_vigencia_from_line(text)
        
        data.año_registro = self.extract_año_registro(text)
        
        data.seccion = seccion_extracted if seccion_extracted else self.extract_field(text, 'SECCI[OÓ]N', r'\d{4}')
        
        data.vigencia = vigencia_extracted if vigencia_extracted else self.extract_vigencia(text)
        
        data.emision = self.extract_emision(text, data.vigencia)
        
        data.estado, _ = self.extract_estado_municipio(data.domicilio, text)
        
        if not data.estado:
            data.estado = self.extract_field(text, 'ESTAD[O0]', r'\d+')
        
        data.municipio = self.extract_municipio_with_consensus(data.domicilio, consensus, data.estado)
        
        if not data.municipio:
            _, municipio_from_dom = self.extract_estado_municipio(data.domicilio, text)
            mun_code = self.extract_field(text, 'MUNICIPIO', r'\d+')
            
            if municipio_from_dom and not municipio_from_dom.isdigit():
                data.municipio = municipio_from_dom
            elif mun_code:
                data.municipio = mun_code
        
        if data.municipio:
            data.municipio = re.sub(r'[\n\r]+', ' ', data.municipio)
            data.municipio = re.sub(r'\s+', ' ', data.municipio).strip()
            
            ruido_prefijos = ['IO', 'AF', 'SS', 'A', 'AP', 'HASS', 'OI', 'ND']
            words = data.municipio.split()
            while words and words[0] in ruido_prefijos:
                words.pop(0)
            data.municipio = ' '.join(words)
            
        data.localidad = self.extract_field(text, 'LOCALIDAD', r'\d+')
        
        logger.info(f"Extracción completada para: {image_path}")
        
        if self.debug:
            self.debug_data['extraction_result'] = asdict(data)
            if self.debug_dir:
                report_path = os.path.join(self.debug_dir, 'debug_report.html')
                self.generate_debug_report(report_path)
        
        return data
    
    def extract_to_dict(self, image_path: str) -> Dict:
        """
        Extrae datos y los retorna como diccionario
        """
        data = self.extract(image_path)
        result = asdict(data)
        del result['raw_text']
        return result
    
    def extract_to_json(self, image_path: str, include_raw: bool = False) -> str:
        """
        Extrae datos y los retorna como JSON
        """
        data = self.extract(image_path)
        result = asdict(data)
        if not include_raw:
            del result['raw_text']
        return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================
# MAPEO DE CÓDIGOS RENAPO A ESTADOS POSTALES
# ============================================================
RENAPO_A_ESTADO = {
    'AS': ['AGS'], 'BC': ['BC'], 'BS': ['BCS'], 'CC': ['CAMP', 'CAM'],
    'CL': ['COAH'], 'CM': ['COL'], 'CS': ['CHIS'], 'CH': ['CHIH'],
    'DF': ['CDMX', 'DF'], 'DG': ['DGO'], 'GT': ['GTO'], 'GR': ['GRO'],
    'HG': ['HGO'], 'JC': ['JAL'], 'MC': ['MICH'], 'MN': ['MOR'],
    'MS': ['MEX', 'EDOMEX'], 'NT': ['NAY'], 'NL': ['NL'], 'OC': ['OAX'],
    'PL': ['PUE'], 'QT': ['QRO'], 'QR': ['QROO'], 'SP': ['SLP'],
    'SL': ['SIN'], 'SR': ['SON'], 'TC': ['TAB'], 'TS': ['TAMPS', 'TAM'],
    'TL': ['TLAX'], 'VZ': ['VER'], 'YN': ['YUC'], 'ZS': ['ZAC'], 'NE': ['NE']
}

ESTADO_A_RENAPO = {}
for renapo, estados in RENAPO_A_ESTADO.items():
    for estado in estados:
        ESTADO_A_RENAPO[estado] = renapo


def cross_validate_curp(data: dict) -> dict:
    """
    Valida la CURP contra los otros campos extraídos.
    Retorna un diccionario con los resultados de validación.
    """
    results = {
        'valid': True,
        'score': 100,
        'checks': {},
        'warnings': [],
        'errors': []
    }
    
    curp = data.get('curp', '').upper().strip()
    if len(curp) < 16:
        results['valid'] = False
        results['errors'].append('CURP muy corta o no encontrada')
        results['score'] = 0
        return results
    
    # Extraer componentes de la CURP
    curp_iniciales = curp[0:4]      # AAPR
    curp_fecha = curp[4:10]         # 630321 (AAMMDD)
    curp_sexo = curp[10]            # H o M
    curp_estado = curp[11:13]       # DF
    curp_consonantes = curp[13:16]  # LRC
    
    # 1. VALIDAR FECHA DE NACIMIENTO
    fecha = data.get('fecha_nacimiento', '')
    if fecha:
        # Formato esperado: DD/MM/AAAA o DD/MM/AA
        parts = fecha.replace('-', '/').split('/')
        if len(parts) == 3:
            dd, mm, aa = parts[0].zfill(2), parts[1].zfill(2), parts[2]
            aa_short = aa[-2:] if len(aa) == 4 else aa
            fecha_curp_esperada = f"{aa_short}{mm}{dd}"
            
            if fecha_curp_esperada == curp_fecha:
                results['checks']['fecha'] = {'status': 'OK', 'detail': f'CURP:{curp_fecha} = Fecha:{fecha}'}
            else:
                results['checks']['fecha'] = {'status': 'ERROR', 'detail': f'CURP:{curp_fecha} ≠ Fecha:{fecha} (esperado:{fecha_curp_esperada})'}
                results['errors'].append(f'Fecha no coincide: CURP tiene {curp_fecha}, extraído {fecha_curp_esperada}')
                results['score'] -= 25
    else:
        results['checks']['fecha'] = {'status': 'SKIP', 'detail': 'Fecha no extraída'}
        results['warnings'].append('No se pudo validar fecha (no extraída)')
    
    # 2. VALIDAR SEXO
    sexo = data.get('sexo', '').upper().strip()
    if sexo:
        if sexo == curp_sexo:
            results['checks']['sexo'] = {'status': 'OK', 'detail': f'CURP:{curp_sexo} = Sexo:{sexo}'}
        else:
            results['checks']['sexo'] = {'status': 'ERROR', 'detail': f'CURP:{curp_sexo} ≠ Sexo:{sexo}'}
            results['errors'].append(f'Sexo no coincide: CURP tiene {curp_sexo}, extraído {sexo}')
            results['score'] -= 20
    else:
        results['checks']['sexo'] = {'status': 'SKIP', 'detail': 'Sexo no extraído'}
    
    # 3. VALIDAR ESTADO
    estado = data.get('estado', '').upper().strip()
    if estado:
        estado_renapo_esperado = ESTADO_A_RENAPO.get(estado, estado)
        if curp_estado == estado_renapo_esperado:
            results['checks']['estado'] = {'status': 'OK', 'detail': f'CURP:{curp_estado} = Estado:{estado} ({estado_renapo_esperado})'}
        elif curp_estado in ENTIDADES_RENAPO:
            # El estado en CURP es válido pero no coincide con el extraído
            # Esto puede ser normal: la persona nació en un estado pero vive en otro
            results['checks']['estado'] = {'status': 'WARN', 'detail': f'CURP:{curp_estado} ≠ Estado INE:{estado} (puede ser diferente lugar de residencia vs nacimiento)'}
            results['warnings'].append(f'Estado de nacimiento (CURP: {curp_estado}) difiere del estado en INE ({estado})')
        else:
            results['checks']['estado'] = {'status': 'ERROR', 'detail': f'CURP:{curp_estado} no es código RENAPO válido'}
            results['errors'].append(f'Código de estado en CURP inválido: {curp_estado}')
            results['score'] -= 15
    else:
        results['checks']['estado'] = {'status': 'SKIP', 'detail': 'Estado no extraído'}
    
    # 4. VALIDAR INICIALES DEL NOMBRE
    nombre = data.get('nombre', '').upper().strip()
    ap_paterno = data.get('apellido_paterno', '').upper().strip()
    ap_materno = data.get('apellido_materno', '').upper().strip()
    
    if ap_paterno and ap_materno and nombre:
        # Primera letra del apellido paterno
        inicial_ap1 = ap_paterno[0] if ap_paterno else ''
        
        # Primera vocal interna del apellido paterno (después de la primera letra)
        vocal_ap1 = ''
        for c in ap_paterno[1:]:
            if c in 'AEIOU':
                vocal_ap1 = c
                break
        
        # Primera letra del apellido materno
        inicial_ap2 = ap_materno[0] if ap_materno else ''
        
        # Primera letra del nombre
        inicial_nombre = nombre.split()[0][0] if nombre else ''
        
        iniciales_esperadas = f"{inicial_ap1}{vocal_ap1}{inicial_ap2}{inicial_nombre}"
        
        if iniciales_esperadas == curp_iniciales:
            results['checks']['iniciales'] = {'status': 'OK', 'detail': f'CURP:{curp_iniciales} = Calculado:{iniciales_esperadas}'}
        else:
            # Verificar coincidencia parcial (al menos 3 de 4)
            matches = sum(1 for a, b in zip(curp_iniciales, iniciales_esperadas) if a == b)
            if matches >= 3:
                results['checks']['iniciales'] = {'status': 'WARN', 'detail': f'CURP:{curp_iniciales} ≈ Calculado:{iniciales_esperadas} ({matches}/4 coinciden)'}
                results['warnings'].append(f'Iniciales parcialmente coinciden: CURP={curp_iniciales}, esperado={iniciales_esperadas}')
                results['score'] -= 10
            else:
                results['checks']['iniciales'] = {'status': 'ERROR', 'detail': f'CURP:{curp_iniciales} ≠ Calculado:{iniciales_esperadas}'}
                results['errors'].append(f'Iniciales no coinciden: CURP={curp_iniciales}, esperado={iniciales_esperadas}')
                results['score'] -= 20
    else:
        results['checks']['iniciales'] = {'status': 'SKIP', 'detail': 'Nombre incompleto para validar iniciales'}
    
    # 5. VALIDAR CONSONANTES INTERNAS
    if ap_paterno and ap_materno and nombre:
        # Primera consonante interna del apellido paterno
        cons_ap1 = ''
        for c in ap_paterno[1:]:
            if c in 'BCDFGHJKLMNÑPQRSTVWXYZ':
                cons_ap1 = c
                break
        
        # Primera consonante interna del apellido materno
        cons_ap2 = ''
        for c in ap_materno[1:]:
            if c in 'BCDFGHJKLMNÑPQRSTVWXYZ':
                cons_ap2 = c
                break
        
        # Primera consonante interna del nombre
        primer_nombre = nombre.split()[0] if nombre else ''
        cons_nombre = ''
        for c in primer_nombre[1:]:
            if c in 'BCDFGHJKLMNÑPQRSTVWXYZ':
                cons_nombre = c
                break
        
        consonantes_esperadas = f"{cons_ap1}{cons_ap2}{cons_nombre}"
        
        if consonantes_esperadas == curp_consonantes:
            results['checks']['consonantes'] = {'status': 'OK', 'detail': f'CURP:{curp_consonantes} = Calculado:{consonantes_esperadas}'}
        else:
            matches = sum(1 for a, b in zip(curp_consonantes, consonantes_esperadas) if a == b)
            if matches >= 2:
                results['checks']['consonantes'] = {'status': 'WARN', 'detail': f'CURP:{curp_consonantes} ≈ Calculado:{consonantes_esperadas} ({matches}/3 coinciden)'}
                results['warnings'].append(f'Consonantes parcialmente coinciden')
                results['score'] -= 5
            else:
                results['checks']['consonantes'] = {'status': 'ERROR', 'detail': f'CURP:{curp_consonantes} ≠ Calculado:{consonantes_esperadas}'}
                results['errors'].append(f'Consonantes no coinciden: CURP={curp_consonantes}, esperado={consonantes_esperadas}')
                results['score'] -= 10
    else:
        results['checks']['consonantes'] = {'status': 'SKIP', 'detail': 'Nombre incompleto'}
    
    # Determinar validez final
    results['score'] = max(0, results['score'])
    results['valid'] = len(results['errors']) == 0 and results['score'] >= 70
    
    return results


def clean_nombre_with_curp(nombre_completo: str, apellido_paterno: str, apellido_materno: str, 
                           nombre: str, curp: str, raw_text: str = "") -> tuple:
    """
    Usa la CURP para validar y corregir los componentes del nombre.
    Elimina ruido OCR que no coincide con las iniciales esperadas.
    Si falta el nombre o apellidos, intenta buscarlos en el texto crudo.
    """
    if not curp or len(curp) < 4:
        return nombre_completo, apellido_paterno, apellido_materno, nombre
    
    import re
    curp = curp.upper()
    inicial_ap1 = curp[0]   # Primera letra apellido paterno
    inicial_ap2 = curp[2]   # Primera letra apellido materno
    inicial_nombre = curp[3]  # Primera letra del nombre de pila
    
    # Conjunto de ruido OCR común
    ruido = {'SIE', 'REON', 'LOC', 'COL', 'NUM', 'AV', 'CALLE', 'DOM', 'EXE', 'OER', 
             'BUE', 'PAGE', 'DOWN', 'POETS', 'EARS', 'BAE', 'OW', 'OE', 'SE', 'SER',
             'GHAMIZAL', 'CHAMIZAL', 'ASUNCION', 'CP', 'MZ', 'LT', 'INT', 'EXT',
             'EEE', 'AAA', 'OOO', 'III', 'UUU', 'FAE', 'SES', 'POA', 'GON', 'POMIEIIO',
             'SEXO', 'SEXOM', 'SEXOH', 'NOMBRE', 'DOMICILIO', 'FECHA', 'NACIMIENTO',
             'INSTITUTO', 'NACIONAL', 'ELECTORAL', 'CREDENCIAL', 'VOTAR', 'MEXICO',
             'CLAVE', 'ELECTOR', 'CURP', 'ESTADO', 'MUNICIPIO', 'SECCION', 'VIGENCIA'}
    
    # Nombres comunes
    nombres_comunes = {
        'ADRIAN', 'ALBERTO', 'ALEJANDRO', 'ANTONIO', 'ARTURO', 'ANDRES', 'ANGEL',
        'BRENDA', 'BEATRIZ', 'BARBARA',
        'CARLOS', 'CESAR', 'CLAUDIA', 'CRISTINA', 'CRISTIAN', 'CHRISTIAN',
        'DANIEL', 'DAVID', 'DIANA', 'DIEGO',
        'EDGAR', 'EDUARDO', 'ELENA', 'ELIZABETH', 'ENRIQUE', 'ERNESTO',
        'FERNANDO', 'FRANCISCO', 'FABIAN', 'FERNANDA', 'FELIPE',
        'GABRIEL', 'GUILLERMO', 'GUADALUPE', 'GABRIELA', 'GERARDO',
        'HECTOR', 'HUGO', 'HERIBERTO',
        'ISMAEL', 'IVAN', 'IRENE', 'ISABEL',
        'JAIRO', 'JAVIER', 'JESUS', 'JORGE', 'JOSE', 'JUAN', 'JULIO', 'JUANA',
        'LAURA', 'LETICIA', 'LUCIA', 'LUIS', 'LIZETH',
        'MANUEL', 'MARCO', 'MARIA', 'MARTIN', 'MIGUEL', 'MONICA',
        'OSCAR', 'OMAR',
        'PABLO', 'PATRICIA', 'PAOLA', 'PEDRO',
        'RAFAEL', 'RAMON', 'RAUL', 'RICARDO', 'ROBERTO', 'RODOLFO', 'ROSA', 'RUBEN',
        'SALVADOR', 'SANDRA', 'SERGIO', 'SILVIA', 'SOFIA',
        'TERESA', 'TOMAS',
        'VALERIA', 'VERONICA', 'VICTOR', 'VIRGINIA'
    }
    
    # Apellidos comunes
    apellidos_comunes = {
        'AGUILAR', 'ALVAREZ', 'AYALA',
        'BAUTISTA', 'BENITEZ',
        'CABRERA', 'CAMPOS', 'CASTILLO', 'CASTRO', 'CHAVEZ', 'CONTRERAS', 'CRUZ',
        'DIAZ', 'DOMINGUEZ',
        'ESPINOSA', 'ESPINOZA', 'ESTRADA',
        'FERNANDEZ', 'FLORES',
        'GARCIA', 'GAYTAN', 'GOMEZ', 'GONZALEZ', 'GUERRERO', 'GUTIERREZ',
        'HERNANDEZ', 'HERRERA',
        'IBARRA',
        'JASSO', 'JIMENEZ', 'JUAREZ',
        'LEON', 'LOPEZ', 'LUNA',
        'MACIAS', 'MARTINEZ', 'MEDINA', 'MEJIA', 'MENDEZ', 'MENDOZA', 'MORALES', 'MORENO', 'MUNOZ',
        'NAVARRO', 'NUNEZ',
        'OCHOA', 'ORNELAS', 'ORTEGA', 'ORTIZ',
        'PADILLA', 'PAZ', 'PEREZ', 'PINEDA',
        'RAMIREZ', 'RAMOS', 'REYES', 'RIOS', 'RIVERA', 'RODRIGUEZ', 'ROJAS', 'ROMERO', 'RUIZ',
        'SALAZAR', 'SALVADOR', 'SANCHEZ', 'SANDOVAL', 'SANTIAGO', 'SARMIENTO', 'SILVA', 'SOTO', 'SUAREZ',
        'TORRES',
        'VALDEZ', 'VALENCIA', 'VARGAS', 'VASQUEZ', 'VAZQUEZ', 'VEGA', 'VELAZQUEZ', 'VILLA', 'VILLANUEVA',
        'ZAMORA', 'ZAVALA', 'ZUNIGA'
    }
    
    ap1 = apellido_paterno.upper().strip() if apellido_paterno else ""
    ap2 = apellido_materno.upper().strip() if apellido_materno else ""
    nombre_clean = nombre.upper().strip() if nombre else ""
    
    # Limpiar apellidos de ruido
    if ap1 in ruido:
        ap1 = ""
    if ap2 in ruido:
        ap2 = ""
    
    # Buscar apellido paterno en texto si no coincide con inicial
    if raw_text and (not ap1 or (ap1 and ap1[0] != inicial_ap1)):
        palabras_texto = re.findall(r'\b([A-ZÁÉÍÓÚÑ]{2,})\b', raw_text.upper())
        # Primero buscar coincidencia exacta
        for palabra in palabras_texto:
            if palabra[0] == inicial_ap1 and palabra in apellidos_comunes:
                ap1 = palabra
                break
        # Si no, buscar apellido que contenga fragmento del texto
        if not ap1 or (ap1 and ap1[0] != inicial_ap1):
            for apellido in apellidos_comunes:
                if apellido[0] == inicial_ap1:
                    # Buscar si alguna palabra del texto es sufijo del apellido (ej: "AZ" en "PAZ")
                    for palabra in palabras_texto:
                        if len(palabra) >= 2 and apellido.endswith(palabra) and len(palabra) >= len(apellido) - 1:
                            ap1 = apellido
                            break
                    if ap1 and ap1[0] == inicial_ap1:
                        break
    
    # Buscar apellido materno en texto si no coincide con inicial  
    if raw_text and (not ap2 or (ap2 and ap2[0] != inicial_ap2)):
        palabras_texto = re.findall(r'\b([A-ZÁÉÍÓÚÑ]{2,})\b', raw_text.upper())
        for palabra in palabras_texto:
            if palabra[0] == inicial_ap2 and palabra in apellidos_comunes:
                if palabra != ap1:  # No repetir apellido
                    ap2 = palabra
                    break
        # Si no, buscar apellido que contenga fragmento
        if not ap2 or (ap2 and ap2[0] != inicial_ap2):
            for apellido in apellidos_comunes:
                if apellido[0] == inicial_ap2 and apellido != ap1:
                    for palabra in palabras_texto:
                        if len(palabra) >= 2 and apellido.endswith(palabra) and len(palabra) >= len(apellido) - 1:
                            ap2 = apellido
                            break
                    if ap2 and ap2[0] == inicial_ap2:
                        break
                    break
    
    # Limpiar nombre
    palabras_validas = []
    encontro_nombre_principal = False
    
    if nombre_clean:
        palabras = nombre_clean.split()
        
        for palabra in palabras:
            if not palabra or len(palabra) < 2:
                continue
            if palabra in ruido:
                continue
            
            # Si empieza con la inicial correcta según CURP
            if palabra[0] == inicial_nombre and not encontro_nombre_principal:
                palabras_validas.append(palabra)
                encontro_nombre_principal = True
            elif encontro_nombre_principal:
                # Ya tenemos nombre principal, agregar segundos nombres válidos
                if palabra in nombres_comunes:
                    palabras_validas.append(palabra)
                elif len(palabra) >= 4 and palabra not in ruido:
                    if any(c in 'AEIOU' for c in palabra):
                        palabras_validas.append(palabra)
            # Si parece nombre pero NO coincide con inicial de CURP, verificar si es apellido
            elif palabra in nombres_comunes and palabra not in apellidos_comunes:
                # Solo aceptar si no tenemos la inicial de CURP como referencia
                if not inicial_nombre or palabra[0] == inicial_nombre:
                    palabras_validas.append(palabra)
                    encontro_nombre_principal = True
    
    # Si no encontramos nombre válido, buscar en texto crudo
    if not encontro_nombre_principal and raw_text:
        palabras_texto = re.findall(r'\b([A-ZÁÉÍÓÚÑ]{3,})\b', raw_text.upper())
        
        for palabra in palabras_texto:
            if palabra[0] == inicial_nombre and palabra in nombres_comunes:
                if palabra != ap1 and palabra != ap2:
                    palabras_validas.insert(0, palabra)
                    encontro_nombre_principal = True
                    break
    
    if palabras_validas:
        nombre_clean = ' '.join(palabras_validas)
    elif not encontro_nombre_principal:
        if nombre:
            palabras = [p for p in nombre.upper().split() if p not in ruido and len(p) >= 3]
            nombre_clean = ' '.join(palabras) if palabras else ""
    
    # Reconstruir nombre completo
    partes = [p for p in [ap1, ap2, nombre_clean] if p]
    nombre_completo_clean = ' '.join(partes)
    
    return nombre_completo_clean, ap1, ap2, nombre_clean


def process_single_image(image_path: str, output_path: Optional[str] = None,
                         debug: bool = False, debug_dir: str = None) -> Dict:
    """Procesa una imagen y retorna resultado con validación cruzada"""
    extractor = INEExtractor(debug=debug, debug_dir=debug_dir)
    result = extractor.extract_to_dict(image_path)
    
    # Siempre incluir validación cruzada
    validation = cross_validate_curp(result)
    result['_validacion'] = {
        'valido': validation['valid'],
        'score': validation['score'],
        'checks': {k: v['status'] for k, v in validation['checks'].items()},
        'errores': validation['errors'],
        'advertencias': validation['warnings']
    }
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Resultado guardado en: {output_path}")
    
    return result


def main():
    """Función principal CLI"""
    parser = argparse.ArgumentParser(
        description='Extractor OCR para credenciales INE de México con validación cruzada',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Ejemplos:
  python ine_ocr.py imagen.jpg
  python ine_ocr.py imagen.jpg -o resultado.json
  python ine_ocr.py imagen.jpg --debug
  python ine_ocr.py imagen.jpg --debug --debug-dir ./mi_debug'''
    )
    
    parser.add_argument('image', nargs='?', help='Ruta a la imagen de INE')
    parser.add_argument('-o', '--output', help='Archivo de salida JSON')
    parser.add_argument('--debug', action='store_true', 
                        help='Modo debug: guarda imágenes intermedias y genera reporte HTML')
    parser.add_argument('--debug-dir', default='./debug_output',
                        help='Directorio para archivos de debug (default: ./debug_output)')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.image:
        result = process_single_image(
            args.image, 
            args.output,
            debug=args.debug,
            debug_dir=args.debug_dir if args.debug else None
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        if args.debug:
            print(f"\n[DEBUG] Reporte HTML generado en: {args.debug_dir}/debug_report.html")
            print(f"[DEBUG] Imágenes de preprocesamiento en: {args.debug_dir}/")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()