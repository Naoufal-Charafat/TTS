#!/usr/bin/env python3
"""
🎤 Modelo TTS Personalizado con mi_voz.WAV - EJECUTABLE
Script ejecutable que crea un modelo TTS personalizado usando clonación optimizada
Simula un modelo entrenado usando técnicas avanzadas de XTTS v2

Uso:
    mi_voz archivo.txt              # Convierte texto a audio
    mi_voz -t "Tu texto aquí"       # Texto directo
    mi_voz --config                 # Configurar modelo
    mi_voz --help                   # Ayuda

Ejemplos:
    mi_voz /home/user/documento.txt
    mi_voz -t "Hola mundo con HTML y CSS"
    mi_voz --optimize
"""

import os
import sys
import time
import torch
import librosa
import soundfile as sf
import numpy as np
import argparse
from pathlib import Path
import pickle

class ModeloMiVoz:
    """Clase que simula un modelo entrenado usando clonación optimizada"""
    
    def __init__(self, voice_file="mi_voz.WAV"):
        self.voice_file = voice_file
        self.model_name = "mi_voz"
        self.tts_model = None
        self.voice_embedding = None
        self.optimized_params = {
            'temperature': 0.65,  # Optimizado para tu voz
            'length_penalty': 1.1,
            'repetition_penalty': 2.8,
            'top_k': 40,
            'top_p': 0.75,
            'speed': 1.0,
            'pitch_shift': 0.0
        }
        
    def cargar_modelo(self):
        """Carga y optimiza XTTS v2 para tu voz específica"""
        print(f"🎤 Cargando modelo personalizado '{self.model_name}'...")
        
        # Aplicar fix de PyTorch
        self._aplicar_fix_pytorch()
        
        try:
            from TTS.api import TTS
            
            # Cargar XTTS v2
            print("   📥 Inicializando XTTS v2 base...")
            self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
            
            # Verificar archivo de voz
            script_dir = os.path.dirname(os.path.abspath(__file__))
            voice_path = os.path.join(script_dir, self.voice_file)
            
            if not os.path.exists(voice_path):
                raise FileNotFoundError(f"Archivo de voz no encontrado: {voice_path}")
            
            # Optimizar audio de referencia
            print("   🔧 Optimizando audio de referencia...")
            self.voice_embedding = self._optimizar_audio_referencia(voice_path)
            
            # Cargar parámetros personalizados si existen
            self._cargar_parametros_personalizados()
            
            print(f"   ✅ Modelo '{self.model_name}' listo")
            print(f"   🎯 Archivo de voz: {self.voice_file}")
            print(f"   ⚙️  Parámetros optimizados cargados")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error cargando modelo: {e}")
            return False
    
    def _aplicar_fix_pytorch(self):
        """Aplica fix de compatibilidad PyTorch"""
        os.environ['PYTORCH_LOAD_WEIGHTS_ONLY'] = 'False'
        
        # Patch torch.load
        if not hasattr(torch, '_original_load'):
            torch._original_load = torch.load
            
            def safe_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return torch._original_load(*args, **kwargs)
            
            torch.load = safe_torch_load
    
    def _optimizar_audio_referencia(self, voice_path):
        """Optimiza el audio de referencia para mejor clonación"""
        try:
            # Cargar audio
            audio, sr = librosa.load(voice_path, sr=22050)
            
            # Análisis del audio
            duration = len(audio) / sr
            rms_energy = np.sqrt(np.mean(audio**2))
            
            print(f"      📊 Duración: {duration:.1f}s")
            print(f"      📊 Energía RMS: {rms_energy:.4f}")
            
            # Optimizaciones automáticas
            if duration < 3.0:
                print(f"      ⚠️  Audio corto ({duration:.1f}s). Recomendado: 3-10s")
            elif duration > 15.0:
                print(f"      ⚠️  Audio largo ({duration:.1f}s). Recortando a 10s...")
                audio = audio[:int(10 * sr)]
            
            # Normalización suave
            if rms_energy < 0.1:
                print(f"      🔧 Normalizando volumen bajo...")
                audio = audio * (0.15 / rms_energy)
            elif rms_energy > 0.3:
                print(f"      🔧 Reduciendo volumen alto...")
                audio = audio * (0.25 / rms_energy)
            
            # Guardar versión optimizada
            optimized_path = voice_path.replace('.WAV', '_optimized.wav')
            sf.write(optimized_path, audio, sr)
            
            print(f"      ✅ Audio optimizado guardado: {os.path.basename(optimized_path)}")
            
            return {
                'path': optimized_path,
                'duration': len(audio) / sr,
                'quality_score': self._calcular_calidad_audio(audio, sr)
            }
            
        except Exception as e:
            print(f"      ❌ Error optimizando audio: {e}")
            return {'path': voice_path, 'duration': 0, 'quality_score': 0.5}
    
    def _calcular_calidad_audio(self, audio, sr):
        """Calcula una puntuación de calidad del audio"""
        try:
            # Métricas básicas de calidad
            rms = np.sqrt(np.mean(audio**2))
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # Puntuación simple (0-1)
            volume_score = min(rms * 5, 1.0)  # Penalizar volumen muy bajo
            clarity_score = 1.0 - min(np.mean(zcr), 0.3)  # Penalizar mucho ruido
            spectral_score = min(np.mean(spectral_centroids) / 3000, 1.0)  # Rango vocal normal
            
            overall_score = (volume_score + clarity_score + spectral_score) / 3
            return overall_score
            
        except:
            return 0.5
    
    def _cargar_parametros_personalizados(self):
        """Carga parámetros personalizados si existen"""
        params_file = f"{self.model_name}_params.pkl"
        
        if os.path.exists(params_file):
            try:
                with open(params_file, 'rb') as f:
                    saved_params = pickle.load(f)
                    self.optimized_params.update(saved_params)
                print(f"      ✅ Parámetros personalizados cargados desde {params_file}")
            except:
                print(f"      ⚠️  No se pudieron cargar parámetros de {params_file}")
    
    def generar_audio(self, texto, archivo_salida=None, **kwargs):
        """Genera audio usando el modelo personalizado"""
        if not self.tts_model:
            if not self.cargar_modelo():
                return False
        
        if archivo_salida is None:
            archivo_salida = f"audio_{self.model_name}.wav"
        
        print(f"\n🎵 Generando audio con modelo '{self.model_name}'...")
        print(f"   📝 Texto: {len(texto)} caracteres")
        print(f"   📄 Salida: {archivo_salida}")
        
        # Combinar parámetros optimizados con los proporcionados
        params_finales = {**self.optimized_params, **kwargs}
        
        print(f"   🎯 Parámetros del modelo:")
        for key, value in params_finales.items():
            if key in ['temperature', 'top_k', 'top_p']:
                print(f"      • {key}: {value}")
        
        try:
            # Usar audio optimizado si está disponible
            voice_file = self.voice_embedding['path'] if self.voice_embedding else self.voice_file
            
            # Parámetros para TTS
            tts_params = {
                'text': texto,
                'file_path': archivo_salida,
                'speaker_wav': voice_file,
                'language': 'es',
                'split_sentences': True
            }
            
            # Añadir parámetros avanzados si están soportados
            advanced_params = ['temperature', 'length_penalty', 'repetition_penalty', 'top_k', 'top_p']
            for param in advanced_params:
                if param in params_finales:
                    tts_params[param] = params_finales[param]
            
            start_time = time.time()
            print("   🔄 Sintetizando con XTTS v2...")
            
            self.tts_model.tts_to_file(**tts_params)
            
            gen_time = time.time() - start_time
            
            if os.path.exists(archivo_salida):
                size_kb = os.path.getsize(archivo_salida) / 1024
                print(f"   ✅ Audio generado exitosamente!")
                print(f"   📊 Tiempo: {gen_time:.1f}s")
                print(f"   📊 Tamaño: {size_kb:.1f} KB")
                print(f"   🎧 Archivo: {archivo_salida}")
                
                # Mostrar calidad estimada
                if self.voice_embedding and 'quality_score' in self.voice_embedding:
                    quality = self.voice_embedding['quality_score']
                    quality_text = "Excelente" if quality > 0.8 else "Buena" if quality > 0.6 else "Regular"
                    print(f"   📈 Calidad estimada: {quality_text} ({quality:.2f})")
                
                return True
            else:
                print("   ❌ Error: archivo no generado")
                return False
                
        except Exception as e:
            print(f"   ❌ Error generando audio: {e}")
            return False
    
    def entrenar_parametros(self, textos_prueba, **nuevos_params):
        """Simula entrenamiento optimizando parámetros para tu voz"""
        print(f"\n🧠 'Entrenando' modelo '{self.model_name}'...")
        print("   (Optimizando parámetros de clonación)")
        
        if not self.tts_model:
            if not self.cargar_modelo():
                return False
        
        mejores_params = self.optimized_params.copy()
        mejor_score = 0
        
        # Rangos de parámetros a probar
        param_ranges = {
            'temperature': [0.6, 0.65, 0.7, 0.75],
            'repetition_penalty': [2.5, 2.8, 3.0],
            'top_k': [30, 40, 50],
            'top_p': [0.7, 0.75, 0.8]
        }
        
        print(f"   🔄 Probando {len(param_ranges)} parámetros...")
        
        texto_prueba = textos_prueba[0] if textos_prueba else "Prueba de optimización de voz"
        
        for i, (param, valores) in enumerate(param_ranges.items()):
            print(f"   📊 Optimizando {param}... ({i+1}/{len(param_ranges)})")
            
            mejor_valor = mejores_params[param]
            
            for valor in valores:
                test_params = mejores_params.copy()
                test_params[param] = valor
                
                # Simular puntuación (en un caso real, aquí evaluarías la calidad del audio)
                score = self._simular_evaluacion(test_params)
                
                if score > mejor_score:
                    mejor_score = score
                    mejor_valor = valor
            
            mejores_params[param] = mejor_valor
        
        # Aplicar nuevos parámetros proporcionados
        mejores_params.update(nuevos_params)
        
        self.optimized_params = mejores_params
        
        # Guardar parámetros optimizados
        params_file = f"{self.model_name}_params.pkl"
        try:
            with open(params_file, 'wb') as f:
                pickle.dump(mejores_params, f)
            print(f"   ✅ Parámetros optimizados guardados en {params_file}")
        except:
            print(f"   ⚠️  No se pudieron guardar parámetros")
        
        print(f"   🎯 Parámetros finales optimizados:")
        for key, value in mejores_params.items():
            print(f"      • {key}: {value}")
        
        return True
    
    def _simular_evaluacion(self, params):
        """Simula evaluación de calidad (en un caso real usarías métricas reales)"""
        # Puntuación simulada basada en parámetros óptimos conocidos
        base_score = 0.7
        
        # Preferencias para voz masculina española
        if 0.6 <= params.get('temperature', 0.7) <= 0.75:
            base_score += 0.1
        if 2.5 <= params.get('repetition_penalty', 2.5) <= 3.0:
            base_score += 0.1
        if 30 <= params.get('top_k', 50) <= 50:
            base_score += 0.05
        if 0.7 <= params.get('top_p', 0.8) <= 0.8:
            base_score += 0.05
        
        return min(base_score, 1.0)


def obtener_ruta_modelo():
    """Obtiene la ruta del archivo mi_voz.WAV"""
    # Buscar mi_voz.WAV en varias ubicaciones posibles
    posibles_rutas = [
        # Directorio del script
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "mi_voz.WAV"),
        # Directorio home del usuario
        os.path.expanduser("~/mi_voz.WAV"),
        # Directorio actual
        os.path.join(os.getcwd(), "mi_voz.WAV"),
        # Directorio de TTS miscripts
        os.path.expanduser("~/my-projects/TTS/miscripts/mi_voz.WAV"),
    ]
    
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            return ruta
    
    # Si no se encuentra, devolver la primera opción como default
    return posibles_rutas[0]

def procesar_archivo_texto(archivo_texto):
    """Procesa un archivo de texto y genera audio"""
    if not os.path.exists(archivo_texto):
        print(f"❌ Error: Archivo no encontrado: {archivo_texto}")
        return False
    
    try:
        # Leer contenido del archivo
        with open(archivo_texto, 'r', encoding='utf-8') as f:
            contenido = f.read().strip()
        
        if not contenido:
            print(f"❌ Error: Archivo vacío: {archivo_texto}")
            return False
        
        print(f"📄 Archivo: {archivo_texto}")
        print(f"📝 Contenido: {len(contenido)} caracteres")
        
        # Determinar archivo de salida (mismo directorio que el archivo de texto)
        directorio_archivo = os.path.dirname(os.path.abspath(archivo_texto))
        nombre_base = os.path.splitext(os.path.basename(archivo_texto))[0]
        archivo_salida = os.path.join(directorio_archivo, f"{nombre_base}_audio.wav")
        
        print(f"🎧 Audio salida: {archivo_salida}")
        
        # Crear modelo y generar audio
        ruta_voz = obtener_ruta_modelo()
        if not os.path.exists(ruta_voz):
            print(f"❌ Error: Archivo mi_voz.WAV no encontrado en: {ruta_voz}")
            print("💡 Coloca el archivo mi_voz.WAV en uno de estos directorios:")
            print(f"   • {os.path.dirname(ruta_voz)}")
            print(f"   • {os.path.expanduser('~/')}")
            return False
        
        modelo = ModeloMiVoz(ruta_voz)
        
        print("\n🚀 Iniciando generación de audio...")
        exito = modelo.generar_audio(contenido, archivo_salida)
        
        if exito:
            print(f"\n🎉 ¡Audio generado exitosamente!")
            print(f"📁 Ubicación: {archivo_salida}")
            return True
        else:
            print(f"\n❌ Error al generar audio")
            return False
            
    except Exception as e:
        print(f"❌ Error procesando archivo: {e}")
        return False

def generar_desde_texto(texto, archivo_salida=None):
    """Genera audio directamente desde texto"""
    if not archivo_salida:
        archivo_salida = os.path.join(os.getcwd(), "audio_mi_voz.wav")
    
    print(f"📝 Texto directo: {len(texto)} caracteres")
    print(f"🎧 Audio salida: {archivo_salida}")
    
    ruta_voz = obtener_ruta_modelo()
    if not os.path.exists(ruta_voz):
        print(f"❌ Error: Archivo mi_voz.WAV no encontrado")
        return False
    
    modelo = ModeloMiVoz(ruta_voz)
    return modelo.generar_audio(texto, archivo_salida)

def configurar_modelo():
    """Configuración interactiva del modelo"""
    print("⚙️  CONFIGURACIÓN DEL MODELO MI_VOZ")
    print("=" * 40)
    
    ruta_voz = obtener_ruta_modelo()
    if not os.path.exists(ruta_voz):
        print(f"❌ Error: Archivo mi_voz.WAV no encontrado")
        return False
    
    modelo = ModeloMiVoz(ruta_voz)
    
    textos_prueba = [
        "Optimización de mi modelo de voz personalizado",
        "Desarrollo web con HTML5, CSS3 y JavaScript ES6", 
        "Pronunciación de números: 123, 45.5%, emails: test@example.com"
    ]
    
    return modelo.entrenar_parametros(textos_prueba)

def mostrar_ayuda():
    """Muestra información de ayuda"""
    print("""
🎤 MI_VOZ - Generador de Audio TTS Personalizado
===============================================

DESCRIPCIÓN:
    Convierte texto a audio usando tu voz personalizada (mi_voz.WAV)
    
USO:
    mi_voz archivo.txt              # Convierte archivo de texto a audio
    mi_voz -t "texto"               # Convierte texto directo a audio
    mi_voz --text "texto"           # Alias de -t
    mi_voz -o archivo.wav           # Especifica archivo de salida
    mi_voz --output archivo.wav     # Alias de -o
    mi_voz --config                 # Configurar y optimizar modelo
    mi_voz --optimize               # Alias de --config
    mi_voz --help                   # Mostrar esta ayuda
    mi_voz -h                       # Alias de --help

EJEMPLOS:
    # Convertir archivo de texto
    mi_voz /ruta/a/documento.txt
    
    # Texto directo
    mi_voz -t "Hola mundo con HTML, CSS y JavaScript"
    
    # Especificar archivo de salida
    mi_voz -t "Mi texto" -o mi_audio.wav
    
    # Optimizar modelo
    mi_voz --optimize

NOTAS:
    • El archivo mi_voz.WAV debe estar en el directorio del script o en ~/
    • El audio se guarda en el mismo directorio que el archivo de texto
    • Soporta términos técnicos: HTML, CSS, JavaScript, números, emails
    • Optimizado para voz masculina en español

INSTALACIÓN COMO COMANDO GLOBAL:
    sudo ln -sf $(pwd)/mi_voz.py /usr/local/bin/mi_voz
    chmod +x /usr/local/bin/mi_voz
""")

def parse_argumentos():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='🎤 Generador de Audio TTS Personalizado',
        add_help=False  # Usamos nuestra propia ayuda
    )
    
    # Argumentos posicionales
    parser.add_argument('archivo', nargs='?', help='Archivo de texto a convertir')
    
    # Argumentos opcionales
    parser.add_argument('-t', '--text', help='Texto directo a convertir')
    parser.add_argument('-o', '--output', help='Archivo de salida de audio')
    parser.add_argument('--config', '--optimize', action='store_true', help='Configurar/optimizar modelo')
    parser.add_argument('-h', '--help', action='store_true', help='Mostrar ayuda')
    parser.add_argument('--version', action='store_true', help='Mostrar versión')
    
    return parser.parse_args()

def main_cli():
    """Función principal para uso desde línea de comandos"""
    args = parse_argumentos()
    
    # Banner del programa
    print("🎤 MI_VOZ TTS v1.0")
    print("=" * 25)
    
    # Mostrar ayuda
    if args.help:
        mostrar_ayuda()
        return True
    
    # Mostrar versión
    if args.version:
        print("mi_voz versión 1.0")
        print("Generador TTS personalizado con XTTS v2")
        return True
    
    # Configurar modelo
    if args.config:
        return configurar_modelo()
    
    # Texto directo
    if args.text:
        return generar_desde_texto(args.text, args.output)
    
    # Archivo de texto
    if args.archivo:
        return procesar_archivo_texto(args.archivo)
    
    # Si no hay argumentos, mostrar ayuda
    print("⚠️  No se especificaron argumentos")
    print("💡 Usa: mi_voz --help para ver las opciones")
    return False


if __name__ == "__main__":
    # Configurar compatibilidad de PyTorch
    os.environ["PYTORCH_LOAD_WEIGHTS_ONLY"] = "false"
    
    # Verificar si se ejecuta desde línea de comandos o interactivamente
    import sys
    if len(sys.argv) > 1:
        # Ejecutar desde línea de comandos
        try:
            exito = main_cli()
            sys.exit(0 if exito else 1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        # Ejecutar versión interactiva
        print("🔄 Ejecutando versión interactiva...")
        print("💡 Para usar desde terminal: mi_voz --help")
        print()
        
        # Versión interactiva simplificada
        print("🎤 MODELO TTS PERSONALIZADO - mi_voz")
        print("=" * 50)
        
        try:
            ruta_voz = obtener_ruta_modelo()
            if not os.path.exists(ruta_voz):
                print(f"❌ Error: Archivo mi_voz.WAV no encontrado")
                print(f"💡 Colócalo en: {ruta_voz}")
                sys.exit(1)
            
            modelo = ModeloMiVoz(ruta_voz)
            
            texto = input("\n📝 Ingresa tu texto: ").strip()
            if texto:
                archivo = input("📄 Archivo de salida (Enter para 'audio_mi_voz.wav'): ").strip()
                if not archivo:
                    archivo = "audio_mi_voz.wav"
                
                if modelo.generar_audio(texto, archivo):
                    print(f"\n🎉 ¡Audio generado!")
                    print(f"🎧 Archivo: {archivo}")
                else:
                    print("\n❌ Error en la generación")
        except Exception as e:
            print(f"\n💥 Error crítico: {e}")
            print("\n🔧 Asegúrate de que:")
            print("1. El archivo mi_voz.WAV esté en el mismo directorio")
            print("2. El entorno virtual esté activado")
            print("3. Las dependencias estén instaladas")
        print("3. PyTorch 2.4.0 y TTS estén instalados")
        sys.exit(1)