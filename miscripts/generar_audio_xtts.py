#!/usr/bin/env python3
"""
🎤 Generador de Audio XTTS v2 - Solo XTTS, Sin Fallback
Script directo para convertir texto a audio usando exclusivamente XTTS v2
Optimizado para voz masculina en español con pronunciación técnica avanzada
"""

import os
import sys
import time
import torch
from pathlib import Path

def configure_pytorch_compatibility():
    """Configura la compatibilidad para PyTorch 2.6+"""
    print("🔧 Configurando compatibilidad PyTorch para XTTS v2...")
    
    # Solución 1: Variable de entorno (más compatible)
    os.environ['PYTORCH_LOAD_WEIGHTS_ONLY'] = 'False'
    print("   ✅ PYTORCH_LOAD_WEIGHTS_ONLY=False aplicado")
    
    # Solución 2: Configurar torch.serialization para clases XTTS
    try:
        import torch.serialization
        
        # Lista completa de clases TTS que necesitan ser registradas como seguras
        safe_classes = []
        
        # Clases principales de XTTS
        try:
            from TTS.tts.models.xtts import XttsAudioConfig, Xtts
            from TTS.tts.configs.xtts_config import XttsConfig
            safe_classes.extend([XttsAudioConfig, Xtts, XttsConfig])
        except ImportError as e:
            print(f"   ⚠️  No se pudieron importar clases XTTS: {e}")
        
        # Clases de configuración compartida
        try:
            from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig, BaseTrainingConfig
            safe_classes.extend([BaseDatasetConfig, BaseAudioConfig, BaseTrainingConfig])
        except ImportError as e:
            print(f"   ⚠️  No se pudieron importar configuraciones base: {e}")
        
        # Clases de configuración TTS
        try:
            from TTS.tts.configs.shared_configs import BaseTTSConfig, CharactersConfig
            safe_classes.extend([BaseTTSConfig, CharactersConfig])
        except ImportError as e:
            print(f"   ⚠️  No se pudieron importar configuraciones TTS: {e}")
        
        # Clases de modelos generales
        try:
            from TTS.tts.models.base_tts import BaseTTS
            from TTS.utils.audio import AudioProcessor
            safe_classes.extend([BaseTTS, AudioProcessor])
        except ImportError as e:
            print(f"   ⚠️  No se pudieron importar modelos base: {e}")
        
        # Clases específicas de datasets y vocoder si existen
        try:
            from TTS.vocoder.configs.shared_configs import BaseVocoderConfig
            from TTS.tts.datasets import TTSDataset
            safe_classes.extend([BaseVocoderConfig, TTSDataset])
        except ImportError:
            pass  # Estas son opcionales
        
        # Registrar todas las clases como seguras
        registered_count = 0
        for cls in safe_classes:
            try:
                torch.serialization.add_safe_globals([cls])
                print(f"   ✅ Clase {cls.__name__} registrada como segura")
                registered_count += 1
            except Exception as e:
                print(f"   ⚠️  No se pudo registrar {cls.__name__}: {e}")
        
        print(f"   ✅ {registered_count}/{len(safe_classes)} clases registradas correctamente")
        
        # Solución 3: Aplicar configuración global de torch.load
        try:
            # Monkeypatch torch.load para usar weights_only=False por defecto
            original_torch_load = torch.load
            
            def patched_torch_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            torch.load = patched_torch_load
            print("   ✅ Patch aplicado a torch.load")
            
        except Exception as e:
            print(f"   ⚠️  No se pudo aplicar patch a torch.load: {e}")
        
        print("   ✅ Configuración de seguridad PyTorch completada")
        return True
        
    except Exception as e:
        print(f"   ⚠️  Configuración parcial PyTorch: {e}")
        return False

def apply_global_pytorch_fix():
    """Aplica fix global de PyTorch antes de cualquier importación de TTS"""
    # Fix 1: Variable de entorno
    os.environ['PYTORCH_LOAD_WEIGHTS_ONLY'] = 'False'
    
    # Fix 2: Patch global de torch.load
    import torch
    
    # Guardar la función original
    if not hasattr(torch, '_original_load'):
        torch._original_load = torch.load
        
        def safe_torch_load(*args, **kwargs):
            # Forzar weights_only=False para todos los casos
            kwargs['weights_only'] = False
            return torch._original_load(*args, **kwargs)
        
        # Reemplazar torch.load globalmente
        torch.load = safe_torch_load
        print("🔧 Patch global aplicado a torch.load")

def load_xtts_model(use_gpu=None):
    """Carga el modelo XTTS v2 con configuración optimizada"""
    print("📥 Cargando modelo XTTS v2...")
    
    # Aplicar fix antes de importar TTS
    apply_global_pytorch_fix()
    
    try:
        from TTS.api import TTS
        
        # Determinar uso de GPU
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        
        gpu_status = "GPU" if use_gpu else "CPU"
        print(f"   🎮 Dispositivo: {gpu_status}")
        
        if use_gpu:
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   📊 Memoria GPU: {memory_gb:.1f} GB")
        
        # Cargar modelo XTTS v2
        start_time = time.time()
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        print(f"   🔄 Inicializando {model_name}...")
        tts = TTS(model_name=model_name, gpu=use_gpu)
        
        load_time = time.time() - start_time
        print(f"   ✅ Modelo cargado en {load_time:.2f} segundos")
        
        # Verificar capacidades
        print(f"   🌍 Multilingüe: {'✅' if tts.is_multi_lingual else '❌'}")
        print(f"   🎭 Multi-speaker: {'✅' if tts.is_multi_speaker else '❌'}")
        
        if tts.is_multi_lingual and tts.languages:
            spanish_supported = "es" in tts.languages
            print(f"   🇪🇸 Español soportado: {'✅' if spanish_supported else '❌'}")
            if not spanish_supported:
                raise ValueError("El modelo no soporta español")
        
        return tts
        
    except Exception as e:
        print(f"❌ ERROR cargando XTTS v2: {e}")
        
        # Diagnóstico específico
        error_msg = str(e)
        if "WeightsUnpickler error" in error_msg or "weights_only" in error_msg:
            print("\n🚨 DIAGNÓSTICO: Error de compatibilidad PyTorch 2.6+")
            print("💡 SOLUCIONES RECOMENDADAS:")
            print("   1. DOWNGRADE PyTorch:")
            print("      pip install torch==2.4.0 torchvision==0.19.0")
            print("   2. O usar variable de entorno antes de ejecutar:")
            print("      export PYTORCH_LOAD_WEIGHTS_ONLY=False")
            print("   3. O modificar el código TTS directamente")
            
        elif "CUDA" in error_msg or "GPU" in error_msg:
            print("\n🚨 DIAGNÓSTICO: Error de GPU")
            print("💡 Ejecutando en CPU...")
            
        elif "Memory" in error_msg or "OOM" in error_msg:
            print("\n🚨 DIAGNÓSTICO: Memoria insuficiente")
            print("💡 Reduce el tamaño del texto o usa GPU con más memoria")
        
        # Re-lanzar la excepción para forzar el fallo (sin fallback)
        raise e

def generate_audio_xtts(text, output_file="audio_xtts.wav", reference_audio=None, **kwargs):
    """
    Genera audio usando XTTS v2 exclusivamente
    
    Args:
        text (str): Texto a convertir en audio
        output_file (str): Archivo de salida
        reference_audio (str): Archivo de audio de referencia para clonación (opcional)
        **kwargs: Parámetros adicionales para XTTS
    """
    print(f"\n🎵 Generando audio con XTTS v2...")
    print(f"   📝 Texto: {len(text)} caracteres")
    print(f"   📄 Archivo salida: {output_file}")
    
    # Configurar parámetros optimizados para voz masculina española
    default_params = {
        'language': 'es',
        'split_sentences': True,
        # Parámetros XTTS optimizados para voz masculina
        'temperature': 0.7,          # Más estable que el default 0.85
        'length_penalty': 1.0,       # Balance natural
        'repetition_penalty': 2.5,   # Evitar repeticiones
        'top_k': 50,                 # Vocabulario enfocado
        'top_p': 0.8,                # Probabilidad conservadora
    }
    
    # Combinar parámetros por defecto con los proporcionados
    tts_params = {**default_params, **kwargs}
    
    # Configurar compatibilidad PyTorch
    configure_pytorch_compatibility()
    
    try:
        # Cargar modelo
        tts = load_xtts_model()
        
        print(f"   🎯 Configuración:")
        print(f"      • Idioma: {tts_params['language']}")
        print(f"      • Split sentences: {tts_params['split_sentences']}")
        print(f"      • Temperature: {tts_params.get('temperature', 'default')}")
        print(f"      • Clonación: {'✅' if reference_audio else '❌'}")
        
        # Preparar parámetros para TTS
        tts_call_params = {
            'text': text,
            'file_path': output_file,
            'language': tts_params['language'],
            'split_sentences': tts_params['split_sentences']
        }
        
        # Añadir audio de referencia si se proporciona
        if reference_audio:
            if os.path.exists(reference_audio):
                tts_call_params['speaker_wav'] = reference_audio
                print(f"   🎭 Audio de referencia: {reference_audio}")
            else:
                print(f"   ⚠️  Audio de referencia no encontrado: {reference_audio}")
                print("   🔄 Generando sin clonación...")
        
        # Si es multi-speaker y no hay audio de referencia, usar mi_voz.WAV por defecto
        if tts.is_multi_speaker and 'speaker_wav' not in tts_call_params:
            # Buscar el archivo mi_voz.WAV en el directorio del script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_voice_file = os.path.join(script_dir, "mi_voz.WAV")
            
            if os.path.exists(default_voice_file):
                tts_call_params['speaker_wav'] = default_voice_file
                print(f"   🎭 Usando voz de referencia por defecto: mi_voz.WAV")
            else:
                # Buscar otros archivos de voz disponibles
                possible_refs = [
                    "benchmark_resultados/tts_models_es_css10_vits.wav",
                    "tests/inputs/example_1.wav",
                    "tests/data/ljspeech/wavs/LJ001-0001.wav"
                ]
                
                reference_found = False
                for ref_path in possible_refs:
                    full_path = os.path.join(os.getcwd(), ref_path)
                    if os.path.exists(full_path):
                        tts_call_params['speaker_wav'] = full_path
                        print(f"   🎭 Usando audio de referencia alternativo: {ref_path}")
                        reference_found = True
                        break
                
                if not reference_found:
                    print("   ⚠️  No se encontró audio de referencia, el modelo puede fallar")
                    print("   � Coloca un archivo 'mi_voz.WAV' en miscripts/ para usar como referencia")
        
        # Añadir parámetros avanzados solo si el modelo los soporta
        advanced_params = ['temperature', 'length_penalty', 'repetition_penalty', 'top_k', 'top_p']
        for param in advanced_params:
            if param in tts_params and hasattr(tts, 'synthesizer'):
                # Solo añadir si XTTS soporta estos parámetros
                model_type = str(type(tts.synthesizer.tts_model)).lower()
                if 'xtts' in model_type:
                    tts_call_params[param] = tts_params[param]
        
        # Generar audio
        start_time = time.time()
        print(f"   🔄 Generando audio...")
        
        tts.tts_to_file(**tts_call_params)
        
        generation_time = time.time() - start_time
        
        # Verificar resultado
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / 1024
            print(f"   ✅ Audio generado exitosamente!")
            print(f"   📊 Tiempo: {generation_time:.2f}s")
            print(f"   📊 Tamaño: {file_size:.1f} KB")
            print(f"   🎧 Archivo: {output_file}")
            
            # Análisis del texto para mostrar características procesadas
            text_analysis = analyze_text(text)
            if text_analysis:
                print(f"   🎯 Características procesadas:")
                for feature, count in text_analysis.items():
                    if count > 0:
                        print(f"      • {feature}: {count}")
            
            return True
        else:
            print(f"   ❌ Error: archivo no generado")
            return False
            
    except Exception as e:
        print(f"❌ ERROR generando audio: {e}")
        return False

def analyze_text(text):
    """Analiza el texto para identificar características complejas"""
    import re
    
    analysis = {
        'Números': len(re.findall(r'\d+', text)),
        'Porcentajes': len(re.findall(r'\d+\.?\d*%', text)),
        'Emails': len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
        'URLs': len(re.findall(r'https?://\S+', text)),
        'Términos HTML/CSS': len(re.findall(r'\b(HTML|CSS|JavaScript|JS|HTTP|HTTPS|API|JSON|XML)\b', text, re.IGNORECASE)),
        'Acrónimos': len(re.findall(r'\b[A-Z]{2,}\b', text)),
        'Símbolos especiales': len(re.findall(r'[#$%&*+=<>{}|~^`]', text)),
        'Palabras en inglés': len(re.findall(r'\b(software|hardware|framework|responsive|design|web|app|server)\b', text, re.IGNORECASE))
    }
    
    return analysis

def demo_pronunciation_test():
    """Ejecuta una demostración con texto que prueba pronunciación avanzada"""
    print("\n🧪 DEMO: Test de pronunciación avanzada XTTS v2")
    print("=" * 60)
    
    demo_text = """Demostración de pronunciación técnica avanzada en español:
    
    1. Desarrollo web moderno con HTML5, CSS3, JavaScript ES6
    2. APIs REST, bases de datos MySQL, PostgreSQL  
    3. Frameworks: React.js, Vue.js, Angular.js, Node.js
    4. Herramientas: Git, Docker, Kubernetes, AWS, Azure
    5. Números y métricas: 123 usuarios, 45.5% de conversión, $2,750.50 de ingresos
    6. Contacto técnico: desarrollador@empresa.com, +34 123-456-789
    7. Protocolos: HTTP/HTTPS, TCP/IP, SSL/TLS, OAuth 2.0
    8. Lenguajes: Python 3.11, TypeScript 4.8, Go 1.19, Rust 1.65
    
    Este test evalúa la pronunciación de términos técnicos, números, emails y acrónimos."""
    
    return generate_audio_xtts(
        text=demo_text,
        output_file="demo_pronunciacion_xtts.wav",
        temperature=0.7,
        repetition_penalty=2.5
    )

def main():
    """Función principal con interfaz interactiva"""
    print("🎤 GENERADOR DE AUDIO XTTS v2")
    print("=" * 50)
    print("Script especializado en XTTS v2 - Sin fallback")
    print("Optimizado para pronunciación técnica en español")
    print("=" * 50)
    
    # Aplicar fix de PyTorch inmediatamente
    apply_global_pytorch_fix()
    
    # Verificar PyTorch
    pytorch_version = torch.__version__
    print(f"🔧 PyTorch version: {pytorch_version}")
    
    version_parts = pytorch_version.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    
    if (major > 2) or (major == 2 and minor >= 6):
        print("⚠️  PyTorch 2.6+ detectado - Fix aplicado automáticamente")
    else:
        print("✅ PyTorch compatible")
    
    # Opciones del script
    print("\n📋 OPCIONES:")
    print("1. Demo de pronunciación técnica")
    print("2. Texto personalizado")
    print("3. Texto personalizado con clonación de voz")
    print("4. Salir")
    
    try:
        choice = input("\n👉 Selecciona una opción (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Ejecutando demo de pronunciación...")
            success = demo_pronunciation_test()
            if success:
                print("\n🎉 ¡Demo completado exitosamente!")
                print("🎧 Reproduce 'demo_pronunciacion_xtts.wav' para escuchar el resultado")
            else:
                print("\n❌ Demo falló - Revisa la configuración PyTorch")
        
        elif choice == "2":
            print("\n📝 Ingresa tu texto:")
            text = input("Texto: ").strip()
            
            if text:
                output_file = input("Archivo de salida (Enter para 'mi_audio_xtts.wav'): ").strip()
                if not output_file:
                    output_file = "mi_audio_xtts.wav"
                
                print(f"\n🚀 Generando audio...")
                success = generate_audio_xtts(text, output_file)
                
                if success:
                    print(f"\n🎉 ¡Audio generado exitosamente!")
                    print(f"🎧 Reproduce '{output_file}' para escuchar el resultado")
                else:
                    print("\n❌ Generación falló")
            else:
                print("❌ Texto vacío")
        
        elif choice == "3":
            print("\n📝 Ingresa tu texto:")
            text = input("Texto: ").strip()
            
            print("🎭 Archivo de audio de referencia:")
            ref_audio = input("Ruta al archivo de referencia: ").strip()
            
            if text:
                output_file = input("Archivo de salida (Enter para 'mi_audio_clonado_xtts.wav'): ").strip()
                if not output_file:
                    output_file = "mi_audio_clonado_xtts.wav"
                
                print(f"\n🚀 Generando audio con clonación...")
                success = generate_audio_xtts(
                    text=text, 
                    output_file=output_file,
                    reference_audio=ref_audio if ref_audio else None
                )
                
                if success:
                    print(f"\n🎉 ¡Audio clonado generado exitosamente!")
                    print(f"🎧 Reproduce '{output_file}' para escuchar el resultado")
                else:
                    print("\n❌ Generación falló")
            else:
                print("❌ Texto vacío")
        
        elif choice == "4":
            print("👋 ¡Hasta luego!")
            return True
        
        else:
            print("❌ Opción inválida")
            return False
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")
        print("\n🔧 SOLUCIONES:")
        print("1. Instala torch==2.4.0: pip install torch==2.4.0 torchvision==0.19.0")
        print("2. O ejecuta con: PYTORCH_LOAD_WEIGHTS_ONLY=False python Generar_audio_xtts.py")
        print("3. Verifica que TTS esté instalado: pip install TTS")
        sys.exit(1)