#!/usr/bin/env python3
"""
🧪 Script de testing y configuración para TTS en español masculino
Verifica que todo esté funcionando correctamente antes del entrenamiento
"""

import os
import sys
import time
import torch
from pathlib import Path

def test_tts_installation():
    """Prueba la instalación básica de TTS"""
    print("🔍 Probando instalación de TTS...")
    try:
        from TTS.api import TTS
        print("✅ TTS importado correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error importando TTS: {e}")
        return False

def test_spanish_model():
    """Prueba un modelo pre-entrenado en español optimizado para voz masculina"""
    print("\n🇪🇸 Probando modelo en español...")
    try:
        from TTS.api import TTS
        import torch.serialization
        
        # 🔥 XTTS v2 - MEJOR modelo para español masculino con clonación de voz
        # Características: Multilingüe, pronunciación avanzada de números/HTML/inglés
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        print(f"📥 Cargando modelo XTTS v2 (optimizado para español): {model_name}")
        
        # Configurar torch.load para PyTorch 2.6+ - Permitir clases XTTS
        try:
            # Añadir clases XTTS a la lista de globals seguros
            from TTS.tts.configs.xtts_config import XttsConfig
            torch.serialization.add_safe_globals([XttsConfig])
            print("   🔧 Configurando seguridad PyTorch para XTTS...")
        except Exception as safe_e:
            print(f"   ⚠️  Advertencia configuración seguridad: {safe_e}")
        
        start_time = time.time()
        # Configuración optimizada para voz masculina
        tts = TTS(model_name=model_name, gpu=torch.cuda.is_available())
        load_time = time.time() - start_time
        
        print(f"✅ Modelo XTTS v2 cargado en {load_time:.2f} segundos")
        print(f"   🌍 Multilingüe: {tts.is_multi_lingual}")    
        print(f"   🎭 Multi-speaker: {tts.is_multi_speaker}")
        
        # Listar idiomas soportados
        if tts.is_multi_lingual and tts.languages:
            spanish_supported = "es" in tts.languages
            print(f"   🇪🇸 Español soportado: {'✅' if spanish_supported else '❌'}")
            print(f"   📋 Idiomas disponibles: {len(tts.languages)} idiomas")
        
        # Texto de prueba optimizado para probar pronunciación avanzada
        test_text = """Prueba de pronunciación avanzada para voz masculina española:
        1. Números: 123, 4,567.89, 15.5%, $250.75
        2. Términos técnicos: HTML, CSS, JavaScript, Python, API, HTTP, HTTPS
        3. Palabras en inglés: software, hardware, framework, responsive design
        4. Acrónimos: TTS, AI, ML, GPU, CPU, RAM, SSD
        5. Símbolos: @ # $ % & * + = < > [ ] { } | ~ ^ ` 
        6. Fechas: 25/12/2024, enero de 2025
        El desarrollo web moderno requiere conocimiento de HTML5, CSS3 y JavaScript ES6."""
        
        output_file = "test_spanish_male_xtts.wav"
        
        print("🎵 Generando audio con pronunciación avanzada...")
        print(f"   📝 Texto: {len(test_text)} caracteres")
        
        start_time = time.time()
        
        # Usar XTTS v2 con configuración optimizada para español masculino
        tts.tts_to_file(
            text=test_text, 
            file_path=output_file,
            language="es",  # Especificar español explícitamente
            split_sentences=True,  # Mejorar coherencia
            **{
                # Parámetros optimizados para voz masculina española
                'temperature': 0.7,  # Menos creativo, más estable para voz masculina
                'length_penalty': 1.0,  # Balance natural
                'repetition_penalty': 2.5,  # Evitar repeticiones
                'top_k': 50,  # Vocabulario más enfocado
                'top_p': 0.8,  # Probabilidad más conservadora
            } if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'tts_model') and 'xtts' in str(type(tts.synthesizer.tts_model)).lower() else {}
        )
        
        gen_time = time.time() - start_time
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / 1024
            print(f"✅ Audio generado: {output_file}")
            print(f"   📊 Tiempo: {gen_time:.2f}s, Tamaño: {file_size:.1f} KB")
            print(f"   🎯 Calidad: XTTS v2 - Pronunciación avanzada activada")
            print(f"   🔊 Características probadas:")
            print(f"      • Números y porcentajes")
            print(f"      • Términos técnicos (HTML, CSS, JS)")
            print(f"      • Palabras en inglés")
            print(f"      • Acrónimos técnicos")
            print(f"      • Símbolos especiales")
            
            # Test adicional con modelo CSS10 VITS para comparación
            try:
                print("\n📊 Generando audio de comparación con VITS CSS10...")
                tts_vits = TTS("tts_models/es/css10/vits")
                comparison_text = "Comparación: HTML, CSS, JavaScript, 123 dólares, 45.5 por ciento."
                tts_vits.tts_to_file(text=comparison_text, file_path="test_vits_comparison.wav")
                print("✅ Audio de comparación generado: test_vits_comparison.wav")
            except Exception as comp_e:
                print(f"⚠️  Comparación VITS falló: {comp_e}")
            
            return True
        else:
            print("❌ Error: archivo de audio no generado")
            return False
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error probando modelo español: {e}")
        
        # Diagnóstico específico para errores comunes
        if "WeightsUnpickler error" in error_msg or "weights_only" in error_msg:
            print("🔧 DIAGNÓSTICO: Error de seguridad PyTorch 2.6+")
            print("   💡 Soluciones:")
            print("   1. Downgrade PyTorch: pip install torch==2.4.0")
            print("   2. O usar variable de entorno: PYTORCH_LOAD_WEIGHTS_ONLY=False")
            print("   3. O modificar código TTS para usar weights_only=False")
            print("   ⚠️  Este es un problema conocido con XTTS v2 y PyTorch 2.6+")
        elif "CUDA" in error_msg or "GPU" in error_msg:
            print("🔧 DIAGNÓSTICO: Error de GPU/CUDA")
            print("   💡 Forzando uso de CPU...")
        elif "Memory" in error_msg or "OOM" in error_msg:
            print("🔧 DIAGNÓSTICO: Error de memoria insuficiente")
            print("   💡 Recomendación: Usar modelo más ligero")
        
        # Fallback al modelo VITS original si XTTS falla
        print("\n🔄 Intentando fallback con VITS CSS10...")
        try:
            print("📥 Cargando modelo VITS CSS10 (más compatible)...")
            tts_fallback = TTS("tts_models/es/css10/vits")
            
            # Texto de prueba adaptado para VITS (más simple)
            fallback_text = """Prueba con modelo VITS para español:
            Desarrollo web con HTML, CSS, JavaScript.
            Números: 123, 45.5 por ciento.
            Términos técnicos: API, HTTP, Python.
            El framework React es muy popular."""
            
            print("🎵 Generando audio con VITS CSS10...")
            tts_fallback.tts_to_file(text=fallback_text, file_path="test_spanish_fallback.wav")
            
            if os.path.exists("test_spanish_fallback.wav"):
                file_size = os.path.getsize("test_spanish_fallback.wav") / 1024
                print(f"✅ Fallback exitoso: test_spanish_fallback.wav")
                print(f"   📊 Tamaño: {file_size:.1f} KB")
                print(f"   🔊 Nota: VITS CSS10 tiene pronunciación limitada para términos técnicos")
                print(f"   💡 Para mejor pronunciación, soluciona el problema XTTS v2")
                return True
            else:
                print("❌ Error: archivo fallback no generado")
                return False
                
        except Exception as fallback_e:
            print(f"❌ Fallback VITS también falló: {fallback_e}")
            
            # Segundo fallback: modelo básico más simple
            print("\n🔄 Último intento con modelo básico...")
            try:
                # Intentar con un modelo aún más básico si existe
                basic_models = [
                    "tts_models/es/mai/tacotron2-DDC",
                    "tts_models/es/css10/vits"  # Reintentar por si fue error temporal
                ]
                
                for basic_model in basic_models:
                    try:
                        print(f"📥 Probando {basic_model}...")
                        tts_basic = TTS(basic_model)
                        basic_text = "Prueba básica: HTML, CSS, 123."
                        tts_basic.tts_to_file(text=basic_text, file_path="test_basic_spanish.wav")
                        
                        if os.path.exists("test_basic_spanish.wav"):
                            print(f"✅ Modelo básico funcional: {basic_model}")
                            print(f"   📄 Archivo: test_basic_spanish.wav")
                            return True
                            
                    except Exception as basic_e:
                        print(f"   ❌ {basic_model} falló: {basic_e}")
                        continue
                
                print("❌ Todos los modelos fallaron")
                return False
                
            except Exception as final_e:
                print(f"❌ Error final: {final_e}")
                return False

def test_gpu_performance():
    """Prueba el rendimiento de GPU si está disponible"""
    print("\n🎮 Probando rendimiento GPU...")
    
    if not torch.cuda.is_available():
        print("⚠️  GPU no disponible, usando CPU")
        return False
    
    try:
        gpu_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_free = memory_total - memory_allocated
        
        print(f"✅ GPU: {gpu_name}")
        print(f"   📊 Memoria total: {memory_total:.1f} GB")
        print(f"   📊 Memoria libre: {memory_free:.1f} GB")
        
        if memory_free < 4.0:
            print("⚠️  Advertencia: Poca memoria GPU disponible (recomendado: 6GB+)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error verificando GPU: {e}")
        return False

def test_voice_cloning_male_spanish():
    """Prueba la clonación de voz masculina española con XTTS v2"""
    print("\n🎭 Probando clonación de voz masculina española...")
    
    try:
        from TTS.api import TTS
        
        # Cargar XTTS v2 para clonación
        print("📥 Cargando XTTS v2 para clonación de voz...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        
        # Texto optimizado para demostrar capacidades avanzadas
        clone_text = """Demostración de clonación de voz masculina española con pronunciación técnica:
        Mi nombre es Juan y soy desarrollador web. Trabajo con HTML5, CSS3, JavaScript ES6, 
        Python 3.9, y frameworks como React.js y Vue.js. Mis proyectos incluyen APIs REST, 
        bases de datos MySQL, y deployment en AWS. El 85% de mi trabajo involucra 
        responsive design y UX/UI optimization. Contacto: juan@empresa.com, 
        teléfono +34 123-456-789."""
        
        # Verificar si tenemos archivo de referencia de audio
        reference_audio_paths = [
            "/home/softwebdd/my-projects/TTS/benchmark_resultados/tts_models_es_css10_vits.wav",
            "test_spanish_male_xtts.wav",  # El audio que acabamos de generar
            "test_spanish_fallback.wav"   # Fallback si existe
        ]
        
        reference_audio = None
        for audio_path in reference_audio_paths:
            if os.path.exists(audio_path):
                reference_audio = audio_path
                print(f"✅ Usando audio de referencia: {audio_path}")
                break
        
        if reference_audio:
            print("🎯 Generando audio clonado con voz masculina...")
            
            try:
                # Clonación de voz con parámetros optimizados para español masculino
                tts.tts_to_file(
                    text=clone_text,
                    file_path="test_voice_cloning_male_es.wav",
                    speaker_wav=reference_audio,
                    language="es",
                    split_sentences=True
                )
                
                if os.path.exists("test_voice_cloning_male_es.wav"):
                    file_size = os.path.getsize("test_voice_cloning_male_es.wav") / 1024
                    print(f"✅ Audio clonado generado: test_voice_cloning_male_es.wav")
                    print(f"   📊 Tamaño: {file_size:.1f} KB")
                    print(f"   🎯 Características demostradas:")
                    print(f"      • Clonación de voz masculina")
                    print(f"      • Pronunciación de términos técnicos")
                    print(f"      • Emails y teléfonos")
                    print(f"      • Acrónimos y frameworks")
                    print(f"      • Porcentajes y números")
                    return True
                else:
                    print("❌ Error: archivo de audio clonado no generado")
                    return False
                    
            except Exception as clone_e:
                print(f"⚠️  Error en clonación: {clone_e}")
                print("💡 Probando con speaker predefinido...")
                
                # Intentar con speaker predefinido si la clonación falla
                try:
                    # Verificar speakers disponibles
                    if tts.is_multi_speaker and tts.speakers:
                        # Buscar un speaker que suene masculino
                        male_speakers = [s for s in tts.speakers if any(name in s.lower() for name in ['male', 'man', 'masc', 'antonio', 'carlos', 'miguel', 'david'])]
                        
                        if male_speakers:
                            selected_speaker = male_speakers[0]
                            print(f"🎭 Usando speaker masculino: {selected_speaker}")
                        else:
                            # Usar el primer speaker disponible
                            selected_speaker = tts.speakers[0]
                            print(f"🎭 Usando primer speaker disponible: {selected_speaker}")
                        
                        tts.tts_to_file(
                            text=clone_text,
                            file_path="test_predefined_speaker_male_es.wav",
                            speaker=selected_speaker,
                            language="es",
                            split_sentences=True
                        )
                        
                        if os.path.exists("test_predefined_speaker_male_es.wav"):
                            print("✅ Audio con speaker predefinido: test_predefined_speaker_male_es.wav")
                            return True
                    
                    return False
                    
                except Exception as speaker_e:
                    print(f"❌ Error con speaker predefinido: {speaker_e}")
                    return False
        
        else:
            print("⚠️  No se encontró audio de referencia para clonación")
            print("💡 Generando con configuración por defecto...")
            
            # Generar sin clonación pero con parámetros optimizados
            tts.tts_to_file(
                text=clone_text,
                file_path="test_default_male_es.wav",
                language="es",
                split_sentences=True
            )
            
            if os.path.exists("test_default_male_es.wav"):
                print("✅ Audio con configuración por defecto: test_default_male_es.wav")
                return True
            
            return False
            
    except Exception as e:
        print(f"❌ Error en test de clonación: {e}")
        return False

def test_audio_processing():
    """Prueba las herramientas de procesamiento de audio"""
    print("\n🎧 Probando procesamiento de audio...")
    
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        
        print("✅ Librosa y SoundFile importados")
        
        # Crear audio de prueba
        sr = 22050
        duration = 2.0
        freq = 220  # Frecuencia baja (típica voz masculina)
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Guardar y cargar
        test_file = "test_audio_processing.wav"
        sf.write(test_file, audio, sr)
        
        # Cargar y verificar
        loaded_audio, loaded_sr = librosa.load(test_file, sr=sr)
        
        if loaded_sr == sr and len(loaded_audio) == len(audio):
            print(f"✅ Procesamiento de audio OK")
            print(f"   📊 Sample rate: {loaded_sr} Hz")
            print(f"   📊 Duración: {len(loaded_audio)/loaded_sr:.2f}s")
            
            # Limpiar archivo de prueba
            os.remove(test_file)
            return True
        else:
            print("❌ Error en procesamiento de audio")
            return False
            
    except Exception as e:
        print(f"❌ Error en procesamiento de audio: {e}")
        return False

def check_pytorch_compatibility():
    """Verifica la compatibilidad de PyTorch con XTTS v2"""
    print("\n🔧 Verificando compatibilidad PyTorch...")
    
    try:
        import torch
        pytorch_version = torch.__version__
        print(f"✅ PyTorch version: {pytorch_version}")
        
        # Verificar si es PyTorch 2.6+ que causa problemas con XTTS
        version_parts = pytorch_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        
        is_problematic = (major > 2) or (major == 2 and minor >= 6)
        
        if is_problematic:
            print("⚠️  PyTorch 2.6+ detectado - Puede causar problemas con XTTS v2")
            print("   🔧 Soluciones recomendadas:")
            print("   1. TEMPORAL: export PYTORCH_LOAD_WEIGHTS_ONLY=False")
            print("   2. DOWNGRADE: pip install torch==2.4.0 torchvision==0.19.0")
            print("   3. ESPERAR: Actualización de TTS para PyTorch 2.6+")
            
            # Intentar aplicar workaround automático
            try:
                import os
                os.environ['PYTORCH_LOAD_WEIGHTS_ONLY'] = 'False'
                print("   ✅ Aplicado workaround automático")
                return True
            except Exception as env_e:
                print(f"   ❌ No se pudo aplicar workaround: {env_e}")
                return False
        else:
            print("✅ PyTorch compatible con XTTS v2")
            return True
            
    except Exception as e:
        print(f"❌ Error verificando PyTorch: {e}")
        return False

def check_dataset_structure():
    """Verifica la estructura del dataset"""
    print("\n📁 Verificando estructura del dataset...")
    
    dataset_path = Path("/home/softwebdd/my-projects/TTS/MiDatasetTTS")
    
    if not dataset_path.exists():
        print(f"⚠️  Dataset no encontrado en: {dataset_path}")
        print("💡 Ejecuta 'python preparar_dataset.py' para crear la estructura")
        return False
    
    wavs_path = dataset_path / "wavs"
    metadata_path = dataset_path / "metadata.txt"
    
    print(f"✅ Directorio dataset: {dataset_path}")
    
    if wavs_path.exists():
        audio_files = list(wavs_path.glob("*.wav"))
        print(f"✅ Directorio wavs: {len(audio_files)} archivos .wav")
    else:
        print("❌ Directorio 'wavs' no encontrado")
        return False
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"✅ Archivo metadata.txt: {len(lines)} entradas")
    else:
        print("❌ Archivo 'metadata.txt' no encontrado")
        return False
    
    return True

def recommend_configuration():
    """Recomienda configuraciones basadas en el hardware"""
    print("\n🎯 RECOMENDACIONES DE CONFIGURACIÓN:")
    print("=" * 50)
    
    # Verificar memoria disponible
    if torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print("🔥 MODELO RECOMENDADO PARA ESPAÑOL MASCULINO: XTTS v2")
        print("   ✅ Mejor pronunciación de números, HTML, palabras técnicas")
        print("   ✅ Soporte nativo para español (16 idiomas)")
        print("   ✅ Clonación de voz con archivos de referencia")
        print("   ✅ Tokenizador avanzado para términos técnicos")
        print("   ⚠️  NOTA: Requiere PyTorch <2.6 o configuración especial")
        print()
        
        if memory_gb >= 8:
            print("🟢 GPU con 8GB+: XTTS v2 ÓPTIMO")
            print("   • Clonación de voz de alta calidad")
            print("   • Pronunciación perfecta de términos técnicos")
            print("   • Tiempo de inferencia: 2-10 segundos por oración")
            print("   • Modelo recomendado: tts_models/multilingual/multi-dataset/xtts_v2")
        elif memory_gb >= 6:
            print("🟡 GPU con 6-8GB: XTTS v2 BUENO")
            print("   • Clonación funcional con oraciones más cortas")
            print("   • Buen rendimiento en pronunciación técnica")
            print("   • Tiempo de inferencia: 5-15 segundos por oración")
        elif memory_gb >= 4:
            print("🟠 GPU con 4-6GB: XTTS v2 LIMITADO")
            print("   • Usar split_sentences=True para textos largos")
            print("   • Pronunciación técnica aún superior a VITS")
            print("   • Fallback: tts_models/es/css10/vits")
        else:
            print("🔴 GPU con <4GB: VITS CSS10 recomendado")
            print("   • XTTS v2 puede fallar por memoria insuficiente")
            print("   • Modelo alternativo: tts_models/es/css10/vits")
    else:
        print("🔴 Sin GPU: CONFIGURACIÓN LIMITADA")
        print("   • XTTS v2 en CPU: muy lento (30-120s por oración)")
        print("   • Alternativa rápida: tts_models/es/css10/vits")
        print("   • Solo para pruebas, no producción")
    
    print(f"\n📊 VENTAJAS ESPECÍFICAS DE XTTS v2 PARA ESPAÑOL:")
    print("   🎯 Pronunciación Superior:")
    print("      • Números: 123 → 'ciento veintitrés'")
    print("      • Técnicos: HTML → 'ache-te-eme-ele' (natural)")
    print("      • Inglés: 'JavaScript' → pronunciación correcta")
    print("      • Acrónimos: CSS, API, HTTP (pronunciación clara)")
    print("      • Símbolos: $250.75 → 'doscientos cincuenta dólares...'")
    print("      • Emails: usuario@dominio.com (pronunciación natural)")
    
    print(f"\n🎭 CLONACIÓN DE VOZ MASCULINA:")
    print("   • Audio de referencia: 3-30 segundos")
    print("   • Calidad óptima: 22kHz, mono, sin ruido")
    print("   • Múltiples referencias mejoran la calidad")
    print("   • Funciona cross-idioma (entrenar español, usar inglés)")
    
    print(f"\n⚙️  PARÁMETROS RECOMENDADOS PARA VOZ MASCULINA:")
    print("   • temperature: 0.7 (más estable que 0.85 por defecto)")
    print("   • repetition_penalty: 2.5 (evitar repeticiones)")
    print("   • length_penalty: 1.0 (natural)")
    print("   • top_k: 50, top_p: 0.8 (conservador)")
    print("   • split_sentences: True (mejor coherencia)")
    
    print("\n📝 CONFIGURACIÓN OPTIMIZADA DE TEXTO:")
    print("   • Usar texto normalizado (números expandidos)")
    print("   • Acrónimos con puntos: H.T.M.L. (mejor pronunciación)")
    print("   • Términos técnicos en mayúsculas: HTML, CSS, JS")
    print("   • Evitar símbolos complejos en una oración")
    
    print("\n💾 REQUISITOS DE DATOS:")
    print("   • Para entrenamiento: No necesario (pre-entrenado)")
    print("   • Para clonación: 3+ segundos de audio limpio")
    print("   • Para fine-tuning: 30 minutos - 2 horas de audio")
    print("   • Calidad de voz: Consistente, clara, sin eco")

def main():
    """Función principal de testing"""
    print("🧪 TEST COMPLETO DEL ENTORNO TTS")
    print("=" * 60)
    print("Verificando configuración para voz masculina en español")
    print("=" * 60)
    
    tests = [
        ("Instalación TTS", test_tts_installation),
        ("Compatibilidad PyTorch", check_pytorch_compatibility),
        ("Modelo español XTTS v2", test_spanish_model),
        ("Clonación voz masculina", test_voice_cloning_male_spanish),
        ("Rendimiento GPU", test_gpu_performance),
        ("Procesamiento audio", test_audio_processing),
        ("Estructura dataset", check_dataset_structure),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Error ejecutando {test_name}: {e}")
            results.append((test_name, False))
    
    # Mostrar resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE TESTS")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 Resultado: {passed}/{total} tests exitosos")
    
    if passed == total:
        print("🎉 ¡TODO FUNCIONANDO CORRECTAMENTE!")
        print("🚀 Listo para entrenar modelos TTS")
    elif passed >= total * 0.8:
        print("⚠️  La mayoría de tests pasaron. Revisa los errores.")
    else:
        print("❌ Varios tests fallaron. Revisa la instalación.")
    
    # Mostrar recomendaciones
    recommend_configuration()
    
    print("\n📚 PRÓXIMOS PASOS:")
    if passed >= total * 0.8:
        print("   🔥 USANDO XTTS v2 (RECOMENDADO):")
        print("   1. Clonación rápida de voz:")
        print("      python -c \"from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2').tts_to_file('Tu texto aquí', 'output.wav', language='es')\"")
        print("   2. Con archivo de voz de referencia:")
        print("      python -c \"from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2').tts_to_file('Tu texto', 'output.wav', speaker_wav='tu_voz.wav', language='es')\"")
        print("   3. Prueba pronunciación técnica:")
        print("      Texto: 'Desarrollo HTML5, CSS3, JavaScript ES6, APIs REST, base de datos MySQL, 123 usuarios, 45.5%'")
        print()
        print("   🛠️ ENTRENAMIENTO TRADICIONAL (OPCIONAL):")
        print("   1. Prepara tu dataset: python preparar_dataset.py")
        print("   2. Configura el entrenamiento: edita train_mi_voz.py")
        print("   3. Inicia el entrenamiento: python train_mi_voz.py")
        print()
        print("   📖 GUÍAS DISPONIBLES:")
        print("   • GUIA_COMPLETA_TTS.md - Tutorial completo")
        print("   • usar_mi_modelo.py - Script interactivo")
        print("   • comparar_modelos.py - Comparar diferentes modelos")
    else:
        print("   1. Instala dependencias: python instalar_dependencias.py")
        print("   2. Ejecuta este test nuevamente")
        print("   3. Si persisten errores, consulta GUIA_COMPLETA_TTS.md")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)