#!/usr/bin/env python3
"""
🎯 COMANDOS ESENCIALES PARA TTS - VOZ MASCULINA EN ESPAÑOL
Guía rápida de referencia con todos los comandos necesarios
"""

def mostrar_comandos_esenciales():
    print("🎤 COMANDOS ESENCIALES TTS - VOZ MASCULINA EN ESPAÑOL")
    print("=" * 65)
    
    print("\n🚀 1. INSTALACIÓN Y SETUP INICIAL")
    print("-" * 35)
    print("# Instalar todas las dependencias optimizadas")
    print("python instalar_dependencias.py")
    print()
    print("# Verificar que todo funciona correctamente")
    print("python test_entorno.py")
    print()
    print("# Probar demo con modelo español masculino")
    print("python demo_tts_simple.py")
    
    print("\n🎤 2. ENTRENAR TU MODELO PERSONALIZADO")
    print("-" * 38)
    print("# Paso 1: Preparar tu dataset de voz")
    print("# REQUISITO: Mínimo 30 minutos de audio WAV, 22050Hz, mono")
    print("python preparar_dataset.py")
    print()
    print("# Paso 2: Entrenar modelo VITS (mejor calidad para voz masculina)")
    print("CUDA_VISIBLE_DEVICES=0 python train_mi_voz.py")
    print()
    print("# Si se interrumpe, continuar entrenamiento:")
    print("python train_mi_voz.py --continue_path ./models_entrenados/checkpoint_latest.pth")
    print()
    print("# Monitorear progreso en tiempo real:")
    print("tensorboard --logdir=./models_entrenados/")
    
    print("\n🎵 3. GENERAR AUDIO CON TU MODELO ENTRENADO")
    print("-" * 42)
    print("# Método 1: Interactivo (recomendado)")
    print("python usar_mi_modelo.py")
    print()
    print("# Método 2: Comando directo")
    print('python -c "from usar_mi_modelo import MiTTSPersonalizado; MiTTSPersonalizado().text_to_speech(\'Hola, esta es mi voz sintética\', \'mi_voz.wav\')"')
    print()
    print("# Método 3: Múltiples textos")
    print("python -c \"\"\"")
    print("from usar_mi_modelo import MiTTSPersonalizado")
    print("tts = MiTTSPersonalizado()")
    print("textos = ['Texto 1', 'Texto 2', 'Texto 3']")
    print("for i, texto in enumerate(textos):")
    print("    tts.text_to_speech(texto, f'audio_{i+1}.wav')")
    print('"""')
    
    print("\n🇪🇸 4. USAR MODELOS PRE-ENTRENADOS EN ESPAÑOL")
    print("-" * 44)
    print("# Listar todos los modelos TTS disponibles")
    print('python -c "from TTS.api import TTS; print(\'\\n\'.join(TTS.list_models()))"')
    print()
    print("# Listar solo modelos en español")
    print('python -c "from TTS.api import TTS; [print(m) for m in TTS.list_models() if \'es\' in m]"')
    print()
    print("# MEJOR MODELO: VITS Español (máxima calidad para voz masculina)")
    print('python -c "from TTS.api import TTS; TTS(\'tts_models/es/css10/vits\').tts_to_file(\'Hola, soy una voz masculina española de alta calidad\', \'voz_masculina_es.wav\')"')
    print()
    print("# Modelo alternativo: Tacotron2 Español")
    print('python -c "from TTS.api import TTS; TTS(\'tts_models/es/mai/tacotron2-DDC\').tts_to_file(\'Texto alternativo\', \'voz_tacotron_es.wav\')"')
    print()
    print("# Comparar rendimiento de diferentes modelos")
    print("python comparar_modelos.py")
    
    print("\n⚙️  5. CONFIGURACIÓN SEGÚN TU HARDWARE")
    print("-" * 37)
    print("# GPU 8GB+ (óptimo):")
    print("# MODEL_TYPE='vits', batch_size=32")
    print()
    print("# GPU 6-8GB:")
    print("# MODEL_TYPE='vits', batch_size=16")
    print() 
    print("# GPU <6GB:")
    print("# MODEL_TYPE='glowtts', batch_size=8")
    print()
    print("# Solo CPU (muy lento):")
    print("# MODEL_TYPE='glowtts', batch_size=4")
    
    print("\n📊 6. MONITOREO Y DIAGNÓSTICO")
    print("-" * 29)
    print("# Verificar estado completo del sistema")
    print("python test_entorno.py")
    print()
    print("# Ver logs del entrenamiento en tiempo real")
    print("tail -f ./models_entrenados/train_log.txt")
    print()
    print("# Verificar uso de GPU")
    print("nvidia-smi")
    print()
    print("# Monitoreo con TensorBoard")
    print("tensorboard --logdir=./models_entrenados/ --port=6006")
    
    print("\n🔧 7. COMANDOS DE SOLUCIÓN DE PROBLEMAS")
    print("-" * 38)
    print("# Reinstalar dependencias si hay problemas")
    print("pip uninstall TTS -y && python instalar_dependencias.py")
    print()
    print("# Limpiar cache de modelos")
    print("rm -rf ~/.local/share/tts/")
    print()
    print("# Verificar versión de CUDA")
    print("python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, Versión: {torch.version.cuda}')\"")
    print()
    print("# Liberar memoria GPU")
    print("python -c \"import torch; torch.cuda.empty_cache(); print('Cache GPU limpiado')\"")
    
    print("\n📁 8. ESTRUCTURA DE ARCHIVOS REQUERIDA")
    print("-" * 36)
    print("MiDatasetTTS/")
    print("├── metadata.txt          # archivo_audio|texto_original|texto_normalizado")
    print("└── wavs/                 # carpeta con archivos WAV")
    print("    ├── audio001.wav      # 22050Hz, mono, 3-7 segundos")
    print("    ├── audio002.wav")
    print("    └── ...")
    print()
    print("FORMATO DE AUDIO OBLIGATORIO:")
    print("• Formato: WAV sin compresión")
    print("• Sample Rate: 22050 Hz")
    print("• Canales: Mono (1 canal)")
    print("• Duración total: Mínimo 30 minutos")
    print("• Duración por clip: 3-7 segundos")
    print("• Bit depth: 16-bit mínimo")
    
    print("\n⏱️  9. TIEMPOS ESTIMADOS DE ENTRENAMIENTO")
    print("-" * 40)
    print("GPU RTX 3080/4070+ (8GB+):")
    print("• VITS: 12-24 horas para 10 horas de audio")
    print("• GlowTTS: 6-12 horas para 10 horas de audio")
    print()
    print("GPU RTX 3060/4060 (6-8GB):")
    print("• VITS: 20-36 horas para 10 horas de audio")
    print("• GlowTTS: 8-16 horas para 10 horas de audio")
    print()
    print("CPU (no recomendado):")
    print("• Cualquier modelo: 3-7 días para 10 horas de audio")
    
    print("\n🎯 10. FLUJO DE TRABAJO COMPLETO")
    print("-" * 31)
    print("1️⃣  python instalar_dependencias.py")
    print("2️⃣  python test_entorno.py")
    print("3️⃣  python demo_tts_simple.py")
    print("4️⃣  # Grabar tu voz (30+ minutos, WAV, 22kHz)")
    print("5️⃣  python preparar_dataset.py")
    print("6️⃣  python train_mi_voz.py")
    print("7️⃣  python usar_mi_modelo.py")
    print("8️⃣  # ¡Disfruta tu voz sintética!")
    
    print("\n" + "=" * 65)
    print("💡 CONSEJOS PARA MEJOR CALIDAD DE VOZ MASCULINA:")
    print("   • Habla con tono natural y consistente")
    print("   • Graba en ambiente silencioso")
    print("   • Usa micrófono de buena calidad")
    print("   • Varía el contenido (preguntas, números, emociones)")
    print("   • Mínimo 5-10 horas de audio para calidad profesional")
    print("=" * 65)

def generar_script_rapido():
    """Genera un script de comandos rápidos"""
    script_content = '''#!/bin/bash
# Script rápido para TTS - Voz Masculina Español

echo "🎤 SETUP RÁPIDO TTS - VOZ MASCULINA EN ESPAÑOL"
echo "=============================================="

# 1. Instalación
echo "📦 Instalando dependencias..."
python instalar_dependencias.py

# 2. Verificación
echo "🧪 Verificando instalación..."
python test_entorno.py

# 3. Demo
echo "🎵 Probando demo..."
python demo_tts_simple.py

echo "✅ ¡Setup completado!"
echo "📚 Próximos pasos:"
echo "   1. Graba tu voz (30+ min, WAV, 22kHz)"
echo "   2. python preparar_dataset.py"
echo "   3. python train_mi_voz.py"
echo "   4. python usar_mi_modelo.py"
'''
    
    with open('setup_rapido.sh', 'w') as f:
        f.write(script_content)
    
    # Hacer ejecutable
    import os
    os.chmod('setup_rapido.sh', 0o755)
    
    print("📄 Script creado: setup_rapido.sh")
    print("   Ejecutar con: ./setup_rapido.sh")

if __name__ == "__main__":
    mostrar_comandos_esenciales()
    print("\n" + "="*65)
    generar_script_rapido()