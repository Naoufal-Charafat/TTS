#!/usr/bin/env python3
"""
🚀 Script de instalación optimizada para TTS con voz masculina en español
Basado en análisis de rendimiento y mejores prácticas
"""

import subprocess
import sys
import platform
import pkg_resources

def check_python_version():
    """Verifica que la versión de Python sea compatible"""
    print("🐍 Verificando versión de Python...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ es requerido. Versión actual:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} - Compatible")
    return True

def check_gpu_support():
    """Verifica si hay soporte para GPU"""
    print("\n🎮 Verificando soporte GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detectada: {gpu_name} ({gpu_count} dispositivo(s))")
            return True
        else:
            print("⚠️  No se detectó GPU CUDA. Se usará CPU (más lento)")
            return False
    except ImportError:
        print("⚠️  PyTorch no instalado aún")
        return False

def install_package(package, description=""):
    """Instala un paquete con manejo de errores"""
    try:
        print(f"📦 Instalando {package}...")
        if description:
            print(f"   {description}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package, 
            "--upgrade", "--no-cache-dir"
        ])
        print(f"✅ {package} instalado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {package}: {e}")
        return False

def main():
    """Función principal de instalación"""
    print("🎤 INSTALADOR TTS PARA VOZ MASCULINA EN ESPAÑOL")
    print("=" * 60)
    print("Basado en análisis de rendimiento de 🐸TTS")
    print("Optimizado para modelos VITS, GlowTTS y XTTS v2")
    print("=" * 60)
    
    # Verificar versión de Python
    if not check_python_version():
        return False
    
    print(f"\n💻 Sistema operativo: {platform.system()} {platform.release()}")
    
    # Lista de paquetes esenciales en orden de instalación
    packages = [
        # PyTorch primero (base fundamental)
        ("torch", "Framework de deep learning (base de TTS)"),
        ("torchaudio", "Procesamiento de audio para PyTorch"),
        
        # TTS y dependencias principales
        ("TTS", "🐸 Coqui TTS - Framework principal"),
        
        # Procesamiento de audio optimizado para voces masculinas
        ("librosa", "Análisis avanzado de audio"),
        ("soundfile", "Lectura/escritura de archivos de audio"),
        ("scipy", "Procesamiento científico de señales"),
        
        # Herramientas adicionales
        ("numpy", "Computación numérica"),
        ("matplotlib", "Visualización de espectrogramas"),
        ("tensorboard", "Monitoreo del entrenamiento"),
        
        # Optimizaciones específicas
        ("pyworld", "Análisis de voz (F0 para voces masculinas)"),
        ("espeak-ng", "Síntesis de fonemas en español"),
    ]
    
    print("\n📋 PAQUETES A INSTALAR:")
    for i, (package, desc) in enumerate(packages, 1):
        print(f"   {i:2d}. {package:15} - {desc}")
    
    print(f"\n📊 Total de paquetes: {len(packages)}")
    
    # Confirmar instalación
    response = input("\n¿Continuar con la instalación? (s/N): ").lower().strip()
    if response not in ['s', 'si', 'sí', 'y', 'yes']:
        print("❌ Instalación cancelada")
        return False
    
    # Actualizar pip primero
    print("\n🔧 Actualizando pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Instalar paquetes
    print("\n🚀 INICIANDO INSTALACIÓN...")
    print("=" * 40)
    
    failed_packages = []
    
    for i, (package, description) in enumerate(packages, 1):
        print(f"\n[{i}/{len(packages)}] {package}")
        if not install_package(package, description):
            failed_packages.append(package)
    
    # Verificar soporte GPU después de instalar PyTorch
    check_gpu_support()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE INSTALACIÓN")
    print("=" * 60)
    
    if failed_packages:
        print(f"❌ Paquetes fallidos ({len(failed_packages)}):")
        for package in failed_packages:
            print(f"   • {package}")
        print(f"\n✅ Paquetes exitosos: {len(packages) - len(failed_packages)}")
        print(f"❌ Paquetes fallidos: {len(failed_packages)}")
        print("\n💡 Intenta instalar manualmente los paquetes fallidos:")
        for package in failed_packages:
            print(f"   pip install {package}")
    else:
        print("🎉 ¡TODOS LOS PAQUETES INSTALADOS EXITOSAMENTE!")
        print("\n🎯 LISTO PARA:")
        print("   • Entrenar modelos TTS con voz masculina")
        print("   • Usar modelos pre-entrenados en español")
        print("   • Procesar datasets de audio")
    
    print("\n📚 PRÓXIMOS PASOS:")
    print("   1. Ejecuta: python demo_tts_simple.py")
    print("   2. Prepara tu dataset: python preparar_dataset.py")
    print("   3. Entrena tu modelo: python train_mi_voz.py")
    print("   4. Usa tu modelo: python usar_mi_modelo.py")
    
    print("\n💡 MODELOS RECOMENDADOS PARA ESPAÑOL MASCULINO:")
    print("   🟢 VITS - Mejor calidad (recomendado)")
    print("   🟢 XTTS v2 - Clonación de voz")
    print("   🟢 GlowTTS - Principiantes")
    
    return len(failed_packages) == 0

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 ¡Instalación completada exitosamente!")
        sys.exit(0)
    else:
        print("\n⚠️  Instalación completada con algunos errores")
        sys.exit(1)