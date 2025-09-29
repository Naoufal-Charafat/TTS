#!/usr/bin/env python3
"""
🎤 Script para preparar tu dataset personalizado para entrenamiento TTS
Optimizado para VOCES MASCULINAS en ESPAÑOL
Basado en mejores prácticas del análisis de rendimiento TTS
"""

import os
import librosa
import soundfile as sf
from pathlib import Path

def create_dataset_structure(base_path):
    """Crea la estructura básica del dataset"""
    dataset_path = Path(base_path)
    wavs_path = dataset_path / "wavs"
    
    # Crear directorios
    dataset_path.mkdir(exist_ok=True)
    wavs_path.mkdir(exist_ok=True)
    
    print(f"📁 Estructura del dataset creada en: {dataset_path}")
    return dataset_path, wavs_path

def process_audio_file(input_file, output_file, target_sr=22050):
    """
    Procesa un archivo de audio para cumplir con los requisitos de TTS
    Optimizado para voces masculinas
    """
    try:
        # Cargar audio
        audio, sr = librosa.load(input_file, sr=None)
        
        # Remuestrear si es necesario
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Normalizar audio (importante para voces masculinas)
        audio = librosa.util.normalize(audio)
        
        # Filtro pasa-alto suave para voces masculinas (eliminar ruido de baja frecuencia)
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        
        # Guardar
        sf.write(output_file, audio, target_sr)
        
        duration = len(audio) / target_sr
        print(f"✅ Procesado: {input_file} -> {output_file} ({duration:.2f}s)")
        return True, duration
    except Exception as e:
        print(f"❌ Error procesando {input_file}: {e}")
        return False, 0

def create_sample_metadata():
    """Crea un archivo metadata.txt de ejemplo optimizado para voces masculinas"""
    sample_texts = [
        "Hola, este es mi primer audio de entrenamiento para voz masculina.",
        "Los números uno, dos, tres se pronuncian claramente con tono grave.",
        "La síntesis de voz masculina requiere técnicas especializadas.",
        "Es importante hablar con claridad y proyección adecuada.",
        "Este dataset me ayudará a crear mi voz sintética profunda.",
        "La inteligencia artificial puede imitar perfectamente voces graves.",
        "Cada grabación debe tener entre tres y siete segundos idealmente.",
        "La calidad del audio es fundamental para voces masculinas.",
        "Necesito grabar al menos diez horas de audio para mejor resultado.",
        "Mi voz será convertida en un modelo de alta calidad.",
        "Los tonos graves requieren configuraciones específicas de procesamiento.",
        "La respiración y la dicción son cruciales en voces masculinas.",
        "Este proyecto de síntesis vocal será muy útil.",
        "La tecnología TTS ha avanzado significativamente en español.",
        "Pronuncio cada palabra con claridad y naturalidad."
    ]
    
    metadata_content = []
    for i, text in enumerate(sample_texts, 1):
        audio_name = f"audio{i:03d}"
        # Formato: nombre_archivo|texto_original|texto_normalizado
        metadata_content.append(f"{audio_name}|{text}|{text}")
    
    return "\n".join(metadata_content)

def analyze_dataset(dataset_path):
    """Analiza las características del dataset"""
    wavs_path = Path(dataset_path) / "wavs"
    metadata_path = Path(dataset_path) / "metadata.txt"
    
    if not metadata_path.exists():
        print("❌ No se encontró metadata.txt")
        return
    
    # Leer metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\n📊 ANÁLISIS DEL DATASET")
    print(f"{'='*40}")
    print(f"📝 Total de muestras: {len(lines)}")
    
    # Analizar archivos de audio
    audio_files = list(wavs_path.glob("*.wav"))
    print(f"🎵 Archivos de audio encontrados: {len(audio_files)}")
    
    if audio_files:
        total_duration = 0
        durations = []
        
        for audio_file in audio_files[:10]:  # Analizar primeros 10 para velocidad
            try:
                duration = librosa.get_duration(path=audio_file)
                durations.append(duration)
                total_duration += duration
            except:
                continue
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            print(f"⏱️ Duración promedio por clip: {avg_duration:.2f} segundos")
            print(f"⏱️ Duración total estimada: {total_duration * len(audio_files) / len(durations) / 3600:.2f} horas")
            print(f"⏱️ Rango de duración: {min(durations):.2f}s - {max(durations):.2f}s")
    
    # Analizar textos
    text_lengths = []
    for line in lines:
        if '|' in line:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                text_lengths.append(len(parts[1]))
    
    if text_lengths:
        avg_text_len = sum(text_lengths) / len(text_lengths)
        print(f"📝 Longitud promedio del texto: {avg_text_len:.1f} caracteres")
        print(f"📝 Rango de longitud: {min(text_lengths)} - {max(text_lengths)} caracteres")

def main():
    """Función principal"""
    print("🎤 PREPARADOR DE DATASET PARA TTS")
    print("="*40)
    
    base_path = "/home/softwebdd/my-projects/TTS/MiDatasetTTS"
    
    while True:
        print("\n¿Qué quieres hacer?")
        print("1. Crear estructura de dataset")
        print("2. Crear metadata.txt de ejemplo")
        print("3. Analizar dataset existente")
        print("4. Convertir archivos de audio")
        print("5. Salir")
        
        choice = input("\nElige una opción (1-5): ").strip()
        
        if choice == "1":
            dataset_path, wavs_path = create_dataset_structure(base_path)
            print(f"\n✅ Estructura creada:")
            print(f"   📁 {dataset_path}")
            print(f"   📁 {wavs_path}")
            print(f"\n💡 Ahora debes:")
            print(f"   1. Colocar tus archivos .wav en: {wavs_path}")
            print(f"   2. Crear el archivo metadata.txt en: {dataset_path}")
            
        elif choice == "2":
            dataset_path = Path(base_path)
            metadata_content = create_sample_metadata()
            metadata_file = dataset_path / "metadata.txt"
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(metadata_content)
            
            print(f"\n✅ Archivo metadata.txt de ejemplo creado en: {metadata_file}")
            print("💡 Este es solo un ejemplo. Debes reemplazar con tus propios textos.")
            
        elif choice == "3":
            analyze_dataset(base_path)
            
        elif choice == "4":
            input_dir = input("Ruta de los archivos de audio originales: ").strip()
            if not os.path.exists(input_dir):
                print("❌ La ruta no existe")
                continue
                
            dataset_path, wavs_path = create_dataset_structure(base_path)
            
            input_files = list(Path(input_dir).glob("*.wav")) + list(Path(input_dir).glob("*.mp3"))
            print(f"🎵 Encontrados {len(input_files)} archivos de audio")
            
            for i, input_file in enumerate(input_files):
                output_file = wavs_path / f"audio{i+1:03d}.wav"
                success, duration = process_audio_file(input_file, output_file)
                if success:
                    print(f"✅ Procesado: {input_file.name} -> {output_file.name} ({duration:.2f}s)")
                    
        elif choice == "5":
            print("👋 ¡Hasta luego!")
            break
            
        else:
            print("❌ Opción no válida")

if __name__ == "__main__":
    main()