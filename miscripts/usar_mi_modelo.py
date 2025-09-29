#!/usr/bin/env python3
"""
🎤 Script para usar tu modelo TTS entrenado y generar audio desde texto
Optimizado para modelos de voz masculina en español
"""

import os
import torch
from pathlib import Path
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from TTS.api import TTS

class MiTTSPersonalizado:
    """Clase para usar tu modelo TTS personalizado"""
    
    def __init__(self, model_path=None):
        """
        Inicializa el sintetizador
        
        Args:
            model_path: Ruta al modelo entrenado (opcional)
        """
        self.model_path = model_path or "/home/softwebdd/my-projects/TTS/models_entrenados"
        self.synthesizer = None
        self.load_model()
    
    def load_model(self):
        """Carga el modelo entrenado"""
        try:
            # Buscar el mejor checkpoint
            checkpoint_path = self.find_best_checkpoint()
            config_path = os.path.join(self.model_path, "config.json")
            
            if not os.path.exists(checkpoint_path):
                print("❌ No se encontró modelo entrenado")
                print(f"💡 Asegúrate de haber entrenado un modelo en: {self.model_path}")
                return False
            
            print(f"📂 Cargando modelo desde: {checkpoint_path}")
            
            # Cargar el modelo
            self.synthesizer = Synthesizer(
                tts_checkpoint=checkpoint_path,
                tts_config_path=config_path,
                use_cuda=torch.cuda.is_available()
            )
            
            print("✅ Modelo cargado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def find_best_checkpoint(self):
        """Encuentra el mejor checkpoint disponible"""
        model_dir = Path(self.model_path)
        
        # Buscar checkpoints
        checkpoints = list(model_dir.glob("*.pth"))
        
        if not checkpoints:
            # Buscar en subdirectorios
            for subdir in model_dir.iterdir():
                if subdir.is_dir():
                    subdir_checkpoints = list(subdir.glob("*.pth"))
                    checkpoints.extend(subdir_checkpoints)
        
        if not checkpoints:
            raise FileNotFoundError("No se encontraron checkpoints (.pth)")
        
        # Ordenar por fecha de modificación (más reciente primero)
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return str(checkpoints[0])
    
    def text_to_speech(self, text, output_file="output.wav"):
        """
        Convierte texto a audio usando tu voz
        
        Args:
            text: Texto a sintetizar
            output_file: Archivo de salida
        """
        if not self.synthesizer:
            print("❌ Modelo no cargado")
            return False
        
        try:
            print(f"🎙️ Generando audio para: '{text}'")
            
            # Generar audio
            wav = self.synthesizer.tts(text)
            
            # Guardar archivo
            self.synthesizer.save_wav(wav, output_file)
            
            print(f"✅ Audio generado: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error generando audio: {e}")
            return False

def demo_interactive():
    """Demo interactivo para probar el modelo"""
    print("🎤 DEMO INTERACTIVO - TU VOZ SINTÉTICA")
    print("="*50)
    
    # Inicializar TTS
    tts = MiTTSPersonalizado()
    
    if not tts.synthesizer:
        print("❌ No se pudo cargar el modelo")
        return
    
    print("\n💡 Consejos:")
    print("   - Escribe frases claras y bien puntuadas")
    print("   - Usa números en formato texto (ej: 'tres' en lugar de '3')")
    print("   - Evita abreviaciones")
    print("   - Escribe 'salir' para terminar")
    
    output_dir = Path("audio_generado")
    output_dir.mkdir(exist_ok=True)
    
    counter = 1
    
    while True:
        print(f"\n{'='*30}")
        text = input("📝 Escribe el texto a sintetizar: ").strip()
        
        if text.lower() in ['salir', 'exit', 'quit']:
            print("👋 ¡Hasta luego!")
            break
        
        if not text:
            print("❌ Texto vacío, inténtalo de nuevo")
            continue
        
        # Generar audio
        output_file = output_dir / f"mi_voz_{counter:03d}.wav"
        
        if tts.text_to_speech(text, str(output_file)):
            print(f"🎵 Archivo guardado: {output_file}")
            counter += 1
        
        # Preguntar si quiere continuar
        continue_choice = input("\n¿Generar otro audio? (s/n): ").strip().lower()
        if continue_choice not in ['s', 'si', 'sí', 'y', 'yes']:
            break

def test_pretrained_models():
    """Prueba modelos preentrenados disponibles"""
    print("🧪 PROBANDO MODELOS PREENTRENADOS")
    print("="*40)
    
    # Lista de modelos disponibles en español
    spanish_models = [
        "tts_models/es/css10/vits",  # Español
        "tts_models/multilingual/multi-dataset/your_tts"  # Multilingüe
    ]
    
    test_text = "Hola, este es un ejemplo de síntesis de voz en español."
    
    for model_name in spanish_models:
        try:
            print(f"\n🔄 Probando modelo: {model_name}")
            
            # Inicializar TTS
            tts = TTS(model_name=model_name, progress_bar=False)
            
            # Generar audio
            output_file = f"test_{model_name.replace('/', '_')}.wav"
            tts.tts_to_file(text=test_text, file_path=output_file)
            
            print(f"✅ Audio generado: {output_file}")
            
        except Exception as e:
            print(f"❌ Error con {model_name}: {e}")

def main():
    """Función principal"""
    print("🎙️ SINTETIZADOR TTS PERSONALIZADO")
    print("="*40)
    
    while True:
        print("\n¿Qué quieres hacer?")
        print("1. Usar mi modelo entrenado (demo interactivo)")
        print("2. Probar modelos preentrenados")
        print("3. Generar un audio específico")
        print("4. Salir")
        
        choice = input("\nElige una opción (1-4): ").strip()
        
        if choice == "1":
            demo_interactive()
            
        elif choice == "2":
            test_pretrained_models()
            
        elif choice == "3":
            text = input("📝 Texto a sintetizar: ").strip()
            if text:
                output_file = input("📁 Nombre del archivo (ej: mi_audio.wav): ").strip()
                if not output_file:
                    output_file = "audio_generado.wav"
                
                tts = MiTTSPersonalizado()
                tts.text_to_speech(text, output_file)
            
        elif choice == "4":
            print("👋 ¡Hasta luego!")
            break
            
        else:
            print("❌ Opción no válida")

if __name__ == "__main__":
    main()