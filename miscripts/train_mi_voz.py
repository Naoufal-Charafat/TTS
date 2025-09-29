#!/usr/bin/env python3
"""
🎤 Script Optimizado para Entrenar TTS - Voz Masculina en Español
Basado en análisis de rendimiento TTS y mejores modelos para español

Modelos optimizados según gráfico de rendimiento:
🟢 VITS - MEJOR calidad para español (top performance)
🟢 GlowTTS - Rápido y estable para principiantes
🟢 XTTS v2 - Clonación de voz multilingüe
"""

import os
import sys
import torch
from pathlib import Path

# Verificar instalación de TTS
try:
    from trainer import Trainer, TrainerArgs
    from TTS.tts.configs.glow_tts_config import GlowTTSConfig
    from TTS.tts.configs.vits_config import VitsConfig, VitsAudioConfig, VitsArgs
    from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.glow_tts import GlowTTS
    from TTS.tts.models.vits import Vits
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor
    from TTS.utils.manage import ModelManager
    from TTS.config import Coqpit
except ImportError as e:
    print(f"❌ Error importando TTS: {e}")
    print("💡 Instala TTS con: pip install TTS")
    sys.exit(1)

# ==================================================
# CONFIGURACIÓN PERSONALIZABLE
# ==================================================

# Ruta donde está tu dataset
DATASET_PATH = "/home/softwebdd/my-projects/TTS/MiDatasetTTS"

# Ruta donde se guardarán los modelos entrenados
OUTPUT_PATH = "/home/softwebdd/my-projects/TTS/models_entrenados"

# Idioma del dataset - OPTIMIZADO PARA ESPAÑOL MASCULINO
LANGUAGE = "es"  # español
VOICE_GENDER = "male"  # Optimizado para voz masculina

# MODELOS RECOMENDADOS (basado en análisis de rendimiento TTS):
# 🟢 VITS - MEJOR CALIDAD para español (Top en gráfico de rendimiento)
# 🟢 GlowTTS - RÁPIDO y ESTABLE para principiantes  
# 🟠 XTTS v2 - CLONACIÓN DE VOZ multilingüe (requiere menos datos)
# 🔴 YourTTS - Multi-speaker avanzado

MODEL_TYPE = "vits"  # Cambiado a VITS para mejor calidad en español masculino

# Configuración del dataset
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",  # Usamos el formato LJSpeech (standard)
    meta_file_train="metadata.txt",
    path=DATASET_PATH
)

def get_model_config(model_type, language="es"):
    """
    Retorna la configuración del modelo según el tipo seleccionado
    Basado en el análisis de rendimiento de 🐸TTS
    """
    if model_type.lower() == "glowtts":
        # GlowTTS - Mejor para principiantes (Verde en el gráfico de rendimiento)
        return GlowTTSConfig(
            batch_size=16,
            eval_batch_size=8,
            num_loader_workers=2,
            num_eval_loader_workers=2,
            epochs=1000,
            run_eval=True,
            test_delay_epochs=-1,
            mixed_precision=True,
            text_cleaner="phoneme_cleaners",
            use_phonemes=True,
            phoneme_language=language,
            phoneme_cache_path=os.path.join(OUTPUT_PATH, "phoneme_cache"),
            sample_rate=22050,
            hop_length=256,
            win_length=1024,
            print_step=25,
            print_eval=True,
            save_step=1000,
            save_n_checkpoints=5,
            output_path=OUTPUT_PATH,
            datasets=[dataset_config],
        ), GlowTTS
        
    elif model_type.lower() == "vits":
        # VITS - MEJOR CALIDAD para español masculino (Top rendimiento)
        return VitsConfig(
            audio=VitsAudioConfig(
                sample_rate=22050,
                hop_length=256,
                win_length=1024,
                num_mels=80,
                mel_fmin=0,
                mel_fmax=None,
                # Optimización para voz masculina (frecuencias más bajas)
                mel_fmin=0,           # Incluir frecuencias bajas para voces masculinas
                mel_fmax=8000,        # Rango optimizado para voz masculina
            ),
            model_args=VitsArgs(
                use_speaker_embedding=False,  # Single speaker para tu voz
                use_sdp=True,                 # Predictor de duración estocástico 
                noise_scale=0.667,            # Optimizado para español
                noise_scale_dp=1.0,           # Control de duración
                length_scale=1.0,             # Velocidad natural
                segment_size=8192,            # Tamaño de segmento optimizado
            ),
            batch_size=32,              # Aumentado para VITS
            eval_batch_size=16,
            epochs=1000,
            save_step=1000,
            checkpoint_interval=1000,
            print_step=50,
            print_eval=True,
            mixed_precision=True,
            max_seq_len=Coqpit.default_max_seq_len * 2,  # Secuencias más largas
            output_path=OUTPUT_PATH,
            datasets=[dataset_config],
            # Configuración específica para español
            text_cleaner="spanish_cleaners",     # Limpiador específico para español
            use_phonemes=True,                   # Mejor pronunciación
            phoneme_language=language,
            phoneme_cache_path=os.path.join(OUTPUT_PATH, "phoneme_cache"),
            # Optimización para voz masculina
            characters=CharactersConfig(
                characters="abcdefghijklmnopqrstuvwxyzáéíóúñü¿¡",
                punctuations="!(),-.:;? ",
                phonemes="ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ",
            ),
            # Parámetros específicos para voces masculinas
            run_name=f"vits_{language}_male_voice",
            run_description="Modelo VITS optimizado para voz masculina en español",
        ), Vits
    
    else:
        # Default to GlowTTS
        print("⚠️ Modelo no reconocido, usando GlowTTS por defecto")
        return get_model_config("glowtts", language)

# ==================================================
# CONFIGURACIÓN DEL MODELO - PERSONALIZABLE
# ==================================================

# Obtener configuración del modelo seleccionado
config, ModelClass = get_model_config(MODEL_TYPE, LANGUAGE)

def main():
    """Función principal de entrenamiento"""
    
    # Crear directorio de salida
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    print("🎤 Iniciando entrenamiento de TTS con tu voz...")
    print(f"📁 Dataset: {DATASET_PATH}")
    print(f"💾 Modelos se guardarán en: {OUTPUT_PATH}")
    print(f"🌍 Idioma: {LANGUAGE}")
    print(f"🤖 Modelo seleccionado: {MODEL_TYPE.upper()}")
    
    # Mostrar información del modelo
    model_info = {
        "glowtts": "🟢 Rápido, estable, recomendado para principiantes",
        "vits": "🟢 Mejor calidad, end-to-end, más lento pero mejor resultado"
    }
    print(f"ℹ️ {model_info.get(MODEL_TYPE.lower(), 'Modelo personalizado')}")
    
    # INICIALIZAR PROCESADOR DE AUDIO
    print("🔊 Configurando procesador de audio...")
    ap = AudioProcessor.init_from_config(config)
    
    # INICIALIZAR TOKENIZER
    print("📝 Configurando tokenizer...")
    tokenizer, config = TTSTokenizer.init_from_config(config)
    
    # CARGAR MUESTRAS DE DATOS
    print("📊 Cargando dataset...")
    try:
        train_samples, eval_samples = load_tts_samples(
            dataset_config,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=config.eval_split_size,
        )
        print(f"✅ Muestras de entrenamiento: {len(train_samples)}")
        print(f"✅ Muestras de evaluación: {len(eval_samples)}")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        print("💡 Asegúrate de que:")
        print("   - El directorio del dataset existe")
        print("   - El archivo metadata.txt está en el formato correcto")
        print("   - Los archivos de audio existen en la carpeta wavs/")
        return
    
    # INICIALIZAR MODELO
    print(f"🧠 Inicializando modelo {MODEL_TYPE.upper()}...")
    model = ModelClass(config, ap, tokenizer, speaker_manager=None)
    
    # Contar parámetros del modelo
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 El modelo tiene {total_params:,} parámetros")
    
    # INICIALIZAR TRAINER
    print("🚀 Configurando entrenador...")
    trainer = Trainer(
        TrainerArgs(),
        config,
        OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples
    )
    
    # COMENZAR ENTRENAMIENTO
    print("\n" + "="*50)
    print("🎯 INICIANDO ENTRENAMIENTO")
    print("="*50)
    print("💡 Consejos durante el entrenamiento:")
    print("   - Monitorea la pérdida (loss) - debe disminuir gradualmente")
    print("   - El entrenamiento puede tomar horas/días dependiendo del dataset")
    print("   - Puedes detener con Ctrl+C y reanudar después")
    print("   - Los checkpoints se guardan automáticamente")
    print()
    
    try:
        trainer.fit()
        print("\n🎉 ¡Entrenamiento completado exitosamente!")
        print(f"📁 Modelo final guardado en: {OUTPUT_PATH}")
    except KeyboardInterrupt:
        print("\n⏸️ Entrenamiento detenido por el usuario")
        print(f"📁 Progreso guardado en: {OUTPUT_PATH}")
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")

if __name__ == "__main__":
    main()