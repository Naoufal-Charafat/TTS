# 🎤 Guía Completa: Entrenar TTS con Tu Propia Voz

Esta guía te ayudará a entrenar un modelo de síntesis de voz (TTS) con tu propia voz usando el framework 🐸TTS.

**🎯 OPTIMIZADA PARA VOZ MASCULINA EN ESPAÑOL**  
*Basada en análisis de rendimiento de modelos TTS*

## 📋 Índice
1. [Instalación Rápida](#instalación-rápida)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Preparación del Dataset](#preparación-del-dataset)
4. [Entrenamiento Paso a Paso](#entrenamiento-paso-a-paso)
5. [Generar Audio con Tu Modelo](#generar-audio-con-tu-modelo)
6. [Usar Modelos Pre-entrenados](#usar-modelos-pre-entrenados)
7. [Configuración Avanzada](#configuración-avanzada)
8. [Solución de Problemas](#solución-de-problemas)

## 🚀 Instalación Rápida

### Paso 1: Instalar Dependencias
```bash
# Instalar todas las dependencias optimizadas
python instalar_dependencias.py

# Verificar que todo funciona
python test_entorno.py
```

### Paso 2: Probar TTS
```bash
# Demo con modelo español masculino
python demo_tts_simple.py
```

## 🖥️ Requisitos del Sistema

### Hardware Mínimo:
- **GPU**: NVIDIA con al menos 6GB VRAM (recomendado: 8GB+)
- **RAM**: 16GB (recomendado: 32GB+)
- **Almacenamiento**: 50GB libres
- **CPU**: 4+ cores

### Software:
- Python 3.8+
- CUDA 11.0+ (para GPU)
- FFmpeg (para procesamiento de audio)

## 📊 Preparación del Dataset

### 1. Requisitos del Audio para Voz Masculina

#### 🎤 Duración Mínima Requerida:
- **Mínimo absoluto**: 30 minutos de audio limpio
- **Recomendado para calidad básica**: 2-3 horas 
- **Ideal para alta calidad**: 5-10 horas
- **Profesional**: 10+ horas

#### 📁 Formato de Audio Obligatorio:
- **Formato**: WAV (sin compresión)
- **Sample Rate**: 22050 Hz (recomendado para voces masculinas)
- **Bit Depth**: 16-bit mínimo (24-bit ideal)
- **Canales**: Mono (1 canal)
- **Duración por clip**: 3-7 segundos (ideal para español)

#### 🎯 Especificaciones para Voz Masculina:
- **Frecuencia fundamental**: 85-180 Hz (típico hombre)
- **Tono**: Consistente y natural
- **Proyección**: Clara y bien articulada
- **Ambiente**: Silencioso, sin eco ni reverberación

#### Calidad de Grabación:
- **Ambiente**: Silencioso, sin eco
- **Micrófono**: Calidad constante
- **Voz**: Tono consistente, pronunciación clara
- **Ruido**: Mínimo ruido de fondo

### 2. Estructura del Dataset

```
MiDatasetTTS/
├── metadata.txt          # Archivo de metadatos
└── wavs/                 # Carpeta con archivos de audio
    ├── audio001.wav
    ├── audio002.wav
    ├── audio003.wav
    └── ...
```

### 3. Formato del Archivo metadata.txt

Cada línea debe tener el formato: `nombre_archivo|texto_original|texto_normalizado`

```
audio001|Hola, ¿cómo estás hoy?|Hola, cómo estás hoy?
audio002|Los números 1, 2, 3|Los números uno, dos, tres
audio003|¡Excelente día!|Excelente día
audio004|El precio es $50.25|El precio es cincuenta dólares veinticinco centavos
```

### 4. Consejos para un Buen Dataset

#### Variedad de Contenido:
- **Oraciones cortas y largas**
- **Diferentes emociones** (neutral, alegre, serio)
- **Variedad fonética** (cubrir todos los sonidos del idioma)
- **Números, fechas, abreviaciones**
- **Signos de puntuación**

#### Consistencia:
- **Misma persona** grabando todo
- **Mismo micrófono** y configuración
- **Mismo ambiente** de grabación
- **Velocidad de habla** consistente

## ⚙️ Configuración del Entrenamiento

### Modelos Recomendados (Basado en Análisis de Rendimiento):

#### 1. **VITS** 🟢 (MEJOR PARA ESPAÑOL MASCULINO)
- **Pros**: Máxima calidad, end-to-end, no requiere vocoder
- **Contras**: Requiere más GPU, entrenamiento más largo
- **Tiempo de entrenamiento**: 12-24 horas
- **GPU requerida**: 8GB+ VRAM
- **Script**: `train_mi_voz.py` (configurado por defecto)

#### 2. **GlowTTS** 🟢 (RECOMENDADO PARA PRINCIPIANTES)
- **Pros**: Estable, rápido entrenamiento, menor uso de memoria
- **Contras**: Requiere modelo de vocoder separado
- **Tiempo de entrenamiento**: 6-12 horas
- **GPU requerida**: 6GB+ VRAM
- **Script**: `train_mi_voz.py` (cambiar MODEL_TYPE = "glowtts")

#### 3. **XTTS v2** 🟢 (MEJOR PARA CLONACIÓN)
- **Pros**: Clonación con pocos datos, multilingüe
- **Contras**: Más complejo de configurar
- **Tiempo de entrenamiento**: 2-6 horas
- **Datos requeridos**: 30 minutos - 2 horas
- **Ideal para**: Clonación rápida de voz masculina

### Configuraciones Personalizables:

```python
# En train_mi_voz.py
config = GlowTTSConfig(
    # RENDIMIENTO
    batch_size=16,           # Reduce si tienes poca memoria GPU
    num_loader_workers=2,    # Ajusta según tus CPU cores
    
    # ENTRENAMIENTO  
    epochs=1000,             # Más épocas = mejor calidad
    mixed_precision=True,    # Ahorra memoria GPU
    
    # AUDIO
    sample_rate=22050,       # Calidad del audio
    hop_length=256,          # Resolución temporal
    
    # TEXTO
    use_phonemes=True,       # Mejora pronunciación
    phoneme_language="es",   # Cambia según tu idioma
    
    # GUARDADO
    save_step=1000,          # Frecuencia de guardado
    print_step=25,           # Frecuencia de logs
)
```

## 🏋️ Entrenamiento Paso a Paso

### Paso 1: Preparar Tu Dataset
```bash
# Crear estructura y procesar audio
python preparar_dataset.py

# Esto creará:
# MiDatasetTTS/
# ├── metadata.txt
# └── wavs/
#     ├── audio001.wav
#     ├── audio002.wav
#     └── ...
```

### Paso 2: Configurar el Modelo
Edita `train_mi_voz.py` para ajustar:
```python
# Cambiar la ruta de tu dataset
DATASET_PATH = "/ruta/a/tu/dataset"

# Seleccionar modelo (recomendado VITS para voz masculina)
MODEL_TYPE = "vits"  # Mejor calidad para español masculino

# Configurar idioma
LANGUAGE = "es"
VOICE_GENDER = "male"
```

### Paso 3: Iniciar Entrenamiento
```bash
# Entrenar con GPU (recomendado)
CUDA_VISIBLE_DEVICES=0 python train_mi_voz.py

# Si no tienes GPU (muy lento)
python train_mi_voz.py

# Con múltiples GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m trainer.distribute --script train_mi_voz.py
```

### Paso 4: Monitorear el Progreso
Durante el entrenamiento verás:
```
> STEP: 500/50000 -- GLOBAL_STEP: 500
| > loss: 1.2456          # Debe disminuir gradualmente
| > current_lr: 2.5e-04    # Tasa de aprendizaje
| > step_time: 3.21        # Tiempo por paso
| > Memory: 6.2GB/8.0GB    # Uso de GPU
```

### Paso 5: Continuar Entrenamiento (si se interrumpe)
```bash
# Reanudar desde el último checkpoint
python train_mi_voz.py --continue_path ./models_entrenados/checkpoint_latest.pth
```

## 🎵 Generar Audio con Tu Modelo

### Opción 1: Script Interactivo (Recomendado)
```bash
# Usar tu modelo entrenado de forma interactiva
python usar_mi_modelo.py

# El script te pedirá:
# 1. Escribe el texto a convertir
# 2. Especifica el archivo de salida
# 3. ¡Escucha tu voz sintética!
```

### Opción 2: Comando Directo
```bash
# Generar audio específico
python -c "
from usar_mi_modelo import MiTTSPersonalizado
tts = MiTTSPersonalizado()
tts.text_to_speech('Hola, esta es mi voz sintética masculina', 'mi_audio.wav')
print('Audio generado: mi_audio.wav')
"
```

### Opción 3: API Programática
```python
# En tu propio script Python
from usar_mi_modelo import MiTTSPersonalizado

# Inicializar con tu modelo
tts = MiTTSPersonalizado("/ruta/a/tu/modelo")

# Generar audio
texto = "Hola mundo, soy una voz sintética masculina en español"
tts.text_to_speech(texto, "salida.wav")
```

## 🇪🇸 Usar Modelos Pre-entrenados en Español Masculino

### Listar Modelos Disponibles
```bash
# Ver todos los modelos disponibles
python -c "
from TTS.api import TTS
print('🎤 MODELOS TTS DISPONIBLES:')
print('=' * 40)
models = TTS.list_models()
spanish_models = [m for m in models if 'es' in m.lower() or 'spanish' in m.lower()]
for i, model in enumerate(spanish_models, 1):
    print(f'{i}. {model}')
"
```

### Modelo Recomendado: VITS Español
```bash
# Usar el mejor modelo para español masculino
python -c "
from TTS.api import TTS

# Modelo VITS español (mejor calidad según análisis)
tts = TTS('tts_models/es/css10/vits')

# Generar audio
texto = 'Hola, soy una voz masculina en español de alta calidad'
tts.tts_to_file(text=texto, file_path='voz_masculina_es.wav')
print('Audio generado: voz_masculina_es.wav')
"
```

### Comparar Diferentes Modelos
```bash
# Probar múltiples modelos españoles
python comparar_modelos.py

# Este script generará audio con diferentes modelos para comparar
```

### Comando Rápido para Generar Audio
```bash
# Una línea para generar audio con el mejor modelo español
python -c "from TTS.api import TTS; TTS('tts_models/es/css10/vits').tts_to_file('Tu texto aquí', 'output.wav')"
```

## 📱 Comandos Útiles de Referencia

### Generar Audio desde Texto (Tu Modelo)
```bash
# Método 1: Interactivo
python usar_mi_modelo.py

# Método 2: Directo
python -c "
from usar_mi_modelo import MiTTSPersonalizado
MiTTSPersonalizado().text_to_speech('Tu texto', 'salida.wav')
"
```

### Generar Audio con Modelo Pre-entrenado
```bash
# Mejor modelo español masculino (VITS)
python -c "
from TTS.api import TTS
TTS('tts_models/es/css10/vits').tts_to_file('Tu texto aquí', 'audio.wav')
"

# Modelo alternativo (Tacotron2)
python -c "
from TTS.api import TTS
TTS('tts_models/es/mai/tacotron2-DDC').tts_to_file('Tu texto aquí', 'audio2.wav')
"TS
```

### Ver Estado del Entrenamiento
```bash
# Monitorear progreso del entrenamiento
tensorboard --logdir=./models_entrenados/

# Ver logs del entrenamiento
tail -f ./models_entrenados/train_log.txt
```

### Listar Todos los Modelos TTS
```bash
# Lista completa de modelos
python -c "
from TTS.api import TTS
import json
models = TTS.list_models()
print(json.dumps(models, indent=2, ensure_ascii=False))
"

# Solo modelos en español
python -c "
from TTS.api import TTS
models = TTS.list_models()
spanish = [m for m in models if 'es' in m or 'spanish' in m.lower()]
for model in spanish: print(model)
"
```

## 📋 Resumen de Comandos Esenciales

### 🚀 Instalación y Setup Inicial
```bash
# 1. Instalar dependencias
python instalar_dependencias.py

# 2. Verificar instalación
python test_entorno.py

# 3. Probar demo
python demo_tts_simple.py
```

### � Entrenar Tu Modelo de Voz Masculina
```bash
# 1. Preparar dataset (mínimo 30 min de audio WAV)
python preparar_dataset.py

# 2. Entrenar modelo VITS (recomendado)
CUDA_VISIBLE_DEVICES=0 python train_mi_voz.py

# 3. Continuar entrenamiento si se interrumpe
python train_mi_voz.py --continue_path ./models_entrenados/checkpoint_latest.pth
```

### 🎵 Generar Audio con Tu Modelo
```bash
# Método interactivo (recomendado)
python usar_mi_modelo.py

# Método directo
python -c "from usar_mi_modelo import MiTTSPersonalizado; MiTTSPersonalizado().text_to_speech('Tu texto aquí', 'mi_voz.wav')"
```

### 🇪🇸 Usar Mejores Modelos Pre-entrenados
```bash
# Listar modelos disponibles
python -c "from TTS.api import TTS; [print(m) for m in TTS.list_models() if 'es' in m]"

# Mejor modelo español masculino (VITS)
python -c "from TTS.api import TTS; TTS('tts_models/es/css10/vits').tts_to_file('Tu texto aquí', 'voz_es.wav')"

# Comparar modelos
python comparar_modelos.py
```

### 📊 Monitoreo y Debugging
```bash
# Ver progreso del entrenamiento
tensorboard --logdir=./models_entrenados/

# Verificar estado del sistema
python test_entorno.py

# Ver logs de entrenamiento
tail -f ./models_entrenados/train_log.txt
```

### 1. Aspectos Personalizables del Modelo

#### **Arquitectura del Modelo:**
- **Número de capas** en encoder/decoder
- **Dimensión de embedding**
- **Número de attention heads**
- **Tamaño de hidden layers**

#### **Procesamiento de Audio:**
- **Sample rate** (16kHz, 22kHz, 44kHz)
- **Hop length** (resolución temporal)
- **Win length** (ventana de análisis)
- **N° de mel bands** (resolución frecuencial)
- **F0 range** (rango de pitch)

#### **Procesamiento de Texto:**
- **Limpieza de texto** (english_cleaners, spanish_cleaners)
- **Uso de fonemas** (mejora pronunciación)
- **Idioma de fonemas** (es, en, fr, etc.)
- **Normalización** de números, fechas, etc.

#### **Entrenamiento:**
- **Learning rate** y scheduler
- **Batch size** (afecta velocidad y memoria)
- **Épocas** y early stopping
- **Data augmentation**
- **Regularización** (dropout, weight decay)

### 2. Configuración Avanzada de GlowTTS

```python
config = GlowTTSConfig(
    # ARQUITECTURA
    hidden_channels=192,         # Canales ocultos
    filter_channels=768,         # Canales de filtros  
    filter_channels_dp=256,      # Canales duration predictor
    out_channels=80,             # Salida (mel bands)
    
    # FLUJO NORMALIZADO
    num_flow_blocks_dec=12,      # Bloques en decoder
    kernel_size_dec=5,           # Tamaño kernel decoder
    dilation_rate=1,             # Tasa de dilatación
    num_block_layers=4,          # Capas por bloque
    
    # ATTENTION
    num_heads=2,                 # Attention heads
    hidden_channels_enc=192,     # Canales encoder
    hidden_channels_dec=192,     # Canales decoder
    
    # DURACIÓN
    kernel_size_dp=3,            # Kernel duration predictor
    dropout_p_dp=0.1,            # Dropout duration predictor
    
    # OPTIMIZACIÓN
    optimizer="AdamW",           # Optimizador
    lr=1e-3,                     # Learning rate inicial
    lr_scheduler="ExponentialLR", # Scheduler
    lr_decay=0.999,              # Decay rate
    weight_decay=1e-6,           # Regularización L2
    
    # TRAINING
    grad_clip=5.0,               # Gradient clipping
    mixed_precision=True,        # Precisión mixta FP16
    
    # EVALUACIÓN
    test_sentences=[             # Frases de prueba
        "Hola, este es un test de mi voz sintética.",
        "Los números uno, dos, tres suenan bien.",
        "¿Cómo te encuentras hoy?"
    ]
)
```

### 3. Personalización para Diferentes Idiomas

#### Español:
```python
config.text_cleaner = "spanish_cleaners"
config.phoneme_language = "es"
config.characters = "abcdefghijklmnopqrstuvwxyzáéíóúñü.,!?¡¿"
```

#### Inglés:
```python
config.text_cleaner = "english_cleaners"
config.phoneme_language = "en-us"
```

#### Multilingüe:
```python
config.text_cleaner = "multilingual_cleaners"
config.phoneme_language = "es"  # Idioma principal
config.use_phonemes = True
```

### 4. Optimización por Hardware

#### GPU con poca memoria (<8GB):
```python
config.batch_size = 8
config.eval_batch_size = 4
config.mixed_precision = True
config.num_loader_workers = 1
```

#### GPU potente (>16GB):
```python
config.batch_size = 64
config.eval_batch_size = 32
config.num_loader_workers = 8
config.accumulate_grad_batches = 1
```

#### Solo CPU:
```python
config.batch_size = 4
config.mixed_precision = False
config.num_loader_workers = 2
# Agregar en train script:
# model.cpu()
```

## 🔧 Solución de Problemas

### Problemas Comunes:

#### 1. **"CUDA out of memory"**
**Solución**:
- Reducir `batch_size` a 8 o menos
- Activar `mixed_precision=True`
- Reducir `num_loader_workers`

#### 2. **"FileNotFoundError: metadata.txt"**
**Solución**:
- Verificar que el archivo existe en la ruta correcta
- Verificar formato del archivo (UTF-8)
- Ejecutar `preparar_dataset.py`

#### 3. **"Audio files not found"**
**Solución**:
- Verificar nombres en metadata.txt coinciden con archivos
- Verificar que archivos estén en carpeta `wavs/`
- Verificar extensión `.wav`

#### 4. **Mala calidad de síntesis**
**Solución**:
- Aumentar cantidad de datos de entrenamiento
- Verificar calidad del dataset original
- Entrenar más épocas
- Ajustar parámetros de audio (`sample_rate`, `hop_length`)

#### 5. **Entrenamiento muy lento**
**Solución**:
- Verificar que está usando GPU: `torch.cuda.is_available()`
- Aumentar `num_loader_workers`
- Usar `mixed_precision=True`
- Reducir `print_step` frecuencia

### Tips para Mejor Calidad:

1. **Dataset**: Más datos = mejor calidad
2. **Consistencia**: Mismas condiciones de grabación
3. **Limpieza**: Remover clips con ruido/errores
4. **Paciencia**: El entrenamiento toma tiempo
5. **Monitoreo**: Vigilar métricas de pérdida

## 📈 Evaluación del Modelo

### Métricas a Monitorear:

1. **Training Loss**: Debe disminuir consistentemente
2. **Validation Loss**: No debe aumentar (overfitting)
3. **Duration Loss**: Precisión en duración de fonemas
4. **Alignment**: Calidad del alineamiento texto-audio

### Pruebas de Calidad:

```python
# Frases de prueba variadas
test_sentences = [
    "Frase corta.",
    "Esta es una oración más larga para probar la capacidad del modelo.",
    "Números: uno, dos, tres, mil doscientos.",
    "¿Cómo maneja las preguntas el modelo?",
    "¡Exclamaciones y emociones!"
]
```

## 🚀 Siguientes Pasos

1. **Experimenta** con diferentes configuraciones
2. **Recolecta más datos** para mejorar calidad
3. **Prueba diferentes modelos** (VITS, YourTTS)
4. **Optimiza** para tu caso de uso específico
5. **Integra** en tus aplicaciones

---

## 📞 Soporte

Si tienes problemas:
1. Revisa esta guía
2. Consulta la documentación oficial de 🐸TTS
3. Ejecuta `preparar_dataset.py` para verificar tu dataset
4. Usa los scripts de ejemplo proporcionados

¡Buena suerte entrenando tu modelo de voz personalizado! 🎤✨

---

## 📋 Resumen de Comandos Esenciales

### 🚀 Instalación y Setup Inicial
```bash
# 1. Instalar dependencias
python instalar_dependencias.py

# 2. Verificar instalación
python test_entorno.py

# 3. Probar demo
python demo_tts_simple.py
```

### 🎤 Entrenar Tu Modelo de Voz Masculina
```bash
# 1. Preparar dataset (mínimo 30 min de audio WAV)
python preparar_dataset.py

# 2. Entrenar modelo VITS (recomendado)
CUDA_VISIBLE_DEVICES=0 python train_mi_voz.py

# 3. Continuar entrenamiento si se interrumpe
python train_mi_voz.py --continue_path ./models_entrenados/checkpoint_latest.pth
```

### 🎵 Generar Audio con Tu Modelo
```bash
# Método interactivo (recomendado)
python usar_mi_modelo.py

# Método directo
python -c "from usar_mi_modelo import MiTTSPersonalizado; MiTTSPersonalizado().text_to_speech('Tu texto aquí', 'mi_voz.wav')"
```

### 🇪🇸 Usar Mejores Modelos Pre-entrenados
```bash
# Listar modelos disponibles
python -c "from TTS.api import TTS; [print(m) for m in TTS.list_models() if 'es' in m]"

# Mejor modelo español masculino (VITS)
python -c "from TTS.api import TTS; TTS('tts_models/es/css10/vits').tts_to_file('Tu texto aquí', 'voz_es.wav')"

# Comparar modelos
python comparar_modelos.py
```

### 📊 Monitoreo y Debugging
```bash
# Ver progreso del entrenamiento
tensorboard --logdir=./models_entrenados/

# Verificar estado del sistema
python test_entorno.py

# Ver logs de entrenamiento
tail -f ./models_entrenados/train_log.txt
```

## 🎯 Optimización para Hardware Específico

### GPU con 8GB+ VRAM (Recomendado):
```python
# En train_mi_voz.py
MODEL_TYPE = "vits"
batch_size = 32
mixed_precision = True
```

### GPU con 6-8GB VRAM:
```python
MODEL_TYPE = "vits"
batch_size = 16
mixed_precision = True
```

### GPU con <6GB VRAM:
```python
MODEL_TYPE = "glowtts"
batch_size = 8
mixed_precision = True
```

### Solo CPU (muy lento):
```python
MODEL_TYPE = "glowtts"
batch_size = 4
mixed_precision = False
```