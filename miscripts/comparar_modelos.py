#!/usr/bin/env python3
"""
Script para            "multilingüe": [
                {
                    "nombre": "tts_models/multilingual/multi-dataset/xtts_v2",
                    "descripcion": "🟢 XTTS v2 - MEJOR para clonación de voz masculina",
                    "tipo": "multi_speaker",
                    "calidad": "muy_alta",
                    "velocidad": "rapida",
                    "requiere_speaker": True,
                    "optimizado_masculino": True,
                    "idiomas": ["es", "en", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn"]
                },
                {
                    "nombre": "tts_models/multilingual/multi-dataset/your_tts",
                    "descripcion": "🟡 YourTTS - Multi-speaker con clonación",
                    "tipo": "multi_speaker",
                    "calidad": "alta",
                    "velocidad": "media",
                    "requiere_speaker": True
                }
            ],
            "entrenamiento": [
                {
                    "nombre": "VITS",
                    "descripcion": "🟢 Mejor opción para entrenar voz masculina en español",
                    "recomendado_para": "Máxima calidad, voz única",
                    "tiempo_entrenamiento": "12-24 horas",
                    "datos_requeridos": "10+ horas de audio limpio"
                },
                {
                    "nombre": "GlowTTS", 
                    "descripcion": "🟢 Mejor para principiantes",
                    "recomendado_para": "Aprendizaje rápido, estable",
                    "tiempo_entrenamiento": "6-12 horas",
                    "datos_requeridos": "5+ horas de audio"
                },
                {
                    "nombre": "XTTS v2",
                    "descripcion": "🟢 Ideal para clonación con pocos datos",
                    "recomendado_para": "Clonación rápida de voz masculina",
                    "tiempo_entrenamiento": "2-6 horas",
                    "datos_requeridos": "30 minutos - 2 horas"
                }
            ] diferentes modelos TTS preentrenados
Basado en el análisis de rendimiento de 🐸TTS
"""

import os
import torch
import time
from pathlib import Path
from TTS.api import TTS

class ComparadorModelosTTS:
    """Compara diferentes modelos TTS para encontrar el mejor para tu uso"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Usando dispositivo: {self.device}")
        
        # Modelos recomendados basados en el análisis de rendimiento
        self.modelos_recomendados = {
            "español": [
                {
                    "nombre": "tts_models/es/css10/vits",
                    "descripcion": "🟢 VITS en español - MEJOR CALIDAD (Top rendimiento)",
                    "tipo": "single_speaker",
                    "calidad": "muy_alta",
                    "velocidad": "media",
                    "optimizado_masculino": True
                },
                {
                    "nombre": "tts_models/es/mai/tacotron2-DDC",
                    "descripcion": "🟡 Tacotron2 español - Calidad buena, más lento",
                    "tipo": "single_speaker", 
                    "calidad": "media",
                    "velocidad": "lenta"
                }
            ],
            "multilingue": [
                {
                    "nombre": "tts_models/multilingual/multi-dataset/xtts_v2",
                    "descripcion": "🟢 XTTS v2 - 16 idiomas, clonación de voz",
                    "tipo": "multi_speaker",
                    "calidad": "muy_alta",
                    "velocidad": "media",
                    "requiere_speaker": True
                },
                {
                    "nombre": "tts_models/multilingual/multi-dataset/your_tts",
                    "descripcion": "🟢 YourTTS - Multi-speaker, zero-shot",
                    "tipo": "multi_speaker", 
                    "calidad": "alta",
                    "velocidad": "lenta",
                    "requiere_speaker": True
                }
            ],
            "ingles": [
                {
                    "nombre": "tts_models/en/ljspeech/glow-tts",
                    "descripcion": "🟢 GlowTTS - Rápido y confiable",
                    "tipo": "single_speaker",
                    "calidad": "alta",
                    "velocidad": "rapida"
                },
                {
                    "nombre": "tts_models/en/ljspeech/vits",
                    "descripcion": "🟢 VITS - Mejor calidad end-to-end",
                    "tipo": "single_speaker",
                    "calidad": "muy_alta", 
                    "velocidad": "media"
                }
            ]
        }
    
    def listar_modelos_disponibles(self):
        """Lista todos los modelos disponibles"""
        print("📋 MODELOS TTS DISPONIBLES:")
        print("="*50)
        
        try:
            tts = TTS()
            modelos = tts.list_models()
            
            # Categorizar modelos
            categorias = {}
            for modelo in modelos:
                if 'tts_models' in modelo:
                    partes = modelo.split('/')
                    if len(partes) >= 3:
                        idioma = partes[1]
                        if idioma not in categorias:
                            categorias[idioma] = []
                        categorias[idioma].append(modelo)
            
            # Mostrar por categorías
            for idioma, lista_modelos in categorias.items():
                print(f"\n🌍 {idioma.upper()}:")
                for modelo in lista_modelos[:5]:  # Mostrar solo los primeros 5
                    print(f"   📦 {modelo}")
                if len(lista_modelos) > 5:
                    print(f"   ... y {len(lista_modelos) - 5} más")
                    
        except Exception as e:
            print(f"❌ Error listando modelos: {e}")
    
    def probar_modelo(self, nombre_modelo, texto_prueba, output_file, speaker_wav=None):
        """Prueba un modelo específico"""
        try:
            print(f"\n🔄 Probando: {nombre_modelo}")
            inicio = time.time()
            
            # Inicializar TTS
            tts = TTS(model_name=nombre_modelo, progress_bar=False).to(self.device)
            
            # Generar audio
            if speaker_wav and os.path.exists(speaker_wav):
                # Modelo con clonación de voz
                tts.tts_to_file(
                    text=texto_prueba,
                    speaker_wav=speaker_wav,
                    language="es",
                    file_path=output_file
                )
            else:
                # Modelo normal
                tts.tts_to_file(text=texto_prueba, file_path=output_file)
            
            tiempo_total = time.time() - inicio
            
            print(f"✅ Audio generado: {output_file}")
            print(f"⏱️ Tiempo: {tiempo_total:.2f} segundos")
            
            return True, tiempo_total
            
        except Exception as e:
            print(f"❌ Error con {nombre_modelo}: {e}")
            return False, 0
    
    def benchmark_modelos(self, texto_prueba="Hola, este es un test de síntesis de voz.", speaker_wav=None):
        """Realiza benchmark de modelos recomendados"""
        print("\n🏆 BENCHMARK DE MODELOS RECOMENDADOS")
        print("="*50)
        
        output_dir = Path("benchmark_resultados")
        output_dir.mkdir(exist_ok=True)
        
        resultados = []
        
        # Probar modelos recomendados
        for categoria, modelos in self.modelos_recomendados.items():
            print(f"\n📂 Categoría: {categoria.upper()}")
            
            for modelo_info in modelos:
                nombre = modelo_info["nombre"]
                descripcion = modelo_info["descripcion"]
                
                print(f"\n{descripcion}")
                
                output_file = output_dir / f"{nombre.replace('/', '_')}.wav"
                
                # Usar speaker_wav solo si el modelo lo requiere
                usar_speaker = speaker_wav if modelo_info.get("requiere_speaker", False) else None
                
                exito, tiempo = self.probar_modelo(
                    nombre, texto_prueba, str(output_file), usar_speaker
                )
                
                if exito:
                    resultados.append({
                        "modelo": nombre,
                        "categoria": categoria,
                        "tiempo": tiempo,
                        "calidad": modelo_info["calidad"],
                        "velocidad": modelo_info["velocidad"],
                        "archivo": output_file
                    })
        
        # Mostrar resumen
        self.mostrar_resumen_benchmark(resultados)
        
        return resultados
    
    def mostrar_resumen_benchmark(self, resultados):
        """Muestra resumen del benchmark"""
        print(f"\n📊 RESUMEN DEL BENCHMARK")
        print("="*60)
        
        if not resultados:
            print("❌ No se pudieron probar modelos")
            return
        
        # Ordenar por tiempo
        resultados_ordenados = sorted(resultados, key=lambda x: x["tiempo"])
        
        print("🏃‍♂️ VELOCIDAD (más rápido primero):")
        for i, resultado in enumerate(resultados_ordenados, 1):
            print(f"{i}. {resultado['modelo']} - {resultado['tiempo']:.2f}s")
        
        print("\n🎯 RECOMENDACIONES:")
        print("- 🚀 Más rápido:", resultados_ordenados[0]["modelo"])
        
        # Recomendar por calidad
        alta_calidad = [r for r in resultados if r["calidad"] in ["muy_alta", "alta"]]
        if alta_calidad:
            print("- 🎵 Mejor calidad:", alta_calidad[0]["modelo"])
        
        print(f"\n📁 Archivos de audio generados en: benchmark_resultados/")
    
    def recomendar_modelo_para_entrenamiento(self):
        """Recomienda el mejor modelo para entrenar desde cero"""
        print("\n🎯 RECOMENDACIONES PARA ENTRENAMIENTO PERSONALIZADO")
        print("="*50)
        
        recomendaciones = {
            "principiante": {
                "modelo": "GlowTTS",
                "razon": "Rápido entrenamiento, estable, buena documentación",
                "tiempo_entrenamiento": "6-12 horas",
                "calidad_esperada": "Alta",
                "dificultad": "⭐⭐☆☆☆"
            },
            "intermedio": {
                "modelo": "VITS", 
                "razon": "End-to-end, excelente calidad, no requiere vocoder separado",
                "tiempo_entrenamiento": "12-24 horas",
                "calidad_esperada": "Muy Alta",
                "dificultad": "⭐⭐⭐☆☆"
            },
            "avanzado": {
                "modelo": "XTTS v2 Fine-tuning",
                "razon": "Clonación de voz, multilingüe, estado del arte",
                "tiempo_entrenamiento": "4-8 horas (fine-tuning)",
                "calidad_esperada": "Excelente",
                "dificultad": "⭐⭐⭐⭐☆"
            }
        }
        
        for nivel, info in recomendaciones.items():
            print(f"\n🎓 {nivel.upper()}:")
            print(f"   📦 Modelo: {info['modelo']}")
            print(f"   💡 Razón: {info['razon']}")
            print(f"   ⏱️ Tiempo: {info['tiempo_entrenamiento']}")
            print(f"   🎵 Calidad: {info['calidad_esperada']}")
            print(f"   🏔️ Dificultad: {info['dificultad']}")

def main():
    """Función principal"""
    print("🔍 COMPARADOR DE MODELOS TTS")
    print("Basado en análisis de rendimiento de 🐸TTS")
    print("="*50)
    
    comparador = ComparadorModelosTTS()
    
    while True:
        print("\n¿Qué quieres hacer?")
        print("1. Ver modelos disponibles")
        print("2. Hacer benchmark de modelos recomendados")
        print("3. Probar un modelo específico")
        print("4. Ver recomendaciones para entrenamiento")
        print("5. Salir")
        
        choice = input("\nElige una opción (1-5): ").strip()
        
        if choice == "1":
            comparador.listar_modelos_disponibles()
            
        elif choice == "2":
            texto = input("Texto de prueba (Enter para usar por defecto): ").strip()
            if not texto:
                texto = "Hola, este es un test de síntesis de voz en español."
            
            speaker_file = input("Archivo de voz para clonación (opcional, Enter para omitir): ").strip()
            if speaker_file and not os.path.exists(speaker_file):
                speaker_file = None
                print("⚠️ Archivo no encontrado, continuando sin clonación de voz")
            
            comparador.benchmark_modelos(texto, speaker_file)
            
        elif choice == "3":
            modelo = input("Nombre del modelo: ").strip()
            texto = input("Texto a sintetizar: ").strip()
            output = input("Archivo de salida (ej: test.wav): ").strip()
            
            if modelo and texto and output:
                comparador.probar_modelo(modelo, texto, output)
            else:
                print("❌ Faltan parámetros")
                
        elif choice == "4":
            comparador.recomendar_modelo_para_entrenamiento()
            
        elif choice == "5":
            print("👋 ¡Hasta luego!")
            break
            
        else:
            print("❌ Opción no válida")

if __name__ == "__main__":
    main()