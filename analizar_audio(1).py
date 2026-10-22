#!/usr/bin/env python3
"""
Analizador experimental multilingüe de voz y contenido.

Extrae características acústicas, transcribe audio con Whisper y genera un
informe exploratorio. NO determina si una persona miente; solo señala
características y posibles inconsistencias para revisión humana.

Instalación:
    pip install numpy pandas librosa soundfile faster-whisper

Uso:
    python analizar_audio.py grabacion.wav --idioma auto --salida informe.json

Para forzar un idioma, por ejemplo:
    python analizar_audio.py grabacion.wav --idioma es --salida informe.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import librosa
from faster_whisper import WhisperModel


def extraer_caracteristicas(ruta_audio: str) -> dict:
    audio, sr = librosa.load(ruta_audio, sr=16000, mono=True)
    if audio.size == 0:
        raise ValueError("El archivo de audio está vacío o no se pudo leer.")

    duracion = float(len(audio) / sr)
    rms = librosa.feature.rms(y=audio)[0]
    f0 = librosa.yin(audio, fmin=65, fmax=400, sr=sr)
    f0 = f0[np.isfinite(f0)]
    f0 = f0[(f0 >= 65) & (f0 <= 400)]

    # Estimación aproximada de pausas usando energía en ventanas cortas.
    hop = 160  # 10 ms a 16 kHz
    frame_rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=hop)[0]
    umbral = max(float(np.percentile(frame_rms, 20)), 0.008)
    silencioso = frame_rms < umbral
    cambios = np.diff(np.r_[False, silencioso, False].astype(int))
    inicios = np.where(cambios == 1)[0]
    finales = np.where(cambios == -1)[0]
    pausas = [(b - a) * hop / sr for a, b in zip(inicios, finales)]
    pausas_largas = [p for p in pausas if p >= 0.25]

    return {
        "duracion_segundos": round(duracion, 3),
        "tono_medio_hz_estimado": round(float(np.median(f0)), 2) if f0.size else None,
        "variacion_tono_hz_aprox": round(float(np.std(f0)), 2) if f0.size else None,
        "intensidad_rms_media": round(float(np.mean(rms)), 5),
        "numero_pausas_estimadas_250ms_o_mas": len(pausas_largas),
        "duracion_total_pausas_estimadas_segundos": round(float(sum(pausas_largas)), 3),
        "nota": "Las pausas y el tono son estimaciones acústicas; dependen del audio y no indican engaño."
    }


def transcribir(ruta_audio: str, idioma: str):
    # Modelo multilingüe pequeño; la primera ejecución descarga el modelo.
    modelo = WhisperModel("small", device="cpu", compute_type="int8")
    lang = None if idioma.lower() in ("auto", "detectar") else idioma
    segmentos, info = modelo.transcribe(
        ruta_audio,
        language=lang,
        beam_size=5,
        vad_filter=True
    )
    partes = []
    for s in segmentos:
        partes.append({
            "inicio_segundos": round(float(s.start), 2),
            "fin_segundos": round(float(s.end), 2),
            "texto": s.text.strip()
        })
    return " ".join(p["texto"] for p in partes), partes, info.language, info.language_probability


def buscar_repeticiones_y_numeros(texto: str) -> dict:
    # Señala elementos verificables, no contradicciones definitivas.
    numeros = re.findall(r"\b\d+(?:[.,]\d+)?\b", texto)
    fechas_horas = re.findall(
        r"\b(?:hoy|ayer|mañana|lunes|martes|miércoles|jueves|viernes|sábado|domingo|"
        r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|"
        r"\d{1,2}:\d{2})\b",
        texto,
        flags=re.IGNORECASE
    )
    return {
        "numeros_mencionados_para_verificacion": numeros,
        "fechas_horas_o_referencias_temporales": fechas_horas,
        "nota": "Estos elementos requieren contexto o corroboración externa; no son prueba de falsedad."
    }


def main():
    parser = argparse.ArgumentParser(description="Analiza audio y transcribe habla multilingüe.")
    parser.add_argument("audio", help="Ruta a un archivo WAV, MP3, M4A u otro formato compatible.")
    parser.add_argument("--idioma", default="auto", help="Código de idioma Whisper (ej. es, en, fr) o auto.")
    parser.add_argument("--salida", default="informe_voz.json", help="Ruta del informe JSON.")
    args = parser.parse_args()

    ruta = Path(args.audio)
    if not ruta.is_file():
        raise SystemExit(f"No se encuentra el archivo: {ruta}")

    caracteristicas = extraer_caracteristicas(str(ruta))
    texto, segmentos, idioma_detectado, confianza_idioma = transcribir(str(ruta), args.idioma)

    informe = {
        "archivo": ruta.name,
        "idioma_detectado": idioma_detectado,
        "probabilidad_idioma_estimada": round(float(confianza_idioma), 3),
        "transcripcion": texto,
        "segmentos": segmentos,
        "caracteristicas_acusticas": caracteristicas,
        "elementos_para_verificar_en_el_contenido": buscar_repeticiones_y_numeros(texto),
        "limitaciones": [
            "Este programa no detecta ni demuestra mentiras.",
            "Las pausas, el tono y otros rasgos varían por persona, idioma, emoción, salud y condiciones de grabación.",
            "La transcripción automática puede contener errores; revisa el texto antes de interpretarlo.",
            "No uses el informe como única base para acusaciones o decisiones importantes."
        ]
    }

    salida = Path(args.salida)
    salida.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Informe guardado en: {salida.resolve()}")
    print(f"Idioma detectado: {idioma_detectado}")
    print(f"Transcripción: {texto}")
    print("Recuerda: las señales acústicas o lingüísticas no prueban que alguien mienta.")


if __name__ == "__main__":
    main()
