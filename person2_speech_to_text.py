"""
============================================================
  MODULE 2 — SPEECH TO TEXT  (Person 2)
  Captures microphone audio in real time and transcribes
  it using OpenAI Whisper (runs fully offline, no API key).

  INSTALL (run this once):
      pip install openai-whisper sounddevice numpy scipy

  On Windows you also need:
      pip install pyaudio
  On Linux:
      sudo apt-get install portaudio19-dev && pip install pyaudio

  HOW TO RUN:
      python person2_speech_to_text.py

  OUTPUT:
      - Live transcription printed to terminal
      - Writes current transcript to  `speech_output.txt`
        (Person 3 reads this to speak it aloud;
         Person 4's GUI displays it)

  CONTROLS:
      Ctrl-C  = stop
============================================================
"""

import time
import threading
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
import whisper
from scipy.io.wavfile import write as wav_write

# ─── config ───────────────────────────────────────────────
OUTPUT_FILE    = "speech_output.txt"   # shared with Person 3 & 4
SAMPLE_RATE    = 16_000                # Hz — Whisper expects 16 kHz
CHUNK_SECONDS  = 4                     # record N seconds, then transcribe
WHISPER_MODEL  = "base"                # tiny / base / small / medium / large
                                       # 'base' is best speed/accuracy balance
                                       # for a demo on a laptop

# ──────────────────────────────────────────────────────────
audio_queue: queue.Queue = queue.Queue()
full_transcript: list[str] = []


def start_in_background(callback=None):
    """Start Whisper in the background and optionally stream each new phrase."""
    print(f"[INFO] Loading Whisper '{WHISPER_MODEL}' model …")
    model = whisper.load_model(WHISPER_MODEL)
    print("[INFO] Model loaded.\n")

    Path(OUTPUT_FILE).write_text("")

    stop_event = threading.Event()

    def audio_capture_worker():
        while not stop_event.is_set():
            recording = sd.rec(
                int(CHUNK_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )
            sd.wait()
            audio_queue.put(recording.copy())

    def transcription_worker(model):
        tmp_wav = Path("_tmp_chunk.wav")
        while not stop_event.is_set():
            try:
                audio = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            audio_int16 = (audio * 32767).astype(np.int16)
            wav_write(str(tmp_wav), SAMPLE_RATE, audio_int16)

            result = model.transcribe(
                str(tmp_wav),
                language="en",
                fp16=False,
                condition_on_previous_text=False,
            )
            text = result["text"].strip()

            if text and text not in ("", "you", "."):
                full_transcript.append(text)
                combined = " ".join(full_transcript)
                Path(OUTPUT_FILE).write_text(combined)
                if callback:
                    callback(text)
                print(f"  [SPEECH] {text}")
                print(f"  [FULL ] {combined[-120:]}\n")

            audio_queue.task_done()

    t_capture = threading.Thread(target=audio_capture_worker, daemon=True)
    t_transcribe = threading.Thread(target=transcription_worker, args=(model,), daemon=True)
    t_capture.start()
    t_transcribe.start()

    return {"stop": stop_event.set, "threads": (t_capture, t_transcribe)}


# ══════════════════════════════════════════════════════════
#  AUDIO CAPTURE THREAD
# ══════════════════════════════════════════════════════════
def audio_capture_worker():
    """
    Continuously records CHUNK_SECONDS of audio from the
    default microphone and pushes raw numpy arrays onto the
    audio_queue for the transcription worker to consume.
    """
    print(f"[INFO] Audio capture started  (chunk = {CHUNK_SECONDS}s, "
          f"rate = {SAMPLE_RATE} Hz)")
    while True:
        recording = sd.rec(
            int(CHUNK_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        audio_queue.put(recording.copy())


# ══════════════════════════════════════════════════════════
#  TRANSCRIPTION THREAD
# ══════════════════════════════════════════════════════════
def transcription_worker(model):
    """
    Pulls audio chunks from the queue, runs Whisper inference,
    appends new text to the shared output file.
    """
    tmp_wav = Path("_tmp_chunk.wav")
    print("[INFO] Transcription worker ready.  Speak now …\n")

    while True:
        audio = audio_queue.get()                  # blocks until chunk ready

        # Whisper needs a 16-bit PCM wav file (or a raw float32 array)
        audio_int16 = (audio * 32767).astype(np.int16)
        wav_write(str(tmp_wav), SAMPLE_RATE, audio_int16)

        result = model.transcribe(
            str(tmp_wav),
            language="en",
            fp16=False,               # set True if you have a CUDA GPU
            condition_on_previous_text=False
        )
        text = result["text"].strip()

        if text and text not in ("", "you", "."):  # filter whisper artefacts
            full_transcript.append(text)
            combined = " ".join(full_transcript)
            Path(OUTPUT_FILE).write_text(combined)

            print(f"  [SPEECH] {text}")
            print(f"  [FULL ] {combined[-120:]}\n")

        audio_queue.task_done()


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def run_speech_to_text():
    print(f"[INFO] Loading Whisper '{WHISPER_MODEL}' model …")
    print("       (first run downloads the model weights — ~140 MB for 'base')")
    model = whisper.load_model(WHISPER_MODEL)
    print("[INFO] Model loaded.\n")

    # initialise output file
    Path(OUTPUT_FILE).write_text("")

    # launch threads
    t_capture = threading.Thread(target=audio_capture_worker, daemon=True)
    t_transcribe = threading.Thread(
        target=transcription_worker, args=(model,), daemon=True)

    t_capture.start()
    t_transcribe.start()

    print("=" * 55)
    print("  Speech-to-Text running.  Press Ctrl-C to stop.")
    print("=" * 55)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
        combined = " ".join(full_transcript)
        print(f"\n[FINAL TRANSCRIPT]\n{combined}")
        Path(OUTPUT_FILE).write_text(combined)


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_speech_to_text()
