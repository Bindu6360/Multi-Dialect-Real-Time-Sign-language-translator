"""
=============================================================
  MODULE 3 — TEXT TO SPEECH + CAPTION OVERLAY  (Person 3)
  Reads sign-translated text aloud and shows a floating
  caption window that sits on top of Zoom / Google Meet.
=============================================================

SETUP (run once):
    pip install pyttsx3 pywin32   # Windows
    pip install pyttsx3           # Mac / Linux
    pip install tkinter           # usually built-in

    On Linux also run:
    sudo apt install espeak

HOW IT WORKS:
  1. Watches shared_output.txt (written by Person 1) for
     new signed text.
  2. When new text arrives it speaks it aloud via the
     system TTS engine (pyttsx3 — works offline).
  3. Simultaneously updates a transparent, always-on-top
     caption window so everyone in the call can read it.
  4. Optionally also watches shared_speech.txt (Person 2)
     and shows speech captions in a second colour.

CAPTION WINDOW:
  - Semi-transparent dark bar at the bottom of the screen.
  - Always on top of other windows (works over Zoom/Meet).
  - Signed text shown in GREEN, spoken text in CYAN.
  - Press ESC or close to quit.

NOTE FOR YOUR TEAM:
  - Run this ALONGSIDE your Zoom/Meet call.
  - To integrate with Person 4's GUI, import and call
    start_overlay(sign_text, speech_text) directly.
=============================================================
"""

import tkinter as tk
import threading
import time
import os
import pyttsx3

# ── Shared files (written by Persons 1 & 2) ──────────────
SIGN_OUTPUT_FILE   = "shared_output.txt"    # Person 1
SPEECH_OUTPUT_FILE = "shared_speech.txt"    # Person 2

# ── TTS Config ────────────────────────────────────────────
TTS_RATE   = 165    # words per minute (150-200 is natural)
TTS_VOLUME = 1.0    # 0.0 – 1.0

# ── Caption Config ────────────────────────────────────────
CAPTION_HEIGHT      = 110   # px
FONT_FAMILY         = "Helvetica"
SIGNED_TEXT_COLOR   = "#00FF99"   # green  — signed input
SPEECH_TEXT_COLOR   = "#00CFFF"   # cyan   — spoken input
LABEL_COLOR         = "#888888"
BG_COLOR            = "#111111"
OPACITY             = 0.82        # window transparency (0-1)


# ══════════════════════════════════════════════════════════
#  TTS ENGINE
# ══════════════════════════════════════════════════════════

class TTSEngine:
    def __init__(self):
        self.engine     = pyttsx3.init()
        self.engine.setProperty("rate",   TTS_RATE)
        self.engine.setProperty("volume", TTS_VOLUME)
        self._lock      = threading.Lock()
        self._last_said = ""

    def say(self, text):
        """Speak text if it's different from what was last said."""
        if text and text != self._last_said:
            self._last_said = text
            # Run in a thread so it doesn't block the GUI
            threading.Thread(target=self._speak, args=(text,),
                             daemon=True).start()

    def _speak(self, text):
        with self._lock:
            self.engine.say(text)
            self.engine.runAndWait()

    def set_voice_by_gender(self, prefer_female=True):
        """Optional: switch TTS voice."""
        voices = self.engine.getProperty("voices")
        for v in voices:
            if prefer_female and "female" in v.name.lower():
                self.engine.setProperty("voice", v.id)
                return
            if not prefer_female and "male" in v.name.lower():
                self.engine.setProperty("voice", v.id)
                return


# ══════════════════════════════════════════════════════════
#  FILE WATCHER
# ══════════════════════════════════════════════════════════

def read_file_safe(path):
    """Read a file, return '' if missing or empty."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read().strip()
    except Exception:
        pass
    return ""


# ══════════════════════════════════════════════════════════
#  CAPTION OVERLAY WINDOW
# ══════════════════════════════════════════════════════════

class CaptionOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self._setup_window()
        self._build_ui()
        self.tts       = TTSEngine()
        self._last_sign   = ""
        self._last_speech = ""

    def _setup_window(self):
        root = self.root
        root.title("SignBridge Captions")
        root.configure(bg=BG_COLOR)
        root.attributes("-topmost",     True)   # always on top
        root.attributes("-alpha",       OPACITY)
        root.overrideredirect(True)             # no title bar

        # Position at bottom of screen
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{sw}x{CAPTION_HEIGHT}+0+{sh - CAPTION_HEIGHT - 40}")
        root.resizable(False, False)

    def _build_ui(self):
        root = self.root

        # ── Top row: labels ───────────────────────────────
        top_frame = tk.Frame(root, bg=BG_COLOR)
        top_frame.pack(fill="x", padx=12, pady=(8, 0))

        tk.Label(top_frame, text="✋ SIGNED",
                 fg=SIGNED_TEXT_COLOR, bg=BG_COLOR,
                 font=(FONT_FAMILY, 9, "bold")).pack(side="left")

        tk.Label(top_frame, text="🎙 SPOKEN",
                 fg=SPEECH_TEXT_COLOR, bg=BG_COLOR,
                 font=(FONT_FAMILY, 9, "bold")).pack(side="right")

        # Close button
        tk.Button(top_frame, text="✕", fg="#666", bg=BG_COLOR,
                  bd=0, command=self.root.destroy,
                  font=(FONT_FAMILY, 9)).pack(side="right", padx=8)

        # ── Sign text ─────────────────────────────────────
        self.sign_var = tk.StringVar(value="Waiting for sign input…")
        tk.Label(root, textvariable=self.sign_var,
                 fg=SIGNED_TEXT_COLOR, bg=BG_COLOR,
                 font=(FONT_FAMILY, 18, "bold"),
                 wraplength=self.root.winfo_screenwidth() - 40,
                 anchor="w").pack(fill="x", padx=14, pady=(2, 0))

        # ── Speech text ───────────────────────────────────
        self.speech_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.speech_var,
                 fg=SPEECH_TEXT_COLOR, bg=BG_COLOR,
                 font=(FONT_FAMILY, 13),
                 wraplength=self.root.winfo_screenwidth() - 40,
                 anchor="w").pack(fill="x", padx=14)

    def update_sign_text(self, text):
        """Call this from Person 4's GUI or directly."""
        if text != self._last_sign:
            self._last_sign = text
            display = text[-80:] if len(text) > 80 else text
            self.sign_var.set(display or "Waiting for sign input…")
            # Speak only completed words (ends with space or is long enough)
            if text.endswith(" ") or len(text) > 20:
                self.tts.say(text.strip())

    def update_speech_text(self, text):
        """Call this from Person 4's GUI or directly."""
        if text != self._last_speech:
            self._last_speech = text
            self.speech_var.set(text[-100:] if len(text) > 100 else text)

    def _poll_files(self):
        """Background thread: watches shared files and updates UI."""
        while True:
            sign_text   = read_file_safe(SIGN_OUTPUT_FILE)
            speech_text = read_file_safe(SPEECH_OUTPUT_FILE)

            # Schedule UI updates on the main thread
            self.root.after(0, self.update_sign_text,   sign_text)
            self.root.after(0, self.update_speech_text, speech_text)

            time.sleep(0.3)   # poll every 300 ms

    def run(self):
        """Start polling thread then enter Tkinter main loop."""
        t = threading.Thread(target=self._poll_files, daemon=True)
        t.start()
        print("[CaptionOverlay] Caption window running.")
        print("[CaptionOverlay] Close the caption bar or press Ctrl+C to stop.")
        self.root.mainloop()


# ══════════════════════════════════════════════════════════
#  STANDALONE MODE  (run without GUI integration)
# ══════════════════════════════════════════════════════════

def run_caption_overlay():
    """
    Standalone entry point.
    Run this while Zoom/Meet is open — the caption bar will
    float on top of the call window.
    """
    overlay = CaptionOverlay()
    overlay.run()


# ══════════════════════════════════════════════════════════
#  BACKGROUND MODE  (for Person 4's integrated GUI)
# ══════════════════════════════════════════════════════════

def start_tts_only(callback_getter):
    """
    Lightweight mode: no overlay window, just TTS.
    callback_getter: a callable that returns the latest signed text.

    Example (Person 4):
        from person3_tts_overlay import start_tts_only
        start_tts_only(lambda: current_sign_text)
    """
    tts     = TTSEngine()
    last    = ""

    def loop():
        nonlocal last
        while True:
            text = callback_getter()
            if text and text != last:
                last = text
                tts.say(text.strip())
            time.sleep(0.5)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_caption_overlay()
