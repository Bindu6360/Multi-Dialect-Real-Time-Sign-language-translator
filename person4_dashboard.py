"""
=============================================================
  MODULE 4 — GUI DASHBOARD + INTEGRATION  (Person 4)
  Plugs Persons 1, 2, and 3 together into one polished
  desktop application.
=============================================================

SETUP (run once):
    pip install opencv-python mediapipe openai-whisper
        sounddevice numpy scipy pyttsx3 Pillow

    Make sure all four .py files are in the SAME folder:
        person1_sign_to_text.py
        person2_speech_to_text.py
        person3_tts_overlay.py
        person4_dashboard.py   ← this file

HOW IT WORKS:
  Launches a single Tkinter window with:
    LEFT  — live webcam feed with hand skeleton overlay
    RIGHT — two caption panels (signed text / speech text)
             + confidence indicator + session stats
  BOTTOM — control bar: Start / Stop / Clear / Save Transcript

  Each module runs in its own background thread so the
  GUI stays responsive.

NOTE FOR YOUR TEAM:
  - This is the file you demo.  Run: python person4_dashboard.py
  - If a module file is missing it degrades gracefully.
  - Transcript is saved to transcript_<timestamp>.txt on Stop.
=============================================================
"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import time
import os
import datetime
import queue

# ── Try importing team modules (degrade gracefully if missing) ──
try:
    from person1_sign_to_text import (
        load_or_build_classifier, extract_features,
        WordBuilder, write_shared_output
    )
    import cv2
    import mediapipe as mp
    HAS_SIGN = True
    print("[Dashboard] ✅ Sign-to-Text module loaded.")
except ImportError as e:
    HAS_SIGN = False
    print(f"[Dashboard] ⚠️  Sign-to-Text not available: {e}")

try:
    from person2_speech_to_text import start_in_background as start_stt
    HAS_SPEECH = True
    print("[Dashboard] ✅ Speech-to-Text module loaded.")
except ImportError as e:
    HAS_SPEECH = False
    print(f"[Dashboard] ⚠️  Speech-to-Text not available: {e}")

try:
    from person3_tts_overlay import TTSEngine
    HAS_TTS = True
    print("[Dashboard] ✅ TTS module loaded.")
except ImportError as e:
    HAS_TTS = False
    print(f"[Dashboard] ⚠️  TTS not available: {e}")

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── Shared files ──────────────────────────────────────────
SIGN_FILE   = "shared_output.txt"
SPEECH_FILE = "shared_speech.txt"

# ── Colour palette ────────────────────────────────────────
C = {
    "bg":        "#0D1117",
    "panel":     "#161B22",
    "border":    "#30363D",
    "green":     "#3FB950",
    "cyan":      "#58A6FF",
    "yellow":    "#D29922",
    "red":       "#F85149",
    "text":      "#E6EDF3",
    "subtext":   "#8B949E",
    "accent":    "#1F6FEB",
    "highlight": "#21262D",
}

WEBCAM_W, WEBCAM_H = 480, 360


# ══════════════════════════════════════════════════════════
#  SIGN ENGINE THREAD  (wraps Person 1)
# ══════════════════════════════════════════════════════════

class SignThread(threading.Thread):
    """Runs sign recognition and pushes frames + text to queues."""

    def __init__(self, frame_queue, text_queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.text_queue  = text_queue
        self.running     = False
        self.confidence  = 0.0
        self.letter      = ""

    def run(self):
        if not HAS_SIGN:
            return

        import numpy as np
        clf, labels = load_or_build_classifier()
        builder     = WordBuilder()

        mp_hands   = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(0)
        self.running = True

        with mp_hands.Hands(
            static_image_mode        = False,
            max_num_hands            = 1,
            min_detection_confidence = 0.7,
            min_tracking_confidence  = 0.5,
        ) as hands:

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame     = cv2.flip(frame, 1)
                rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = hands.process(rgb)

                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(
                        frame, lm, mp_hands.HAND_CONNECTIONS)

                    features   = extract_features(lm).reshape(1, -1)
                    predicted  = clf.predict(features)[0]
                    proba      = clf.predict_proba(features)[0]
                    self.confidence = float(np.max(proba))
                    self.letter     = predicted
                    builder.update(predicted)

                text = builder.get_display()
                write_shared_output(text)

                # Push frame (resize for GUI)
                small = cv2.resize(frame, (WEBCAM_W, WEBCAM_H))
                if not self.frame_queue.full():
                    self.frame_queue.put(small)

                if not self.text_queue.full():
                    self.text_queue.put((text, self.confidence, self.letter))

        cap.release()

    def stop(self):
        self.running = False

    def clear_text(self):
        write_shared_output("")


# ══════════════════════════════════════════════════════════
#  SPEECH CALLBACK
# ══════════════════════════════════════════════════════════

_latest_speech = [""]

def on_speech(text):
    _latest_speech[0] = text
    # Write to shared file for Person 3
    with open(SPEECH_FILE, "w") as f:
        f.write(text)


# ══════════════════════════════════════════════════════════
#  MAIN GUI
# ══════════════════════════════════════════════════════════

class Dashboard:
    def __init__(self):
        self.root        = tk.Tk()
        self.session_log = []
        self.running     = False
        self.frame_q     = queue.Queue(maxsize=2)
        self.text_q      = queue.Queue(maxsize=10)
        self.sign_thread = None
        self.tts         = TTSEngine() if HAS_TTS else None
        self._last_spoken = ""
        self._build_ui()

    # ──────────────────────────────────────────────────────
    #  UI CONSTRUCTION
    # ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        root.title("SignBridge — Live Accessibility Dashboard")
        root.configure(bg=C["bg"])
        root.geometry("1060x680")
        root.resizable(True, True)

        # ── Fonts ─────────────────────────────────────────
        f_title  = tkfont.Font(family="Courier", size=14, weight="bold")
        f_head   = tkfont.Font(family="Courier", size=11, weight="bold")
        f_body   = tkfont.Font(family="Courier", size=10)
        f_big    = tkfont.Font(family="Courier", size=22, weight="bold")
        f_small  = tkfont.Font(family="Courier", size=9)

        # ── Title bar ─────────────────────────────────────
        title_bar = tk.Frame(root, bg=C["panel"], height=50)
        title_bar.pack(fill="x")
        tk.Label(title_bar,
                 text="✋  SignBridge  —  Real-Time Sign & Speech Translator",
                 fg=C["green"], bg=C["panel"], font=f_title,
                 pady=12).pack(side="left", padx=16)

        # Status pill
        self.status_var = tk.StringVar(value="● IDLE")
        tk.Label(title_bar, textvariable=self.status_var,
                 fg=C["subtext"], bg=C["panel"],
                 font=f_small).pack(side="right", padx=16)

        # ── Main content ──────────────────────────────────
        content = tk.Frame(root, bg=C["bg"])
        content.pack(fill="both", expand=True, padx=12, pady=8)

        # LEFT — webcam
        left = tk.Frame(content, bg=C["panel"],
                        bd=1, relief="flat",
                        highlightbackground=C["border"],
                        highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 8))

        tk.Label(left, text="WEBCAM FEED",
                 fg=C["subtext"], bg=C["panel"],
                 font=f_small, pady=6).pack()

        self.cam_label = tk.Label(left, bg="black",
                                  width=WEBCAM_W, height=WEBCAM_H)
        self.cam_label.pack(padx=8, pady=(0, 8))

        # Letter + confidence
        letter_frame = tk.Frame(left, bg=C["panel"])
        letter_frame.pack(fill="x", padx=8, pady=4)

        tk.Label(letter_frame, text="DETECTING:",
                 fg=C["subtext"], bg=C["panel"], font=f_small).pack(side="left")

        self.letter_var = tk.StringVar(value="—")
        tk.Label(letter_frame, textvariable=self.letter_var,
                 fg=C["green"], bg=C["panel"],
                 font=f_big).pack(side="left", padx=8)

        self.conf_var = tk.StringVar(value="")
        tk.Label(letter_frame, textvariable=self.conf_var,
                 fg=C["yellow"], bg=C["panel"], font=f_body).pack(side="right")

        # Confidence bar
        conf_bar_bg = tk.Frame(left, bg=C["border"],
                               height=6, width=WEBCAM_W - 16)
        conf_bar_bg.pack(padx=8, pady=(0, 8))
        conf_bar_bg.pack_propagate(False)
        self.conf_bar = tk.Frame(conf_bar_bg, bg=C["green"], height=6, width=0)
        self.conf_bar.place(x=0, y=0, relheight=1)

        # RIGHT — caption panels
        right = tk.Frame(content, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)

        # Signed text panel
        self._make_panel(
            right, "✋  SIGNED TEXT  →  (Heard by everyone)",
            C["green"], "sign_text_var",
            "sign_panel"
        )

        # Speech text panel
        self._make_panel(
            right, "🎙  SPOKEN TEXT  →  (Seen by signer)",
            C["cyan"], "speech_text_var",
            "speech_panel"
        )

        # Stats row
        stats = tk.Frame(right, bg=C["highlight"],
                         bd=1, relief="flat",
                         highlightbackground=C["border"],
                         highlightthickness=1)
        stats.pack(fill="x", pady=(6, 0))

        self.words_var = tk.StringVar(value="Words translated: 0")
        self.time_var  = tk.StringVar(value="Session: 00:00")

        for var in (self.words_var, self.time_var):
            tk.Label(stats, textvariable=var,
                     fg=C["subtext"], bg=C["highlight"],
                     font=f_small, padx=16, pady=6).pack(side="left")

        # ── Control bar ───────────────────────────────────
        ctrl = tk.Frame(root, bg=C["panel"], height=54)
        ctrl.pack(fill="x", side="bottom")

        btn_cfg = dict(font=f_head, bd=0, padx=20, pady=10,
                       cursor="hand2", relief="flat")

        self.start_btn = tk.Button(ctrl, text="▶  START",
            bg=C["green"], fg=C["bg"],
            command=self.start_session, **btn_cfg)
        self.start_btn.pack(side="left", padx=(12, 4), pady=6)

        self.stop_btn = tk.Button(ctrl, text="■  STOP",
            bg=C["red"], fg="white",
            command=self.stop_session, state="disabled", **btn_cfg)
        self.stop_btn.pack(side="left", padx=4, pady=6)

        tk.Button(ctrl, text="⌫  CLEAR",
            bg=C["highlight"], fg=C["text"],
            command=self.clear_text, **btn_cfg).pack(side="left", padx=4, pady=6)

        tk.Button(ctrl, text="💾  SAVE TRANSCRIPT",
            bg=C["accent"], fg="white",
            command=self.save_transcript, **btn_cfg).pack(side="right", padx=12, pady=6)

        # Module status indicators
        for label, ok in [
            ("Sign", HAS_SIGN), ("Speech", HAS_SPEECH), ("TTS", HAS_TTS)
        ]:
            color = C["green"] if ok else C["red"]
            mark  = "✔" if ok else "✘"
            tk.Label(ctrl, text=f"{mark} {label}",
                     fg=color, bg=C["panel"],
                     font=f_small).pack(side="right", padx=6)

        # ── Start polling loop ─────────────────────────────
        self._start_time = None
        self._poll()

    def _make_panel(self, parent, title, color, var_name, panel_name):
        f_head = tkfont.Font(family="Courier", size=10, weight="bold")
        f_big  = tkfont.Font(family="Courier", size=16, weight="bold")

        panel = tk.Frame(parent, bg=C["panel"],
                         bd=1, relief="flat",
                         highlightbackground=C["border"],
                         highlightthickness=1)
        panel.pack(fill="both", expand=True, pady=(0, 6))
        setattr(self, panel_name, panel)

        tk.Label(panel, text=title,
                 fg=color, bg=C["panel"],
                 font=f_head, pady=6, padx=10,
                 anchor="w").pack(fill="x")

        tk.Frame(panel, bg=C["border"], height=1).pack(fill="x")

        var = tk.StringVar(value="—")
        setattr(self, var_name, var)
        tk.Label(panel, textvariable=var,
                 fg=C["text"], bg=C["panel"],
                 font=f_big,
                 wraplength=540,
                 justify="left",
                 anchor="nw",
                 padx=12, pady=10).pack(fill="both", expand=True)

    # ──────────────────────────────────────────────────────
    #  SESSION CONTROL
    # ──────────────────────────────────────────────────────

    def start_session(self):
        self.running     = True
        self._start_time = time.time()
        self.status_var.set("● LIVE")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.session_log.clear()

        # Start sign thread
        if HAS_SIGN:
            self.sign_thread = SignThread(self.frame_q, self.text_q)
            self.sign_thread.start()
            print("[Dashboard] Sign recognition started.")

        # Start speech thread
        if HAS_SPEECH:
            start_stt(callback=on_speech)
            print("[Dashboard] Speech-to-text started.")

        print("[Dashboard] Session started.")

    def stop_session(self):
        self.running = False
        if self.sign_thread:
            self.sign_thread.stop()
        self.status_var.set("● STOPPED")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        print("[Dashboard] Session stopped.")

    def clear_text(self):
        self.sign_text_var.set("—")
        self.speech_text_var.set("—")
        write_shared_output("") if HAS_SIGN else None
        self.session_log.clear()

    def save_transcript(self):
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"transcript_{ts}.txt"
        with open(path, "w") as f:
            f.write(f"SignBridge Session Transcript — {ts}\n")
            f.write("=" * 50 + "\n\n")
            for entry in self.session_log:
                f.write(entry + "\n")
        print(f"[Dashboard] Transcript saved → {path}")
        self.status_var.set(f"✔ Saved: {path}")

    # ──────────────────────────────────────────────────────
    #  POLLING LOOP (updates GUI from queues)
    # ──────────────────────────────────────────────────────

    def _poll(self):
        # ── Webcam frame ──────────────────────────────────
        if HAS_PIL and not self.frame_q.empty():
            frame = self.frame_q.get_nowait()
            try:
                import cv2
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = Image.fromarray(rgb)
                photo = ImageTk.PhotoImage(img)
                self.cam_label.config(image=photo)
                self.cam_label.image = photo
            except Exception:
                pass

        # ── Sign text ─────────────────────────────────────
        latest_sign = ""
        while not self.text_q.empty():
            latest_sign, conf, letter = self.text_q.get_nowait()

        if latest_sign:
            self.sign_text_var.set(latest_sign[-60:] or "—")
            self.letter_var.set(letter or "—")
            self.conf_var.set(f"{conf*100:.0f}%")
            bar_w = int((WEBCAM_W - 16) * conf)
            self.conf_bar.place(x=0, y=0, relheight=1, width=bar_w)

            # TTS
            if self.tts and latest_sign != self._last_spoken:
                if latest_sign.endswith(" ") or len(latest_sign) > 25:
                    self.tts.say(latest_sign.strip())
                    self._last_spoken = latest_sign

            # Log
            if latest_sign and (not self.session_log or
                                 self.session_log[-1] != f"[SIGN] {latest_sign}"):
                self.session_log.append(f"[SIGN]   {latest_sign}")

        # ── Speech text ───────────────────────────────────
        speech = _latest_speech[0]
        if speech:
            self.speech_text_var.set(speech[-80:] or "—")
            self.session_log.append(f"[SPEECH] {speech}")
            _latest_speech[0] = ""

        # ── Stats ─────────────────────────────────────────
        sign_text = self.sign_text_var.get()
        words     = len(sign_text.split()) if sign_text != "—" else 0
        self.words_var.set(f"Words translated: {words}")

        if self._start_time:
            elapsed = int(time.time() - self._start_time)
            m, s    = divmod(elapsed, 60)
            self.time_var.set(f"Session: {m:02d}:{s:02d}")

        # Schedule next poll
        self.root.after(80, self._poll)   # ~12 fps

    # ──────────────────────────────────────────────────────

    def run(self):
        print("[Dashboard] SignBridge running. Close window to exit.")
        self.root.mainloop()


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = Dashboard()
    app.run()
