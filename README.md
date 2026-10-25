# SignBridge — Team Setup Guide

A real-time Sign Language ↔ Speech converter that works as an
overlay on top of Zoom, Google Meet, or any meeting platform.

---

## Team Split

| Person | File | What it does |
|--------|------|-------------|
| Person 1 | `person1_sign_to_text.py` | Webcam → ASL hand detection → text |
| Person 2 | `person2_speech_to_text.py` | Microphone → Whisper → text |
| Person 3 | `person3_tts_caption_overlay.py` | Sign text → spoken aloud + floating captions |
| Person 4 | `person4_gui_dashboard.py` | Master dashboard that ties everything together |

---

## Quick Start (Everyone)

### Step 1 — Install dependencies
```
pip install -r requirements.txt
```

Linux extra:  sudo apt-get install espeak portaudio19-dev

### Step 2 — Put all 4 .py files in the same folder

### Step 3 — Run
Person 4 launches the dashboard first, then uses the START buttons inside it.
Or each person runs their own file in a separate terminal.

---

## How the modules share data

They write to plain text files in the same folder:

  person1 ──writes──> sign_output.txt
  person2 ──writes──> speech_output.txt

  person3 reads sign_output.txt → speaks aloud + captions
  person4 reads both files → shows in dashboard

No sockets, no APIs. Simple and reliable for a 3-day build.

---

## Person 1 — Getting real accuracy

The model ships with synthetic data so it runs immediately.
To collect real ASL training data, call collect_training_data()
for each letter, then call train_from_csv().
See comments inside person1_sign_to_text.py for exact steps.

---

## Person 2 — Whisper model sizes

Edit WHISPER_MODEL in person2_speech_to_text.py:
  tiny   = 75 MB,  fastest,  OK accuracy
  base   = 140 MB, fast,     good accuracy  ← recommended
  small  = 460 MB, medium,   better accuracy

---

## Demo checklist for 40% submission

[ ] Person 1: Webcam opens, hand landmarks shown, letters appear
[ ] Person 2: Mic captures audio, transcription prints live
[ ] Person 3: Floating caption window shows, TTS speaks sign text
[ ] Person 4: Dashboard shows webcam + both text feeds + controls

That is a complete, integrated, demonstrable prototype.
