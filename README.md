# LOVER GIRL (ASCII)

Lover Girl (Laufey) · video to ASCII, 2026
Created by Eclipsetv (Fernando Abraham Peña Muñoz), in the style of the JIN2 pieces

## About

The official music video of Laufey's "Lover Girl", played live as ASCII code art:
every frame of the video becomes a grid of letters in real time.
Brightness picks the letter, edges become | / \ - _, and the colour comes from
the video itself, quantised to a 32-colour k-means palette.
The audio track plays in sync while the letters draw the video.

## Technology

Python · NumPy · pygame-ce · OpenCV

## How It Works

1. Palette: frames sampled across the whole video are clustered with k-means
   into 64 colours, so every letter takes one of the video's own tones.
2. Grid: each video frame is reduced to a grid of letter cells (288 across,
   137 down on a 4:3 video; tune with --cols).
3. Letters: the brightness of each cell picks a character from a 70-level ramp
   of letters, numbers and symbols (dark -> bright).
4. Edges: a Sobel gradient, measured at twice the grid resolution, picks
   | / \ - _ along the edge direction.
5. Render: white glyphs are tinted per cell, a trail keeps part of the last
   frame, and glow passes add light. Up/Down tunes the light live.
6. Fullscreen: the piece starts fullscreen (letterboxed to keep the letters
   square) and F toggles it at any time.

## Run

```
pip install -r requirements.txt
python lover_girl.py            # uses video.mp4 next to this file
python lover_girl.py other.mp4  # or any video you have
python lover_girl.py --cols 320 # more letters across (rebuilds the cache)
```

Space pause, R restart, S screenshot, M mute, Up/Down brightness,
F fullscreen, Esc quit.

A loading screen shows on every run: it reads (or builds) the letter cache
and warms the engine first, measuring how much glow the machine can afford,
so playback never lags behind the song.

## Credits & Use

ASCII piece created by Eclipsetv (Fernando Abraham Peña Muñoz),
written in the style of the JIN2 (@jin_22ee) code art pieces.

This repository contains the code only: video.mp4, lover_girl.wav and
lover_girl_frames.npz stay local (size limits and copyright) — the script
rebuilds everything from your own copy of the video on first run.

Video: "Lover Girl" — Laufey (official music video), https://youtu.be/obLSGG-oEyw
All rights to the song and video belong to their creators.
This is a personal, local rendering for private use only; do not redistribute
the video or its letters.
