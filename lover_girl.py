# Lover Girl (Laufey) - video to ASCII, played live
# Created by Eclipsetv (Fernando Abraham Peña Muñoz)
# A JIN2-style piece: every frame of the music video becomes a grid of
# letters. Brightness picks the letter, edges become | / \ - _, and the
# colour comes from the video, quantised to a k-means palette.
#
# All frames are pre-computed once into lover_girl_frames.npz (like the
# JIN2 pieces do), so playback is smooth and locked to the audio.
#
#   pip install -r requirements.txt
#   python lover_girl.py [video.mp4]        # plays, building the cache first if needed
#   python lover_girl.py --cols 320         # more letters across (rebuilds the cache)
#   python lover_girl.py --build            # only (re)build the letter cache
#
# Space pause, R restart, S screenshot, M mute, Up/Down brightness,
# F fullscreen, Esc quit
#
# The source video is Laufey's "Lover Girl" official music video and
# belongs to its creators. This player renders it as letters locally,
# for personal use only. Video: https://youtu.be/obLSGG-oEyw

import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

TITLE = "Lover Girl (Laufey) - ASCII · Eclipsetv"
SIGN = "creado por Eclipsetv (Fernando Abraham Peña Muñoz)"
VIDEO_SIGN = "video · laufey · lover girl · rendered in letters"
FPS_MAX = 30                               # display cap, the video keeps its own rate
FULLSCREEN = True                          # start fullscreen, F toggles it at any time
CW, CH, FONT_PX = 7, 11, 12                # one letter cell, in pixels
COLS = 288                                 # letters across (detail), or pass --cols N
TRAIL = 0.25                              # how much of the last frame stays behind
GAMMA = 0.8                               # lift dark frames a little
SOFT = 0.75                               # playback gamma: opens the shadows (light boost)
BRIGHT = 1.20                             # playback light gain, Up/Down to tune it live
SAT = 1.7                                # colour punch: saturate the video's own palette
GLOW_A, GLOW_B = 0.85, 0.55               # strength of the two glow passes
EDGE_GAIN = 2.2                           # edges = mean + EDGE_GAIN * std, on a 2x grid
EDGE_SCALE = 2                            # edges are measured at 2x the letter grid
PALETTE_K = 72                             # colours of the whole video, by k-means
CACHE_VERSION = 5                         # bump to force a rebuild of the letter cache
DIM, PINK = (110, 104, 120), (255, 122, 182)
MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"

# dark -> bright, letters, numbers and symbols (the JIN2 70-level ramp)
CHARS = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
EDGE_CHARS = "-\\|/"                       # bucket order: 0 '-' 1 '\' 2 '|' 3 '/'

HERE = Path(__file__).parent


def parse_args():
    # [video.mp4] [--cols N] [--build]
    global COLS
    video = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--cols" and i + 1 < len(args):
            i += 1
            try:
                COLS = max(40, min(400, int(args[i])))
            except ValueError:
                sys.exit("--cols needs a number, e.g. --cols 256")
        elif a == "--build":
            pass
        elif video is None:
            video = Path(a)
        i += 1
    return video


VIDEO = parse_args() or HERE / "video.mp4"
AUDIO = HERE / "lover_girl.wav"            # extracted next to this file, if present
CACHE = HERE / "lover_girl_frames.npz"     # finished letters of the whole video

if not VIDEO.exists():
    sys.exit(f"video not found: {VIDEO}\nput video.mp4 next to this file, or pass a path.")


def load_video():
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        sys.exit(f"could not open video: {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 16
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 9
    return cap, float(fps), n, w, h


def build_palette(cap, n, k=PALETTE_K):
    # sample frames spread over the whole video, k-means on their pixels
    samples, step = [], max(1, n // 64)
    for f in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        px = cv2.cvtColor(frame[::6, ::6], cv2.COLOR_BGR2RGB).astype(np.float32)
        samples.append(px.reshape(-1, 3))
    data = np.concatenate(samples)
    data = data[np.random.default_rng(2).choice(len(data), min(len(data), 20000), replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, _, centers = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.clip(centers, 0, 255)          # (k, 3) RGB


def edge_layer(frame):
    # Sobel edges measured at EDGE_SCALE x the letter grid, then the strongest
    # gradient of each block of cells wins, and its angle picks | / \ - _
    w, h = COLS * EDGE_SCALE, ROWS * EDGE_SCALE
    g2 = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    lum2 = cv2.cvtColor(g2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(lum2, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum2, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = cv2.magnitude(gx, gy)
    ang2 = (np.degrees(np.arctan2(gx, -gy)) + 360.0) % 180.0      # angle of the edge line
    s = EDGE_SCALE
    m4 = mag2.reshape(ROWS, s, COLS, s).transpose(0, 2, 1, 3).reshape(ROWS, COLS, s * s)
    a4 = ang2.reshape(ROWS, s, COLS, s).transpose(0, 2, 1, 3).reshape(ROWS, COLS, s * s)
    k = m4.argmax(-1)[..., None]
    return np.take_along_axis(m4, k, -1)[..., 0], np.take_along_axis(a4, k, -1)[..., 0]


def convert(frame, palette):
    # one video frame -> letters, brightness and palette index per cell
    small = cv2.resize(frame, (COLS, ROWS), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    lum = (rgb @ np.array([0.299, 0.587, 0.114], np.float32)) / 255.0   # perceived brightness
    light = lum ** GAMMA

    letters = np.clip((light * (len(CHARS) - 1) + 0.5), 0, len(CHARS) - 1).astype(np.int32)

    mag, ang = edge_layer(frame)
    thr = max(mag.mean() + EDGE_GAIN * mag.std(), 0.05)
    edge = mag > thr
    bucket = ((ang + 22.5) // 45).astype(np.int32) % 4
    letters = np.where(edge, len(CHARS) + bucket, letters).astype(np.uint8)
    light = np.where(edge, np.maximum(light, 0.55), light)         # keep edges visible

    tone = np.argmin(((rgb[..., None, :] - palette) ** 2).sum(-1), axis=-1).astype(np.uint8)
    return letters, light.astype(np.float32), tone


def build(cap, nframes, fps, palette, hook=None):
    # convert the whole video once, like the JIN2 pieces ship their frames
    letters_all = np.empty((nframes, ROWS, COLS), np.uint8)
    light_all = np.empty((nframes, ROWS, COLS), np.uint8)
    tone_all = np.empty((nframes, ROWS, COLS), np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    done = 0
    t0 = time.time()
    while done < nframes:
        ok, frame = cap.read()
        if not ok:
            break
        letters_all[done], light, tone_all[done] = convert(frame, palette)
        light_all[done] = (np.clip(light, 0, 1) * 255 + 0.5).astype(np.uint8)
        done += 1
        if hook is not None and (done % 50 == 0 or done == nframes):
            hook(done / nframes)
        if done % 200 == 0 or done == nframes:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  letters {done}/{nframes}   {rate:5.1f} frames/s", flush=True)
    np.savez(CACHE, version=CACHE_VERSION, letters=letters_all[:done], light=light_all[:done],
             tone=tone_all[:done], palette=palette, fps=fps, n=done, cols=COLS, rows=ROWS)
    print(f"saved {done} frames -> {CACHE.name}", flush=True)
    return done


def vivid(palette, boost):
    # more colour: saturate the video's palette once, at load time
    hsv = cv2.cvtColor(palette.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 1] = np.clip(hsv[:, 1] * boost, 0, 255)
    return cv2.cvtColor(hsv.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_HSV2RGB).reshape(-1, 3)


def glyphs():
    font = pygame.font.SysFont(MONO, FONT_PX)
    atlas = []
    for ch in CHARS + EDGE_CHARS:
        s = pygame.Surface((CW, CH))
        g = font.render(ch, True, (255, 255, 255))
        s.blit(g, ((CW - g.get_width()) // 2, (CH - g.get_height()) // 2))
        atlas.append(pygame.surfarray.array3d(s)[:, :, 0].T.astype(np.float32) / 255)
    return np.stack(atlas)


class Canvas:
    def __init__(self, atlas, palette, win_size):
        self.atlas = (atlas * 255).astype(np.uint8)
        self.palette = palette
        self.shape = pygame.Surface((COLS * CW, ROWS * CH))
        self.win = win_size
        self.trail = pygame.Surface(win_size)
        self.fade = (int(TRAIL * 255),) * 3

    def reset(self):
        self.trail.fill((0, 0, 0))

    def resize(self, win_size):
        # keep the trail alive when the window mode changes
        if self.trail.get_size() != win_size:
            self.trail = pygame.transform.smoothscale(self.trail, win_size)
        self.win = win_size

    def draw(self, letters, light, tone):
        # letters in white, tinted with their colour, and the trail runs at
        # window size: every expensive pixel pass is as small as it can be
        mask = self.atlas[letters.T].transpose(0, 3, 1, 2).reshape(COLS * CW, ROWS * CH)
        pygame.surfarray.blit_array(self.shape, np.repeat(mask[..., None], 3, 2))
        col = (self.palette[tone] * np.clip(light, 0, 1)[..., None]).astype(np.uint8)
        tint = pygame.transform.scale(pygame.surfarray.make_surface(col.transpose(1, 0, 2)), self.shape.get_size())
        self.shape.blit(tint, (0, 0), special_flags=pygame.BLEND_MULT)
        frame = pygame.transform.smoothscale(self.shape, self.win) if self.shape.get_size() != self.win else self.shape
        self.trail.fill(self.fade, special_flags=pygame.BLEND_MULT)      # previous frame fades
        self.trail.blit(frame, (0, 0), special_flags=pygame.BLEND_MAX)
        return self.trail


def wblur(surf, k, radius):
    w, h = surf.get_size()
    small = pygame.transform.smoothscale(surf, (max(1, w // k), max(1, h // k)))
    if hasattr(pygame.transform, "gaussian_blur"):
        small = pygame.transform.gaussian_blur(small, max(1, radius // k))
    return pygame.transform.smoothscale(small, (w, h))


def compose(art, tier):
    # glow passes at window size, only once per new frame; the tier adapts
    # to the machine: 2 = both glows, 1 = one, 0 = none
    out = pygame.Surface(art.get_size())
    out.fill((0, 0, 0))
    out.blit(art, (0, 0))
    if tier >= 1:
        out.blit(glow_layer(wblur(art, 4, 6), GLOW_A), (0, 0), special_flags=pygame.BLEND_ADD)
    if tier >= 2:
        out.blit(glow_layer(wblur(art, 8, 26), GLOW_B), (0, 0), special_flags=pygame.BLEND_ADD)
    return out


def glow_layer(surf, amount):
    layer = surf.copy()
    layer.fill((int(255 * amount),) * 3, special_flags=pygame.BLEND_MULT)
    return layer


fullscreen = FULLSCREEN                    # live fullscreen state, F toggles it


def fit():
    # the biggest art-sized rect that fits the current display, centred
    info = pygame.display.Info()
    if fullscreen:
        sw, sh = info.current_w, info.current_h
    else:
        k = min(1.0, 0.85 * info.current_h / H, 0.95 * info.current_w / W) if info.current_w > 0 else 1.0
        sw, sh = int(W * k), int(H * k)
    f = min(sw / W, sh / H)
    w, h = int(W * f), int(H * f)
    return (w, h), ((sw - w) // 2, (sh - h) // 2)


def open_window(full):
    # fullscreen at the desktop resolution, the art letterboxed inside it
    global fullscreen
    fullscreen = full
    if full:
        window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        return window, fit()
    window, _ = pygame.display.set_mode((0, 0)), None
    return window, fit()


class Loader:
    # a loading screen with title, credit and an ascii progress bar, so the
    # window stays alive while the letter cache is built or read
    def __init__(self, window, screen, clock, win, off):
        self.window, self.screen, self.clock = window, screen, clock
        self.win, self.off = win, off
        self.big = pygame.font.SysFont(MONO, 52, bold=True)
        self.small = pygame.font.SysFont(MONO, 17)
        self.shown, self.t0 = 0.0, time.time()

    def draw(self, frac, label):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        now = time.time() - self.t0
        if frac is None:                                       # sweep while waiting
            target = 0.5 + 0.5 * math.sin(now * 2.2)
        else:
            target = max(frac, 0.05)
        self.shown += (target - self.shown) * 0.15
        spin = "|/-\\"[int(now * 8) % 4]

        s = self.screen
        s.fill((0, 0, 0))
        title = self.big.render("LOVER GIRL", True, PINK)
        s.blit(title, ((W - title.get_width()) // 2, int(H * 0.28)))
        sub = self.small.render("laufey · el video, en letras", True, DIM)
        s.blit(sub, ((W - sub.get_width()) // 2, int(H * 0.28) + 78))
        credit = self.small.render("creado por Eclipsetv (Fernando Abraham Peña Muñoz)", True, (208, 204, 218))
        s.blit(credit, ((W - credit.get_width()) // 2, int(H * 0.28) + 112))

        line = self.small.render(f"{spin} {label}", True, PINK)
        s.blit(line, ((W - line.get_width()) // 2, int(H * 0.56) - 34))
        nbar = 56
        filled = int(round(self.shown * nbar))
        bar = self.small.render("█" * filled + "·" * (nbar - filled), True, PINK)
        s.blit(bar, ((W - bar.get_width()) // 2, int(H * 0.56)))
        pct = self.small.render(f"{int(self.shown * 100):3d}%", True, DIM)
        s.blit(pct, ((W - pct.get_width()) // 2, int(H * 0.56) + 30))

        hint = self.small.render("espacio pausa · r reiniciar · s captura · m silencio · f pantalla completa", True, DIM)
        s.blit(hint, ((W - hint.get_width()) // 2, H - 64))
        frame = pygame.transform.smoothscale(s, self.win) if s.get_size() != self.win else s
        self.window.fill((0, 0, 0))
        self.window.blit(frame, self.off)
        pygame.display.flip()
        self.clock.tick(30)


def run_job(loader, label, fn):
    # run fn(hook) in a thread while the loading screen animates
    box = {"frac": 0.0, "done": False, "error": None, "result": None}

    def hook(f):
        box["frac"] = f

    def work():
        try:
            box["result"] = fn(hook)
        except Exception as err:
            box["error"] = err
        finally:
            box["done"] = True

    threading.Thread(target=work, daemon=True).start()
    while not box["done"]:
        loader.draw(box["frac"], label)
    if box["error"] is not None:
        pygame.quit()
        raise box["error"]
    return box["result"]


def cache_ok():
    if not CACHE.exists() or CACHE.stat().st_mtime < VIDEO.stat().st_mtime:
        return False
    try:
        with np.load(CACHE) as d:
            return (int(d["version"]) == CACHE_VERSION
                    and int(d["cols"]) == COLS and int(d["rows"]) == ROWS)
    except Exception:
        return False


def main():
    global COLS, ROWS, W, H
    cap, fps, nframes, vw, vh = load_video()
    ROWS = max(8, round(COLS * (vh / vw) * (CW / CH)))        # letters down
    W, H = COLS * CW, ROWS * CH

    if "--build" in sys.argv:                                 # console-only rebuild
        palette = build_palette(cap, nframes)
        build(cap, nframes, fps, palette)
        cap.release()
        return

    # the window opens right away: a loading screen keeps it alive
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    window, (win, off) = open_window(FULLSCREEN)
    pygame.display.set_caption(TITLE)
    screen = pygame.Surface((W, H))
    clock = pygame.time.Clock()
    small = pygame.font.SysFont(MONO, 17)
    loader = Loader(window, screen, clock, win, off)

    if not cache_ok():
        def build_job(hook):
            palette = build_palette(cap, nframes)
            build(cap, nframes, fps, palette, hook=hook)
        run_job(loader, "convirtiendo el video en letras", build_job)
    cap.release()

    def cache_job(hook):
        with np.load(CACHE) as d:
            hook(0.15)
            letters, light = d["letters"], d["light"]
            hook(0.65)
            tone, palette = d["tone"], d["palette"]
            hook(0.92)
            meta = (float(d["fps"]), int(d["n"]), int(d["cols"]), int(d["rows"]))
        return letters, light, tone, palette, meta

    LETTERS, LIGHT, TONE, PALETTE, (fps, n, cols_, rows_) = run_job(loader, "cargando letras y colores", cache_job)
    PALETTE = vivid(PALETTE, SAT)
    COLS, ROWS = int(cols_), int(rows_)
    W, H = COLS * CW, ROWS * CH
    loop = n / fps
    win, off = fit()                                          # re-fit the letterbox to the final grid

    have_audio = AUDIO.exists()
    if have_audio:
        try:
            pygame.mixer.music.load(str(AUDIO))
        except pygame.error:
            have_audio = False

    canvas = Canvas(glyphs(), PALETTE, win)
    bright, dirty = BRIGHT, False
    glow_tier = 2

    def warm_job(hook):
        # draw a few frames hot and pick how much glow this machine can
        # afford, so playback never lags behind the song
        nonlocal glow_tier
        times, steps = [], [3, 12, 21, 30]
        for j, i in enumerate(steps):
            light = np.clip((LIGHT[i].astype(np.float32) / 255) ** SOFT * bright, 0, 1)
            t0 = time.perf_counter()
            compose(canvas.draw(LETTERS[i], light, TONE[i]), 2)
            times.append(time.perf_counter() - t0)
            hook((j + 1) / len(steps))
        avg, budget = sum(times) / len(times), 0.92 / fps
        glow_tier = 2 if avg <= budget else (1 if avg <= budget * 1.25 else 0)

    run_job(loader, "preparando el motor (sin atrasos)", warm_job)
    canvas.reset()

    t, clock_t, paused, muted = 0.0, 0.0, False, False
    comp = pygame.Surface(win)
    comp.fill((0, 0, 0))
    shown_i, sync = -1, "clock"
    if have_audio:
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()

    def restart_music():
        if have_audio:
            pygame.mixer.music.stop()
            pygame.mixer.music.play()
            if paused:
                pygame.mixer.music.pause()

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                paused = not paused
                if have_audio:
                    pygame.mixer.music.pause() if paused else pygame.mixer.music.unpause()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                t, clock_t, shown_i = 0.0, 0.0, -1
                canvas.reset()
                restart_music()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                pygame.image.save(window, str(HERE / f"lover_girl_{int(time.time())}.png"))
            if e.type == pygame.KEYDOWN and e.key == pygame.K_m and have_audio:
                muted = not muted
                pygame.mixer.music.set_volume(0.0 if muted else 1.0)
            if e.type == pygame.KEYDOWN and e.key == pygame.K_f:
                window, (win, off) = open_window(not fullscreen)
                canvas.resize(win)
                if comp.get_size() != win:
                    comp = pygame.transform.smoothscale(comp, win)
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_UP, pygame.K_DOWN):
                bright = float(np.clip(bright + (0.05 if e.key == pygame.K_UP else -0.05), 0.4, 2.4))
                dirty = True                                    # redraw with the new light

        dt = clock.tick(FPS_MAX) / 1000
        if not paused:
            clock_t += min(dt, 1 / 10)
            # the song is the master clock: follow the position the mixer has
            # actually played, falling back to the wall clock if it misreports
            pos = pygame.mixer.music.get_pos() if have_audio else -1
            if pos is not None and pos >= 0 and abs(pos / 1000 - clock_t) < 0.5:
                t, sync = pos / 1000, "audio"
            else:
                t, sync = clock_t, "clock"
            ended = t >= loop or (have_audio and clock_t > 0.5 and not pygame.mixer.music.get_busy())
            if ended:
                t, clock_t, shown_i = 0.0, 0.0, -1
                canvas.reset()
                restart_music()
            i = min(int(t * fps), n - 1)
            if i != shown_i or dirty:                          # draw only when the frame changed
                light = np.clip((LIGHT[i].astype(np.float32) / 255) ** SOFT * bright, 0, 1)
                comp = compose(canvas.draw(LETTERS[i], light, TONE[i]), glow_tier)
                shown_i, dirty = i, False
        elif dirty and shown_i >= 0:                           # retune the light while paused
            light = np.clip((LIGHT[shown_i].astype(np.float32) / 255) ** SOFT * bright, 0, 1)
            comp = compose(canvas.draw(LETTERS[shown_i], light, TONE[shown_i]), glow_tier)
            dirty = False

        # every tick: one cheap blit of the finished frame plus the hud
        ox, oy = off
        kx, ky = win[0] / W, win[1] / H
        window.fill((0, 0, 0))
        window.blit(comp, (ox, oy))
        y = H - 108
        window.blit(small.render(SIGN, True, (208, 204, 218)), (int(ox + 64 * kx), int(oy + y * ky)))
        window.blit(small.render(VIDEO_SIGN, True, (120, 112, 128)), (int(ox + 64 * kx), int(oy + (y + 24) * ky)))
        window.blit(small.render(f"frame {shown_i + 1:04d}/{n}   t {t:06.2f}s   sync {sync}", True, DIM), (int(ox + 64 * kx), int(oy + (y + 48) * ky)))
        window.blit(small.render(f"letters {COLS * ROWS:05,d}   palette {len(PALETTE)}   glow {glow_tier}   light x{bright:.2f}", True, DIM), (int(ox + 64 * kx), int(oy + (y + 72) * ky)))
        tt = small.render(f"{'●' if int(t * 2) % 2 == 0 else ' '} {'muted' if muted else 'playing'}", True, PINK)
        window.blit(tt, (int(ox + win[0] - 64 * kx - tt.get_width()), int(oy + 30 * ky)))
        pygame.display.flip()


if __name__ == "__main__":
    sys.exit(main())
