import os
import numpy as np
import librosa
from moviepy.editor import VideoFileClip, concatenate_videoclips

VIDEO_PATH = "input_video/match.mp4"
OUTPUT_PATH = "output/highlights.mp4"
AUDIO_PATH = "temp_audio.wav"

CLIP_BEFORE = 1.5
CLIP_AFTER = 2
MIN_GAP = 8
MAX_RATIO = 0.30   # highlights ≤ 30% of video

# ---------------- AUDIO ENERGY ----------------
def extract_energy(video_path):
    video = VideoFileClip(video_path)
    if video.audio is None:
        raise Exception("Video has no audio track")

    video.audio.write_audiofile(AUDIO_PATH, verbose=False, logger=None)
    audio, sr = librosa.load(AUDIO_PATH, sr=None)
    energy = librosa.feature.rms(y=audio)[0]
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr)

    return energy, times, video.duration

# ---------------- MERGE OVERLAPS ----------------
def merge_segments(segments):
    merged = []
    for start, end in sorted(segments):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged

# ---------------- MAIN ----------------
def generate_highlights():
    energy, times, duration = extract_energy(VIDEO_PATH)

    # Stronger threshold for short videos
    percentile = 95 if duration < 300 else 92
    threshold = np.percentile(energy, percentile)

    points = times[energy > threshold]

    segments = []
    last = -999
    for t in points:
        if t - last >= MIN_GAP:
            start = max(0, t - CLIP_BEFORE)
            end = min(duration, t + CLIP_AFTER)
            segments.append((start, end))
            last = t

    segments = merge_segments(segments)

    video = VideoFileClip(VIDEO_PATH)
    clips = []
    max_time = duration * MAX_RATIO
    used = 0

    for s, e in segments:
        if used >= max_time:
            break
        clip_len = min(e - s, max_time - used)
        clips.append(video.subclip(s, s + clip_len))
        used += clip_len

    if not clips:
        raise Exception("No highlights detected")

    final = concatenate_videoclips(clips)
    final.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac")

    print("\n=== RESULT ===")
    print(f"Original video : {int(duration)} sec")
    print(f"Highlights     : {int(final.duration)} sec")

# ---------------- RUN ----------------
if __name__ == "__main__":
    generate_highlights()
