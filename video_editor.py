from moviepy.editor import VideoFileClip, concatenate_videoclips

def create_highlights(video_path, timestamps, output_path):
    video = VideoFileClip(video_path)
    clips = []

    for t in timestamps:
        start = max(0, t - 2)
        end = min(video.duration, t + 4)
        clips.append(video.subclip(start, end))

    final = concatenate_videoclips(clips)
    final.write_videofile(output_path)
