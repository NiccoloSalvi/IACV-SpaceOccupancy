import subprocess

video_path = "video_path"
output_folder = "output_folder"

# Comando FFmpeg per estrarre tutti i frame
subprocess.run([
    "ffmpeg", "-i", video_path, "-vf", "fps=2", output_folder
])

print(f"Frame salvati nella cartella: {output_folder}")

