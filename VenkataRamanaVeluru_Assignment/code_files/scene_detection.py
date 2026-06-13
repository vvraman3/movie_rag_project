import os
import json
from scenedetect import detect, ContentDetector
from scenedetect.scene_manager import save_images
import moviepy as mp

def detect_scenes(video_path, output_dir, movie_name):
    """
    Detect scenes in a video and save each scene as a separate clip.
    """
    # Detect scene changes
    scene_list = detect(video_path, ContentDetector())

    # Load video
    video = mp.VideoFileClip(video_path)

    metadata = []

    # Extract and save each scene  
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        duration = end_time - start_time
        scene_clip = video.subclipped(start_time, end_time)
        scene_filename = f"{movie_name}_scene_{i+1}.mp4"
        scene_filepath = os.path.join(output_dir, scene_filename)
        scene_clip.write_videofile(scene_filepath, codec="libx264")

        # Collect metadata
        scene_data = {
            "movie_name": movie_name,
            "scene_id": i + 1,
            "file_path": os.path.abspath(scene_filepath),
            "start_timestamp": round(start_time, 2),
            "end_timestamp": round(end_time, 2),
            "duration": round(duration, 2)
        }
        metadata.append(scene_data)

    print(f"Extracted {len(scene_list)} scenes from {video_path}")
    return metadata

if __name__ == "__main__":
    video_dir = "../videos"
    output_dir = "../scenes"
    output_json = "../data/scene_metadata.json"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_metadata = []

    for video_file in os.listdir(video_dir):
        if video_file.endswith(('.mp4', '.avi', '.mkv')):
            video_path = os.path.join(video_dir, video_file)
            movie_name = os.path.splitext(video_file)[0]
            metadata = detect_scenes(video_path, output_dir, movie_name)
            all_metadata.extend(metadata)

    # Save metadata to JSON
    with open(output_json, 'w') as f:
        json.dump(all_metadata, f, indent=4)

    print(f"Metadata saved to {output_json}")