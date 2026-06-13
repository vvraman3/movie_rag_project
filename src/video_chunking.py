import os
import json
import subprocess
from moviepy import VideoFileClip
import librosa
import numpy as np

def extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(audio_path)
    clip.close()

def is_audio_same(audio1, audio2, threshold=0.5):
    try:
        y1, sr1 = librosa.load(audio1, sr=None)
        y2, sr2 = librosa.load(audio2, sr=None)
        if sr1 != sr2:
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr1, n_mfcc=13)
        dist = np.linalg.norm(mfcc1.mean(axis=1) - mfcc2.mean(axis=1))
        return dist < threshold
    except:
        return False

def main():
    data_dir = "../data"
    scenes_dir = "../scenes"
    scenes_audio_dir = "../scenes_audio"
    video_chunks_dir = "../video_chunks"
    audio_chunks_dir = "../audio_chunks"
  
    os.makedirs(scenes_audio_dir, exist_ok=True)
    os.makedirs(video_chunks_dir, exist_ok=True)
    os.makedirs(audio_chunks_dir, exist_ok=True)

    metadata_file = os.path.join(data_dir, "scene_metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Extract audio for each scene
    for scene in metadata:
        video_path = scene['file_path']
        audio_filename = f"{scene['movie_name']}_scene_{scene['scene_id']}.wav"
        audio_path = os.path.join(scenes_audio_dir, audio_filename)
        if not os.path.exists(audio_path):
            extract_audio(video_path, audio_path)
        scene['audio_file_path'] = os.path.abspath(audio_path)

    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Group by movie
    movies = {}
    for scene in metadata:
        movie = scene['movie_name']
        if movie not in movies:
            movies[movie] = []
        movies[movie].append(scene)

    all_chunk_metadata = []

    for movie, scenes in movies.items():
        scenes.sort(key=lambda x: x['scene_id'])

        # Merge consecutive same audio
        merged_segments = []
        current_segment = [scenes[0]]
        for scene in scenes[1:]:
            if is_audio_same(current_segment[-1]['audio_file_path'], scene['audio_file_path']):
                current_segment.append(scene)
            else:
                merged_segments.append(current_segment)
                current_segment = [scene]
        merged_segments.append(current_segment)

        # Create chunks from merged_segments
        chunks = []
        current_chunk = []
        current_duration = 0
        overlap = 3
        previous_end = 0
        for segment in merged_segments:
            segment_duration = sum(s['duration'] for s in segment)
            if current_duration + segment_duration > 60 or (current_duration > 0 and current_duration >= 45):
                if current_duration >= 45:
                    chunks.append((current_chunk, previous_end, previous_end + current_duration))
                    previous_end += current_duration - overlap
                    current_chunk = segment
                    current_duration = segment_duration
                else:
                    current_chunk.extend(segment)
                    current_duration += segment_duration
            else:
                current_chunk.extend(segment)
                current_duration += segment_duration
        if current_chunk:
            chunks.append((current_chunk, previous_end, previous_end + current_duration))

        # Create video and audio for each chunk
        for i, (chunk_scenes, start_ts, end_ts) in enumerate(chunks):
            if not chunk_scenes:
                continue
            # Concatenate videos using ffmpeg
            video_list_file = os.path.join(video_chunks_dir, f"{movie}_chunk_{i+1}_list.txt")
            with open(video_list_file, 'w') as f:
                for scene in chunk_scenes:
                    f.write(f"file '{scene['file_path']}'\n")
            video_filename = f"{movie}_chunk_{i+1}.mp4"
            video_path = os.path.join(video_chunks_dir, video_filename)
            import subprocess
            subprocess.run(f"ffmpeg -f concat -safe 0 -i \"{video_list_file}\" -c copy \"{video_path}\"", shell=True, check=True)
            os.remove(video_list_file)  # Clean up

            # Concatenate audios using ffmpeg
            audio_list_file = os.path.join(audio_chunks_dir, f"{movie}_chunk_{i+1}_list.txt")
            with open(audio_list_file, 'w') as f:
                for scene in chunk_scenes:
                    f.write(f"file '{scene['audio_file_path']}'\n")
            audio_filename = f"{movie}_chunk_{i+1}.wav"
            audio_path = os.path.join(audio_chunks_dir, audio_filename)
            subprocess.run(f"ffmpeg -f concat -safe 0 -i \"{audio_list_file}\" -c copy \"{audio_path}\"", shell=True, check=True)
            os.remove(audio_list_file)  # Clean up

            duration = end_ts - start_ts
            chunk_data = {
                "movie_name": movie,
                "scene_id_list": [s['scene_id'] for s in chunk_scenes],
                "video_file_path": os.path.abspath(video_path),
                "start_timestamp": round(start_ts, 2),
                "end_timestamp": round(end_ts, 2),
                "duration": round(duration, 2),
                "audio_file_path": os.path.abspath(audio_path)
            }
            all_chunk_metadata.append(chunk_data)

    # Save chunk metadata
    chunk_metadata_file = os.path.join(data_dir, "video_chunk_metadata.json")
    with open(chunk_metadata_file, 'w') as f:
        json.dump(all_chunk_metadata, f, indent=4)

if __name__ == "__main__":
    main()