import json
import os
import time
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

# Load API Key
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def analyze_video_with_retry(video_path, max_retries=5):
    """Uploads and analyzes video with automatic retries for Quota limits."""
    attempt = 0
    wait_time = 45  # Initial wait time for video (larger than audio due to file size)

    while attempt < max_retries:
        try:
            print(f"Uploading: {os.path.basename(video_path)}...")
            video_file = client.files.upload(file=video_path)
            
            # Wait for processing
            while video_file.state == "PROCESSING":
                time.sleep(5)
                video_file = client.files.get(name=video_file.name)

            if video_file.state == "FAILED":
                print(f"Server-side failure for {video_path}")
                return None

            # Prompt for extraction - fine-tuned for your Vector Search requirements
            prompt = """
            Analyze this video clip and provide:
            1. A concise summary of the scene.
            2. List key entities: characters, objects, locations.
            3. Detect facial expressions/emotions: sadness, laughter, anger, romance, comedy.
            4. Identify action tags: explosions, crowd, vehicles, guns, swordplay, etc.
            Format as clean text for semantic indexing.
            """

            model_id = "gemini-3-flash-preview"
            response = client.models.generate_content(
                model=model_id,
                contents=[video_file, prompt]
            )
            
            # Cleanup storage
            client.files.delete(name=video_file.name)
            return response.text.strip()

        except errors.ClientError as e:
            if "429" in str(e):
                attempt += 1
                print(f"Quota exceeded. Attempt {attempt}/{max_retries}. Waiting {wait_time}s...")
                time.sleep(wait_time)
                wait_time *= 2  # Double the wait for next time
            else:
                print(f"API Error: {e}")
                break
        except Exception as e:
            print(f"Unexpected Error: {e}")
            break
            
    return None

def main():
    # Use relative paths for better portability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.normpath(os.path.join(base_dir, "..", "data", "video_chunk_metadata.json"))

    if not os.path.exists(json_path):
        print(f"Metadata file not found at {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    processed_count = 0
    for entry in metadata_list:
        video_path = entry.get("video_file_path")
        
        # RESUME LOGIC: Skip if content already exists or file is missing
        if not video_path or not os.path.exists(video_path):
            continue
        
        if "video_content" in entry and entry["video_content"]:
            continue

        analysis = analyze_video_with_retry(video_path)
        
        if analysis:
            entry["video_content"] = analysis
            processed_count += 1
            
            # Intermediate save so you don't lose progress if it crashes
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, indent=4, ensure_ascii=False)
            
            # MANDATORY COOL-DOWN: Respects the Free Tier 15 RPM limit
            print(f"Successfully processed {os.path.basename(video_path)}. Cooling down...")
            time.sleep(12) 

    print(f"\nProcessing complete. {processed_count} new chunks analyzed.")

if __name__ == "__main__":
    main()