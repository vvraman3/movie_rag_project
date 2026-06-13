import json
import os
import time
import warnings
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def analyze_audio_with_retry(audio_path, max_retries=5):
    """Processes audio with automatic retries for Quota/Rate limits."""
    attempt = 0
    wait_time = 30  # Start with 30 seconds if we hit a 429

    while attempt < max_retries:
        try:
            print(f"Uploading: {os.path.basename(audio_path)}...")
            audio_file = client.files.upload(file=audio_path)
            
            while audio_file.state == "PROCESSING":
                time.sleep(5)
                audio_file = client.files.get(name=audio_file.name)

            if audio_file.state == "FAILED":
                return None

            model_id = "gemini-3-flash-preview"
            prompt = """
            Analyze this Telugu audio clip. Return ONLY a JSON object:
            - "audio_transcript": Accurate Telugu transcription.
            - "emotions": Predominant moods.
            - "acoustic_tags": List of background sounds.
            - "summary_en": 1-sentence English summary.
            """

            response = client.models.generate_content(
                model=model_id,
                contents=[audio_file, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            client.files.delete(name=audio_file.name)
            return json.loads(response.text)

        except errors.ClientError as e:
            if "429" in str(e):
                attempt += 1
                print(f"Quota exhausted. Retry {attempt}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff
            else:
                print(f"API Error: {e}")
                break
        except Exception as e:
            print(f"Unexpected Error: {e}")
            break
            
    return None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.normpath(os.path.join(base_dir, "..", "data", "video_chunk_metadata.json"))

    with open(json_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    for entry in metadata_list:
        audio_path = entry.get("audio_file_path")
        
        # Resume check
        if entry.get("audio_transcript"):
            continue

        if audio_path and os.path.exists(audio_path):
            result = analyze_audio_with_retry(audio_path)
            
            if result:
                entry.update(result)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata_list, f, indent=4, ensure_ascii=False)
                
                # Mandatory delay to respect Free Tier RPM (Requests Per Minute)
                time.sleep(10) 
                
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()