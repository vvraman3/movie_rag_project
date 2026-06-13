import json
import os
import time
import warnings
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the new Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def analyze_audio(audio_path):
    """Processes audio using the 2026 GenAI SDK with JSON Mode."""
    print(f"Uploading: {os.path.basename(audio_path)}...")
    
    # 1. Upload file
    audio_file = client.files.upload(file=audio_path)
    
    # 2. Wait for processing state
    while audio_file.state == "PROCESSING":
        time.sleep(2)
        audio_file = client.files.get(name=audio_file.name)

    if audio_file.state == "FAILED":
        raise ValueError(f"File processing failed: {audio_file.error}")

    # 3. Use constrained JSON output
    # This prevents the "Expecting value: line 1 column 1" error
    model_id = "gemini-3-flash-preview"
    
    prompt = """
    Analyze this Telugu audio clip. 
    Return ONLY a JSON object with these keys:
    - "audio_transcript": Accurate Telugu transcription of dialogue.
    - "emotions": Predominant moods (e.g., Angry, Fearful, Heroic).
    - "acoustic_tags": List of background sounds (e.g., Sword clashing, Horse galloping, Screams, Theme music).
    - "summary_en": A 1-sentence English summary of the audio content.
    """

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[audio_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2  # Lower temperature for consistent JSON
            )
        )
        
        # Cleanup file from cloud to manage storage limits
        client.files.delete(name=audio_file.name)
        
        return json.loads(response.text)
        
    except Exception as e:
        print(f"Generation error: {e}")
        return None

def main():
    # Use absolute paths for reliability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "..", "data", "video_chunk_metadata.json")

    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    # Open with utf-8 for Telugu support
    with open(json_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    for entry in metadata_list:
        audio_path = entry.get("audio_file_path")
        
        # Check if already processed
        if entry.get("audio_transcript"):
            continue

        if audio_path and os.path.exists(audio_path):
            result = analyze_audio(audio_path)
            
            if result:
                # Merge the model output into your metadata
                entry.update(result)
                
                # Save immediately to prevent data loss (Atomic Save)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata_list, f, indent=4, ensure_ascii=False)
            
            # Rate limit safety for Free/Standard tier
            time.sleep(1) 

    print("\nProcessing complete. All Telugu transcripts saved.")

if __name__ == "__main__":
    main()