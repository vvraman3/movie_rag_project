import json
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key from your .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def analyze_video(video_path):
    """Uploads and analyzes video using Gemini multimodal capabilities."""
    print(f"Processing: {video_path}")
    
    # Upload the file to Gemini
    video_file = genai.upload_file(path=video_path)
    
    # Wait for the file to be processed by the API
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed on Google servers.")

    # Prompt for extraction
    prompt = """
    Analyze this video clip and provide a concise summary. 
    Specifically, list the key entities (characters, objects, locations) 
    and the primary emotions or mood displayed in the scene.Extract information about entities, objects, Facial expressions to identify emotions like sadness/laughter/anger/romance/comedy, etc., identify explosions, crowd, vehicles, guns and any other tags helpful for retrieval. Format the output as a clean text description.
    """

    model = genai.GenerativeModel(model_name="gemini-3-flash-preview")
    response = model.generate_content([video_file, prompt])
    
    # Cleanup: Delete the file from the API storage after processing
    genai.delete_file(video_file.name)
    
    return response.text.strip()

def main():
    json_path = os.path.join("..","data", "video_chunk_metadata.json")
    
    # 1. Read existing metadata
    with open(json_path, 'r') as f:
        metadata_list = json.load(f)

    # 2. Process each video entry
    for entry in metadata_list:
        video_path = entry.get("video_file_path")
        
        # Check if file exists and if we haven't already processed it
        if os.path.exists(video_path) and "video_content" not in entry:
            try:
                content_analysis = analyze_video(video_path)
                entry["video_content"] = content_analysis
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
        else:
            print(f"Skipping {video_path} (already processed or file missing)")

    # 3. Save updated metadata back to the same file
    with open(json_path, 'w') as f:
        json.dump(metadata_list, f, indent=4)
    
    print(f"\nSuccess! Metadata updated in {json_path}")

if __name__ == "__main__":
    main()