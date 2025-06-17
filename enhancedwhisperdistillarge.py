# Import necessary libraries used by the code

import torch  # PyTorch for tensor computations and model handling
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline  # Hugging Face Transformers for model and pipeline
import gradio as gr  # Gradio for creating web-based interfaces
from langdetect import detect_langs  # langdetect for language detection
import time
import filetype
import subprocess
import os
import requests  # Requests for making HTTP requests
from dotenv import load_dotenv  # dotenv for loading environment variables from a .env file
import json


from concurrent.futures import ThreadPoolExecutor
from pyannote.audio import Pipeline as audiopipeline

load_dotenv()
token = os.getenv("HF_TOKEN")  # Load Hugging Face token from environment variable
tokenValid = False  # Initialize token validity flag

# Function to verify token validity
def is_token_valid(token):
    print(token)
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
        print("Token verification response:", response.status_code, response.json())
        return response.status_code == 200
    except Exception:
        return False


# Function to convert audio to MP3 using FFmpeg
def convert_to_mp3(input_file, output_file):
    command = ["ffmpeg", "-i", input_file, output_file]
    subprocess.run(command, check=True)
    

def identify_speakers(audio_path):
    pipeline = audiopipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    diarization = pipeline(audio_path)
    speaker_segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        print(f"Speaker {speaker} from {segment.start:.1f}s to {segment.end:.1f}s")
        speaker_segments.append({
        "start": segment.start,
        "end": segment.end,
        "speaker": speaker
        })
    return speaker_segments




# Function to detect the language of the given text
def detect_language(text):
    return detect_langs(text)

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

#Convert the received json timestamps to .srt to be used in youtube, etc.
def convert_to_srt(timestamps):
    srt_content = ""
    for index, entry in enumerate(timestamps):
        start_time = entry["timestamp"][0]
        end_time = entry["timestamp"][1] 
        if start_time == None:
            print("No start time, should just continue")
            continue
        if end_time ==None: 
            end_time = start_time + 3
        text = entry["text"]
        #print("{}-{}".format(start_time, end_time))
        # Convert start and end times to SRT format
        start_time_srt = format_time(start_time)
        end_time_srt = format_time(end_time)

        # Append to SRT content
        srt_content += f"{index + 1}\n{start_time_srt} --> {end_time_srt}\n{text}\n\n"

    return srt_content

def get_mime_type(file_path):
    kind = filetype.guess(file_path)
    if kind is None:
        return "Unknown MIME type"
    return kind.mime

# Function to transcribe audio files
def transcribe(audio, selected_file, option):
    start = time.time()
    if selected_file != "None":
        audio = selected_file
    if audio is None:
        # Return a  message if no file is uploaded        
        gr.Info("You need to upload audio or video file to use this")        
    # Use the pipeline to transcribe the audio   
    kind = get_mime_type(audio)
    print(f"Detected MIME type: {kind}")
    if kind.startswith("audio") or kind.startswith("video"):        
        try:
            with ThreadPoolExecutor() as executor:
                future_transcription = executor.submit(pipe, audio, return_timestamps=True if option == "Yes" else False)
                if token and is_token_valid(token):
                    try:
                        print("Token is valid. Identifying speakers...")
                        future_speaker_identification = executor.submit(identify_speakers, audio)
                        speaker_segments = future_speaker_identification.result()
                        print("Speaker identification completed.")
                    except Exception as e:
                        print(f"Failed to identify speakers: {e}")
                        speaker_segments = json.loads("Could not identify speakers..")
                else:
                    if not token:
                        print("HF_TOKEN environment variable not set. Skipping speaker identification.")
                    else:
                        print("Invalid HF_TOKEN. Skipping speaker identification.")                
                result = future_transcription.result()
                
            srt = convert_to_srt(result["chunks"]) if option == "Yes" else "Not requested"        
        except Exception as e:
            gr.Info("Needs to convert the uploaded file into another format before transcribing.")
            #print(f"Error during transcription: {e}")
            mp3_file = audio + ".mp3"
            convert_to_mp3(audio, mp3_file)
            with ThreadPoolExecutor() as executor:
                future_transcription = executor.submit(pipe, mp3_file, return_timestamps=True if option == "Yes" else False)
                future_speaker_identification = executor.submit(identify_speakers, mp3_file)
                result = future_transcription.result()
                speaker_segments = future_speaker_identification.result()
            srt = convert_to_srt(result["chunks"]) if option == "Yes" else "Not requested"

        acc = detect_language(result["text"])
        chunks = result.get("chunks", '{"message": "Not requested"}')
        end = time.time()
        duration = "### Processing took: {} seconds".format(round(end - start, 2))
        return result["text"], result, acc, chunks, srt, duration, speaker_segments   
    else:        
        gr.Info("Uploaded file is not audio or video or cannot be handled correctly. Please try another file format.")        

    

# Function to update the file upload component
def update_file_upload(selected_file):
    if selected_file != "None":
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)

def main():  # Basic Python part begins
    tokenValid = is_token_valid(token)  # Check if the Hugging Face token is valid
    # Determine the device to use (GPU if available, otherwise CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Set the appropriate torch data type based on the device
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # List of multilingual models
    multilingual_models = [
        "openai/whisper-large-v3-turbo", 
        "openai/whisper-tiny",
        "Berly00/whisper-large-v3-spanish"
    ]
    def load_model(selected_model):
        global pipe
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            selected_model,  # The identifier of the pre-trained model to load
            torch_dtype=torch_dtype,  # Set the data type for the model's tensors
            low_cpu_mem_usage=True,  # Optimize the model loading to use less CPU memory
            use_safetensors=True  # Use the safetensors format for loading the model
        ).to(device)  # Move the model to the selected device (GPU if available, otherwise CPU)

        # Load the processor associated with the model
        processor = AutoProcessor.from_pretrained(selected_model)

        # Create a pipeline for automatic speech recognition
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=20,  # Length of audio chunks in seconds for processing
            batch_size=30,  # Batch size for inference
            torch_dtype=torch_dtype,
            device=device
        )
        
    
    # Load the default model
    load_model(multilingual_models[0])

    # List of internal example files provided with the packages
    example_files = [
        "None",
        "examples/English1.mp3",
        "examples/English-tooLong.mp3",
        "examples/Finnish-Olympic1952-100m-Final.mp3",
        "examples/German1.mp3",
        "examples/OSR_us_000_0010_8k.wav",
        "examples/Spanish1.mp3",
        "examples/Video-Teddy-Roosevelt.mp4"
    ]

    # Create a Gradio interface --> UI part starts
    with gr.Blocks(title="Digitalia Speech to Text") as demo:
        # Add a header and instructions
        gr.Markdown("# Audio/Video File Transcription\n### Upload a file or use a provided example to get its transcription. Supported formats include all common audio and video formats")

        # Dropdown to select the model
        model_dropdown = gr.Dropdown(
            label="Select a multilingual model",
            choices=multilingual_models,
            value=multilingual_models[0]
        )
       
        # File upload component
        file_input = gr.File(label="Upload your own file")
        example_dropdown = gr.Dropdown(label="Or select a provided example file", choices=example_files, value="None")

        # Radio button component
        gr.Markdown("### Choose if to create timestamps or not. If created, provided in json and srt formatted")
        radio_options = gr.Radio(label="Create timestamps?", choices=["Yes", "No"], value="No")

        # Button to trigger the transcription
        transcribe_button = gr.Button("Transcribe")
        time_output = gr.Markdown()

        # Output components for displaying results
        with gr.Tab("Transcription"):
            accuracy_output = gr.Textbox(label="Language and accuracy")
            text_output = gr.Textbox(label="Transcribed Text")
            jsontext_output = gr.JSON(label="Full text result as JSON")

        with gr.Tab("Timed transcription"):
            with gr.Row():
                jsonstamps_output = gr.JSON(label="Timestamps as JSON")
                srt_output = gr.Textbox(label="Timestamps as srt format", lines=20)
        #Only shows speaker recognition tab if token is valid
        if tokenValid:
            with gr.Tab("Speaker recognition"):
                with gr.Row():
                    speaker_output = gr.JSON(label="Speakers")

        # Hide or show the file upload part depending on the example_file selection
        example_dropdown.change(fn=update_file_upload, inputs=example_dropdown, outputs=file_input)

        # Update the model when a new model is selected
        model_dropdown.change(fn=load_model, inputs=model_dropdown, outputs=[])

        # Define the action when the button is clicked
        transcribe_button.click(fn=transcribe, inputs=[file_input, example_dropdown, radio_options],
                                outputs=[text_output, jsontext_output, accuracy_output, jsonstamps_output, srt_output, time_output, speaker_output])

    # Launch the Gradio app on the specified port and server name
    demo.launch(server_port=8004, server_name="0.0.0.0")
    # Below is used to launch this on Hippu and access via https://memorylab.fi/demot/asr/
    #demo.launch(server_port=8004, server_name="0.0.0.0", root_path="/demot/asr/")

if __name__ == "__main__":
    main()
