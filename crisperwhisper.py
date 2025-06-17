import gradio as gr
import mimetypes
from transformers import pipeline

# Load the CrisperWhisper model
model = pipeline("automatic-speech-recognition", model="nyrahealth/CrisperWhisper")

# Function to handle file upload and check file type
def process_file(file):
    if file is None:
        return "No file uploaded."
    
    # Get the file name
    file_name = file.name
    
    # Guess the file type based on the file name
    file_type, _ = mimetypes.guess_type(file_name)
    
    # Check if the file is an audio file
    if file_type and file_type.startswith("audio"):
        # Process the audio file with the CrisperWhisper model
        transcription = model(file.name)
        return f"Transcription: {transcription['text']}"
    else:
        return "The uploaded file is not an audio file."

# Create a Gradio interface
with gr.Blocks() as demo:
    # Add a header and instructions
    gr.Markdown("# Audio File Transcription\nUpload an audio file with maximum lenght of 30s  to get its transcription. Supported formats include .wav and .mp3.")
    # File upload component
    file_input = gr.File(label="Upload a file")
    
    # Output component
    output = gr.Textbox(label="Output")
    
    # Define the action when the file is uploaded
    file_input.change(fn=process_file, inputs=file_input, outputs=output)

# Launch the Gradio app on port 8003
demo.launch(server_port=8003)