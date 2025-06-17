# Import necessary libraries used by the code

import torch  # PyTorch for tensor computations and model handling
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline  # Hugging Face Transformers for model and pipeline
import gradio as gr  # Gradio for creating web-based interfaces
from langdetect import detect_langs  # langdetect for language detection

# Function to detect the language of the given text
def detect_language(text):
    return detect_langs(text)

# Function to transcribe audio files
def transcribe(audio, selected_file):
    if selected_file != "None":
        audio = selected_file
    if audio is None:
        # Return a  message if no file is uploaded
        return "Please upload an audio file first.", None, None
    # Use the pipeline to transcribe the audio
    result = pipe(audio)
    # Detect the language of the transcribed text
    acc = detect_language(result["text"])
    # Return the transcribed text, the full result, and the detected language
    return result["text"], result, acc

# Function to update the file upload component
def update_file_upload(selected_file):
    if selected_file != "None":
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)

def main(): #Basic Python part begins
    
    # Determine the device to use (GPU if available, otherwise CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Set the appropriate torch data type based on the device
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model identifier for the pre-trained model
    model_id = "openai/whisper-large-v3-turbo"  # https://huggingface.co/openai/whisper-large-v3-turbo

    """ Load the pre-trained model with specified configurations
    Downloads the model from huggingface before loading into memory, 
    if run via dockerfile the model is downloaded during the build stage"""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,  # The identifier of the pre-trained model to load
        torch_dtype=torch_dtype,  # Set the data type for the model's tensors (float16 if using GPU, otherwise float32, CPUs don't have support for float16)
        low_cpu_mem_usage=True,  # Optimize the model loading to use less CPU memory
        use_safetensors=True  # Use the safetensors format for loading the model, which is more secure and efficient
    ).to(device)  # Move the model to the selected device (GPU if available, otherwise CPU)

    # Load the processor associated with the model
    processor = AutoProcessor.from_pretrained(model_id)

    # Create a pipeline for automatic speech recognition
    #pipeline simplifies the process of using pre-trained models from huggingface
    global pipe
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model, #defined above
        tokenizer=processor.tokenizer, #defined above
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,  # Length of audio chunks  in seconds for processing
        batch_size=16,  # Batch size for inference, number of chunks processed parallel
        torch_dtype=torch_dtype, #defined above, float16 or float32
        device=device, #defined above, cuda:0 or cpu
    )

    # List of internal example files provided with the packages, feel free to alter or convert to dynamic
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
    with gr.Blocks() as demo:
        # Add a header and instructions
        gr.Markdown("# Audio/Video File Transcription\nUpload a file or use a provided example to get its transcription. Supported formats include all common audio and video formats")
        # File upload component
        file_input = gr.File(label="Upload your own file")
        example_dropdown = gr.Dropdown(label="Or select a provided example file", choices=example_files, value="None") 
        # Button to trigger the transcription
        transcribe_button = gr.Button("Transcribe")
        # Output components for displaying results
        with gr.Tab("Transcription"):
            text_output = gr.Textbox(label="Transcribed Text")
            result_output = gr.JSON(label="Json Result")
            accuracy_output = gr.Textbox(label="Language and accuracy")
        with gr.Tab("Timed transcription"):
            gr.Text("Nothing here yet - as a homework feel free to implement this by yourself. It's not hard :)")
        
        
        #Hide or show the file upload part depending on the example_file selection
        example_dropdown.change(fn=update_file_upload, inputs=example_dropdown, outputs=file_input)
        
        # Define the action when the button is clicked
        transcribe_button.click(fn=transcribe, inputs=[file_input, example_dropdown], outputs=[text_output, result_output, accuracy_output])

    # Launch the Gradio app on the specified port and server name
    demo.launch(server_port=8004, server_name="0.0.0.0")

if __name__ == "__main__":
    main()
