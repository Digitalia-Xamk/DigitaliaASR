# Use the official Python 3.11 slim image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install ffmpeg and other dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg less \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script to download the model
COPY download_model.py .

# Run the script to download the model
RUN python download_model.py

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the Gradio app will run on
EXPOSE 8004

# Command to run the application
CMD ["python", "enhancedwhisperdistillarge.py"]
#CMD ["bash"]