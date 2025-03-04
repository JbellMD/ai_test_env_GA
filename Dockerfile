FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set up working directory
WORKDIR /workspace
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set up Jupyter Notebook
RUN pip install jupyterlab
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]