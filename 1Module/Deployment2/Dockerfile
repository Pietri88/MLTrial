# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install any needed packages specified in requirements.txt
COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Run serve_model.py when the container launches
CMD ["python", "serve_model.py"]