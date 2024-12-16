# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies, including vim
RUN apt-get update && apt-get install -y vim && apt-get clean

# Copy requirements first to leverage Docker's caching
COPY requirements.txt ./

# Upgrade pip and install the Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code into the working directory
COPY . .

# Expose the port your app runs on (optional if you're doing a web app)
EXPOSE 8080

# Set environment variables as needed (optional)
# ENV MY_ENV_VAR=value

# Run your application
CMD ["python", "translator_fireworks_rest_app.py"]
