# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies, including vim
RUN apt-get update && apt-get install -y vim && apt-get clean

COPY ./translator /app/translator

# Upgrade pip and install the Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r translator/requirements.txt


# Expose the port your app runs on (optional if you're doing a web app)
EXPOSE 8080

# Set environment variables as needed (optional)
 ENV PYTHONPATH=/app

# Run your application
CMD ["python", "-m", "translator.translator_fireworks_rest_app"]
