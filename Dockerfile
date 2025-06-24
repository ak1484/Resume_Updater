# Use official Python 3.12 image as base
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install uv (for dependency management)
RUN pip install --no-cache-dir uv

# Copy project files
COPY . .

# Copy .env file for environment variables
COPY .env .env

# Set uv to use system Python (not venv)
ENV UV_SYSTEM_PYTHON=1

# Install dependencies using requirements.txt
RUN uv pip install -r requirements.txt


# Expose port 8000
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 

# docker run -d -p 8000:8000 --env-file .env --name resume-updater resume-updater