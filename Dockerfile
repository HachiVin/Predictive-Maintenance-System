# Menggunakan base image Python slim yang ringan
FROM python:3.10-slim

# Menentukan working directory di dalam container
WORKDIR /app

# Mengopi file requirements dan menginstall library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mengopi direktori app dan models ke dalam container
COPY ./app ./app
COPY ./models ./models

# Expose port 8000 untuk FastAPI
EXPOSE 8000

# Perintah untuk menjalankan FastAPI server di dalam container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]