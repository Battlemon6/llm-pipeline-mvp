# 1. Base Image: Use an official lightweight Python image
FROM python:3.12-slim

# 2. Working Directory: Set the working directory inside the container
WORKDIR /app

# 3. Copy and Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Application Code
COPY . .

# 5. Expose Port
EXPOSE 8000

# 6. Run Command: Start the Uvicorn server to run forever
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]