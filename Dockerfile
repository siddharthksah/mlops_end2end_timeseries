# Start from the official Python image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file to the container and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code to the container
COPY . .

# Expose the port on which the app will run
EXPOSE 8000

# Run the app
CMD ["uvicorn", "stock_prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]
