# Use official Python 3.10 image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files from your GitHub repo to the container's /app directory
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 10000 (Flask default or change it if needed)
EXPOSE 10000

# Start the Flask app (edit if your app runs differently)
CMD ["python", "app.py"]
