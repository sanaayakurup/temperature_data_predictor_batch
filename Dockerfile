#Use the python lightweight image
FROM python:3.11-slim 

#set Working dir inside the container
WORKDIR /app

#Copy requirements.txt
COPY requirements.txt .


#Install Dependencies within the container 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the src folder into the container
COPY src/ ./src/
COPY models/ /models/
COPY .env .env

# Set PYTHONPATH so imporDockerfilets from src work
ENV PYTHONPATH=/app/src

# Set default command to run inference.py
# This can be overridden with docker run <image> python src/inference.py
CMD ["python", "src/inference.py"]