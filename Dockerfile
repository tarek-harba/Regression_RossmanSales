FROM python:3.10

# Set environment variables to prevent Pipenv from creating virtual environments
ENV PIPENV_VENV_IN_PROJECT=false \
    PIP_NO_CACHE_DIR=1

# Install Pipenv
RUN pip install pipenv

# Set the working directory
WORKDIR /app

# Copy application files
COPY ./Datasets /app/Datasets
COPY ./docker.py /app/docker.py
# Copy only Pipfile.lock to install dependencies
COPY Pipfile Pipfile.lock /app/

# Install dependencies with Pipenv
RUN pipenv install --system --deploy

# Specify the default command to run the app
CMD ["python", "docker.py"]