# Use a conda-based image
FROM mambaorg/micromamba:0.25.1

# Create a non-root user
RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

# Copy environment file and create conda environment
COPY --chown=appuser:appuser environment.yml /home/appuser/app/environment.yml
RUN micromamba install -y -n base -f /home/appuser/app/environment.yml && \
    micromamba clean --all --yes

# Activate the conda environment for subsequent commands
SHELL ["micromamba", "run", "-n", "base", "/bin/bash", "-c"]

# Copy the application code
COPY --chown=appuser:appuser ./app /home/appuser/app/app

# Set permissions for output directory
RUN mkdir -p /home/appuser/app/outputs

# Expose the port the app runs on
EXPOSE 8000

# Set the default command to run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
