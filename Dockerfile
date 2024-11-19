FROM python:3.10

# Copy app code and set working directory
COPY . /simrec/
WORKDIR /simrec/

# Upgrade pip and install requirements
RUN pip3 install -U pip
RUN pip3 install -r streamlit_app/app_requirements.txt
# Expose port you want your app on
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run
ENTRYPOINT ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]