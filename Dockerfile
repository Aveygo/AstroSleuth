FROM pytorch/pytorch:latest

WORKDIR /app
RUN pip install --no-cache-dir numpy streamlit pillow
COPY . /app
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--browser.gatherUsageStats", "false"]