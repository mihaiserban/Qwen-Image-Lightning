FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN huggingface-cli download lightx2v/Qwen-Image-Lightning --local-dir /workspace/models
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "handler.py"]