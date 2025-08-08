# Dev image
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .[dev]
CMD ["pytest", "-q"]
