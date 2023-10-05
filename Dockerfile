FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN set -x && apt-get update && apt-get dist-upgrade -y && apt-get install -y --no-install-recommends ffmpeg

COPY . .

#CMD ["gunicorn", "--bind", "0.0.0.0", "--timeout", "0", "--workers", "4", "app:app"]
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 0 --workers 4 app:app