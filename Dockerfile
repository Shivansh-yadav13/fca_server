FROM python:3.9

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN set -x && apt-get update && apt-get dist-upgrade && apt-get install -y --no-install-recommends ffmpeg

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]