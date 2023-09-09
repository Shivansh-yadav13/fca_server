FROM python:3.9

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN set -x && apt-get update && apt-get dist-upgrade && apt-get install -y --no-install-recommends ffmpeg

#COPY nginx_config.conf /etc/nginx/conf.d/virtual.conf
#COPY . .

#EXPOSE 80

#CMD ["python", "app.py"]
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]
