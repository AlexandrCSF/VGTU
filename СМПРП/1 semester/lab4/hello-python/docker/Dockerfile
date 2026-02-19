FROM python:3.7

COPY /app/ /code/
WORKDIR /code

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "main.py"]