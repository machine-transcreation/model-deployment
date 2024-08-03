FROM python:3.8.5-buster

WORKDIR /

COPY check.sh /
RUN chmod +x /check.sh

ENTRYPOINT ["/check.sh"]

COPY builder/requirements.txt .
RUN pip install -r requirements.txt

COPY model/update_yaml_paths.py model/handler.py model/

CMD ["sh", "-c", "python -u model/update_yaml_paths.py && python -u model/handler.py"]
