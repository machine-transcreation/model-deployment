FROM python:3.8.5-buster

WORKDIR /

# COPY check.sh /
# RUN chmod +x /check.sh

# ENTRYPOINT ["/check.sh"]

COPY builder/requirements.txt .
RUN pip install -r requirements.txt

ADD src .

CMD ["sh", "-c", "python -u src/update_yaml_paths.py && python -u src/handler.py"]
