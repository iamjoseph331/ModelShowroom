FROM python:3.9-slim AS compile-image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
ENV SERVICE_NAME="josephplatform"
RUN python -m pip install --no-cache-dir --upgrade pip==22.3.1
RUN pip install --no-cache-dir -r requirements.txt --use-pep517
RUN pip uninstall --yes opencv-python-headless opencv-python
RUN pip install --no-cache-dir opencv-python-headless==4.6.0.66

FROM python:3.9-slim AS build-image
COPY --from=compile-image /opt/venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"
COPY . .

EXPOSE 6370
USER $SERVICE_NAME
CMD ["python3", "main.py"]