# batutti web app, with Audiveris for optical music recognition.
# Built for Hugging Face Spaces (Docker), which serves on port 7860.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
ARG AUDIVERIS_VERSION=5.11.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip ca-certificates wget fontconfig \
    && wget -q -O /tmp/audiveris.deb \
        "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu24.04-x86_64.deb" \
    && apt-get install -y --no-install-recommends /tmp/audiveris.deb \
    && rm /tmp/audiveris.deb && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV PATH="/home/user/venv/bin:$PATH" AUDIVERIS=/opt/audiveris/bin/Audiveris

RUN python3 -m venv /home/user/venv
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY --chown=user . .
EXPOSE 7860
# one worker: running jobs are tracked in memory; threads serve page updates
CMD ["gunicorn", "app:server", "--workers", "1", "--threads", "8", \
     "--timeout", "120", "--bind", "0.0.0.0:7860"]
