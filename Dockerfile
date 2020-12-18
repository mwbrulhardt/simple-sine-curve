FROM rayproject/ray-ml

RUN pip install --upgrade pip
RUN pip install tensortrade symfit

WORKDIR /app
