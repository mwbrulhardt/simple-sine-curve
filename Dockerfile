FROM rayproject/ray-ml

RUN pip install --upgrade pip
RUN pip install tensortrade==1.0.1b0 symfit

WORKDIR /app
