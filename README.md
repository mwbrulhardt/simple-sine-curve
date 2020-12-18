# Using TensorTrade for Making a Simple Trading Algorithm

This project serves as a guide for how to make basic trading algorithms in TensorTrade. This code can either run locally or within a Docker container. Just make sure that all the libraries needed are properly downloaded.

```sh
$ pip install -r requirements
```

If you are going to run the code locally use,
```sh
$ python scripts/main.py
```

If you are going to run the in the docker container use,
```sh
$ docker build -t ssc .
$ docker run -it -v <absolute-project-path>:/app --entrypoint /bin/bash ssc
$ python scripts/main.py
$ exit
```
