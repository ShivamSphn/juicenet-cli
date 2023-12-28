FROM nikolaik/python-nodejs:python3.12-nodejs20

WORKDIR /node

RUN npm install -g yencode
RUN npm install -g git+https://github.com/animetosho/Nyuu.git --production --unsafe-perm
RUN npm install -g @animetosho/parpar

WORKDIR /app

COPY . .

RUN pip install .

WORKDIR /media

ENTRYPOINT ["python", "-m", "juicenet", "--config", "/config/juicenet.docker.yaml"]