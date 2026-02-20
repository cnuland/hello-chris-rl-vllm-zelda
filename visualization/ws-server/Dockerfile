# WebSocket relay server for Oracle of Seasons RL map visualization.
# Inspired by LinkMapViz (https://github.com/Xe-Xo/LinkMapViz)
FROM node:20-alpine
WORKDIR /app
COPY package.json .
RUN npm install --production
COPY index.js .
EXPOSE 3344
CMD ["node", "index.js"]
