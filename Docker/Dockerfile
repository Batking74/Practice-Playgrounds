# Parent Image 
FROM node:latest


# Change to app directory inside node image file system (linux) so when I run commands in the future (such as RUN npm start) they will execute from this (app) working directory
WORKDIR /app



COPY package.json .


# The RUN command runs as the image is being built (so at build time)! The image is not a running application it's just blueprint for container so it doesn't make sense to start the server using (node server.js) because I am not trying to run the application when I am building the image!
RUN npm i


# Copy all my files and folders from my current directory and copy them into the root directory of my image (node) file system (.) which will be a linux file system
COPY . .


# Inside the server.js file I listen for port 5000. But my app will be running inside the container, so the port is going to be owned by the container as well. NOT MY COMPUTER, that's why it needs to be exposed.
EXPOSE 5000


# CMD allows me to specify any commands that should be run at runtime when the container begins to run! Allows me to spin up my app inside the container.
CMD [ "npm", "start" ]