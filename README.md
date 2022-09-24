# Overview
This packages is a very basic script that captures video from the webcam, detects all faces in the video and then runs the DeepFace emotional classifier on each face. The output of the emotional classfications will be written to a Unix pipe to be consumed by any other process.

# Instructions
Everything is contained in `emotional_detection/main.py`. You simply install the dependencies and run. This project works out of the box with [poetry](https://python-poetry.org/), but you could install the dependencies using any dependency manager + virtual environment combination you choose.
