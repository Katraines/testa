# Testa

## What this is
This is the backend for our new project Idealisk. It takes an image as an input and outputs the facial data it was able to extract from the image.

## How to start
You need pipenv to get started, it also uses OpenCV. 
At the time of writing, we recommend using Python 3.7.8 and OpenCV 4.4.0; other versions may cause compatibility issues. 

To get started, run the below on the terminal of your choice:
```
git clone https://github.com/Katraines/testa
py -m pip install pipenv
py -m pipenv install
py -m pipenv run flask run -p 80
```
## What we learned
+ Pipenv & Venv
+ OpenCV & Tensorflow

## Status checkpoints
+ *Coughs* Add real data modules instead of fake ones.
+ *Coughing intensifies* Train databases for recognising ethnicity, hair colour, eye colour, and other facial traits.
