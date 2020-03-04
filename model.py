
from flask import Flask, Response
from flask import request
from flask import render_template
import Links_Classification
from PIL import Image
##import matplotlib.pyplot as plt
import matplotlib.image as mpimg

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("my-form.html")

@app.route('/', methods=['POST'])
def my_form_post():
	link = request.form['link']
	image = Image.open("image.jpg")
	#img = mpimg.imread('image.jpg')
	#image = plt.imshow(img)
	return render_template("my-form.html", output=Links_Classification.lime_explanation(link), image=image.show())

if __name__ == '__main__':
    app.run(debug=True)
