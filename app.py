import os
from random import randint

from PIL import Image, ImageFont, ImageDraw
from flask import Flask, render_template, redirect, url_for, request

from model import model_to_predict, cnn_digits_predict

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def button_random():
    """получение слуайного числа, по умолчанию от mn=0, до mx=10"""
    if request.form.get('random') == 'random':
        del_image()
        r = get_random()
        create_numb_image(r)
        return render_template("index.html", numb_message='image')

    if request.form.get('ml') == 'ml':
        file = 'static/img/image.jpg'
        p = predict_image(file)
        return render_template("index.html", numb_message='image', predict_numb=p)

    return render_template("index.html", numb_message='default/none', predict_numb='')


def get_random(mn=0, mx=9):
    """получение слуайного числа, по умолчанию от mn=0, до mx=10"""
    return randint(mn, mx)


def create_numb_image(numb):
    """получение изображения из переданой строки"""
    # image = Image.open("img/grid.png")  # фон изображения
    image = Image.new(mode="RGB", size=(28, 28), color="white")  # фон изображения
    font = ImageFont.truetype("static/font/Roboto-Bold.ttf", size=28)
    # font = ImageFont.load_default()  # шрифт
    draw = ImageDraw.Draw(image)  # открытие изображения для редактирования
    text = str(numb)  # вписываемый текст
    text_position = (2, 0)  # позиция
    text_color = "black"  # black color
    draw.text(text_position, text, fill=text_color, font=font)
    image.save(f"static/img/image.jpg")


def del_image():
    """удаление старого изображения"""
    file = 'static/img/image.jpg'
    if os.path.isfile(file):  # если файл уже существует, удаляем
        os.remove(file)


def predict_image(image):
    """предсказание цифры"""
    model = model_to_predict()
    r = cnn_digits_predict(model, image)
    return r[0]


if __name__ == '__main__':
    app.run(debug=True)
