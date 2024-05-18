import io
import json
from flask import url_for, session, Flask, request, jsonify, make_response, redirect, render_template
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

import numpy as np
import torch
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your secret key'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'test1'

mysql = MySQL(app)

import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_1 = torch.hub.load('./yolov', 'custom', path='./model/number_detect_best.pt', source='local')  # local repo#
model = torch.hub.load('./yolov', 'custom', path='./model/license_plate_best.pt', source='local')  # local repo
model.conf = 0.45
model_1.conf = 0.45


def get_image_from_bytes(binary_image, max_size=1024):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize(
        (
            int(input_image.width * resize_factor),
            int(input_image.height * resize_factor),
        )
    )
    return resized_image


@app.route("/", methods=["GET", "POST"])
def predict():
    number_plate = ""
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=640)
        bbox_raw = results.xyxy[0][0]
        bbox = []
        for bound in bbox_raw:
            bbox.append(int(bound.item()))
        bbox = bbox[:4]
        img_array = np.array(img)
        crop_img = img_array[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        res2 = model_1(crop_img, size=640)

        res2.render()  # updates results.imgs with boxes and labels
        for img in res2.ims:
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image3.jpg", format="JPEG")

        # for debugging
        data = res2.pandas().xyxy[0].to_json(orient="records")
        array = json.loads(data)

        count = 0

        for i in array:
            if 35 <= i['class'] <= 99:
                number_plate += i['name'] + ' Metro'

            elif 11 <= i['class'] <= 35 or 100 <= i['class'] <= 101:
                number_plate += ' ' + i['name'] + ' '
            count += 1
        # Filter data where class is between 0 and 9
        filtered_data = [i for i in array if 0 <= i['class'] <= 9]

        # Sort the filtered data by ymax in ascending order
        filtered_data.sort(key=lambda i: i['xmax'])
        for i in filtered_data:
            number_plate += i['name']
        print(number_plate)

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM accounts WHERE plate_name = % s', (number_plate,))
        account = cursor.fetchone()
        msg = ''
        if account:

            cursor.execute('UPDATE accounts SET ammount =50 WHERE plate_name =% s', (number_plate,))
            mysql.connection.commit()

        else:
            msg = 'no number plate register'

        return msg

    # if request.method == 'GET':
    #   number_plate = plate_name
    # cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    #   cursor.execute('SELECT * FROM accounts WHERE number_plate = %s ', (number_plate,))
    #    account = cursor.fetchone()
    #   if account:
    #      session['loggedin'] = True

    return render_template("index.html")


ALLOWED_EXTENSIONS = ['mp4']


@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''

    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM accounts WHERE username = %s \
            AND password = %s', (username, password,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']

            msg = 'Logged in successfully !'

            return render_template('user.html', msg=account['ammount'])
        else:
            msg = 'Incorrect username / password !'
    return render_template('login.html', msg=msg)


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'plate_name' in request.form:
        username = request.form['username']
        password = request.form['password']
        plate_name = request.form['plate_name']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'

        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'name must contain only characters and numbers !'
        else:
            # Prepare the SQL query
            sql = "INSERT INTO accounts (username, password, plate_name) VALUES (%s, %s, %s)"
            values = (username, password, plate_name)

            # Execute the query
            cursor.execute(sql, values)
            #  cursor.execute('''"INSERT INTO accounts VALUES ( %s, %s, %s, %s, %s))\'"''',
            #                (username, password, email, country, postalcode))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg=msg)


@app.route("/user")
def user():
    if 'loggedin' in session:
        return render_template("user.html")
    return redirect(url_for('login'))


@app.route("/display")
def display():
    if 'loggedin' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s',
                       (session['id'],))
        account = cursor.fetchone()
        cursor.execute('UPDATE accounts SET ammount =0 WHERE id =% s', (session['id'],))
        mysql.connection.commit()

        return render_template("display.html", account=account)
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run()
