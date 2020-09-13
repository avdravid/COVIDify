# Main Flask application file

from flask import Flask, render_template, url_for, flash, redirect, request, g, send_from_directory
from forms import RegistrationForm, LoginForm, UploadForm, ButtonForm
from werkzeug.utils import secure_filename
import os
import shutil
from os import path
import time
import logging

app = Flask(__name__)

app.config['SECRET_KEY'] = '6e8af7ecab9c7e41e861fa359a89f385'

# support bundles should be .tar.gpg, process will fail and exit otherwise
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS


def gan_generate(imagepath):
    # generate new image from upload
    time.sleep(5)
# check if file has correct extension


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(path.join(app.root_path, 'static'), 'virus.png', mimetype='image/vnd.microsoft.icon')

# render home landing page


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/info")
def info():
    return render_template("info.html")


@app.route("/results")
def results():
    return render_template("results.html")

# path for upload page to upload file


@app.route("/upload", methods=['GET', 'POST'])
def upload():
    logging.info('Uploading file')
    form = UploadForm()

    if form.validate_on_submit():
        logging.info('File validated')
        if form.file.data:
            filename = secure_filename(form.file.data.filename)
        else:
            flash('Please upload a file!', 'danger')
            logging.error('Attempted submit without file')
            return render_template('upload.html', form=form)
        if not allowed_file(filename):
            logging.info('Illegal file extension')
            flash('File must have extension .jpg, .jpeg, or .png', 'danger')
        else:
            flash('File accepted!', 'success')
            # save accepted file to uploads folder
            try:
                form.file.data.save('uploads/' + filename)
                logging.info('File saved locally')
            except:
                logging.info('Failed to save file')

            logging.info('Image generation process beginning')

            # here trigger the loady wheel

            gan_generate("dummy_path")
            return render_template('loading.html')

    return render_template('upload.html', form=form)


@app.route("/loading")
def loading():
    uploads = os.listdir("./uploads")
    if uploads:
        with open('./uploads/'+uploads[0], 'rb') as f:
            gan_generate(f)
        f.close()
        return render_template("results.html")
    return render_template("loading.html")


# only true if you run script directly, if imported will be false
if __name__ == '__main__':
    logging.basicConfig(filename='app.log', format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w', level=logging.INFO)
    logging.info('Started')
    app.run(debug=True)
    logging.info('Finished')
