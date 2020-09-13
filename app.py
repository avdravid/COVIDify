# Main Flask application file

from flask import Flask, render_template, url_for, flash, redirect, request, g, send_from_directory
from forms import RegistrationForm, LoginForm, UploadForm, ButtonForm
from werkzeug.utils import secure_filename
import os
import shutil
from os import path
import time
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')

app = Flask(__name__)

app.config['SECRET_KEY'] = '6e8af7ecab9c7e41e861fa359a89f385'

# support bundles should be .tar.gpg, process will fail and exit otherwise
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS


def gan_generate():
    # generate new image from upload here
    # the uploaded image will be in static/uploads/original.jpg
    time.sleep(2)
    # write new image to static/uploads/covidified.jpg
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
            # save accepted file to uploads folder
            try:
                form.file.data.save('static/uploads/original.jpg')
                logging.info('File saved locally')
            except:
                logging.info('Failed to save file')

            logging.info('Image generation process beginning')

            # here trigger the loady wheel

            gan_generate()
            return render_template('results.html')

    return render_template('upload.html', form=form)


nz = 100
ngf = 64
ndf = 64
nl = 2
nc = 3

class Generator(torch.nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                in_channels=nz + nl,
                out_channels=ngf * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=ngf * 8
            ),
            nn.ReLU(
                inplace=True
            ),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(
                in_channels=ngf * 8,
                out_channels=ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=ngf * 4
            ),
            nn.ReLU(
                inplace=True
            ),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(
                in_channels=ngf * 4,
                out_channels=ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=ngf * 2
            ),
            nn.ReLU(
                inplace=True
            ),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(
                in_channels=ngf * 2,
                out_channels=ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=ngf
            ),
            nn.ReLU(
                inplace=True
            ),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(
                in_channels=ngf,
                out_channels=nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Initialize Models
PATH = './lung_generator.pth'
netG = Generator().to(device)
netG.load_state_dict(torch.load(PATH))

# only true if you run script directly, if imported will be false
if __name__ == '__main__':
    logging.basicConfig(filename='app.log', format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w', level=logging.INFO)
    logging.info('Started')
    app.run(debug=True)
    logging.info('Finished')
