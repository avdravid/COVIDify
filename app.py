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
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np

nz = 100
ngf = 64
ndf = 64
nl = 2
nc = 3

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')

app = Flask(__name__)

app.config['SECRET_KEY'] = '6e8af7ecab9c7e41e861fa359a89f385'

# support bundles should be .tar.gpg, process will fail and exit otherwise
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
# Initialize Models
PATH = './lung_generator.pth'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS


def gan_generate():
    # generate new image from upload here
    # the uploaded image will be in static/uploads/original.jpg
    make_stuff(transform_image())
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

    def forward(self, inputs, condition):
        # Concatenate Noise and Condition
        cat_inputs = torch.cat(
            (inputs, condition),
            dim=1
        )

        # Reshape the latent vector into a feature map.
        cat_inputs = cat_inputs.unsqueeze(2).unsqueeze(3)

        return self.main(cat_inputs)


netG = Generator().to(device)
netG.load_state_dict(torch.load(PATH, map_location=device))


class SquashTransform:
    def __call__(self, inputs):
        return 2 * inputs - 1


def transform_image():
    img = Image.open('./static/uploads/original.jpg')
    data_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        SquashTransform()
    ])
    return data_transform(img).unsqueeze(0)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(
                224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(
                224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


def make_stuff(image):
    init_noise = Variable(torch.randn(
        1, nz
    ).to(device), requires_grad=True)

    negative_label = torch.Tensor([[1, 0]]).to(device)
    positive_label = torch.Tensor([[0, 1]]).to(device)

    optim = torch.optim.Adam([init_noise], lr=0.1, betas=(0.5, 0.999))
    original_image = image[0].to(device)
    mask = torch.ones([1, 3, 64, 64]).to(device)
    mask[0, :, 4:60, 20:60] = 2

    for epoch in range(0, 20):
        print('Epoch: ' + str(epoch))
        original_image = image[0].to(device)
        optim.zero_grad()
        sample = netG(init_noise, negative_label).to(device)
        sample = sample.reshape([1, 3, 64, 64])
        original_image = original_image.reshape([1, 3, 64, 64])
        loss_func = VGGPerceptualLoss().to(device)
        loss = loss_func(sample, original_image) + 10 * \
            torch.mean(mask*(original_image - sample)**2)
        #loss = 100* torch.mean(mask*(original_image - sample)**2)
        loss.backward()
        optim.step()

        if (epoch+1) % 10 == 0:
            reconstructed_image = netG(
                init_noise, negative_label
            ).detach().cpu().view(-1, 3, 64, 64)

            reconstructed_image = reconstructed_image[0, ]

            original_image = original_image.cpu().view(3, 64, 64)
            original_image = np.transpose(original_image, (1, 2, 0))
            original_image = (original_image + 1)/2

            reconstructed_image = np.transpose(reconstructed_image, (1, 2, 0))
            reconstructed_image = (reconstructed_image + 1)/2

    original_image = image[0].to(device)
    reconstructed_image_positive = netG(
        init_noise, positive_label
    ).detach().cpu().view(-1, 3, 64, 64)

    reconstructed_image_positive = reconstructed_image_positive[0, ]

    reconstructed_image_negative = netG(
        init_noise, negative_label
    ).detach().cpu().view(-1, 3, 64, 64)

    reconstructed_image_negative = reconstructed_image_negative[0, ]

    original_image = original_image.cpu().view(3, 64, 64)
    original_image = np.transpose(original_image, (1, 2, 0))
    original_image = (original_image + 1)/2

    reconstructed_image_negative = np.transpose(
        reconstructed_image_negative, (1, 2, 0))
    reconstructed_image_negative = (reconstructed_image_negative + 1)/2

    reconstructed_image_positive = (reconstructed_image_positive + 1)/2
    save_image(reconstructed_image_positive, './static/uploads/covidified.jpg')


# only true if you run script directly, if imported will be false
if __name__ == '__main__':
    logging.basicConfig(filename='app.log', format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w', level=logging.INFO)
    logging.info('Started')
    app.run(debug=True)
    logging.info('Finished')
