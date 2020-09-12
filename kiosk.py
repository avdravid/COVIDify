# Main Flask application file

from flask import Flask, render_template, url_for, flash, redirect, request, g, send_from_directory
from forms import RegistrationForm, LoginForm, UploadForm, ButtonForm
from werkzeug.utils import secure_filename
import os
import shutil
from os import path
import gnupg
from dashboard_generator import generate_dashboard
import logging

app = Flask(__name__)

app.config['SECRET_KEY'] = '6e8af7ecab9c7e41e861fa359a89f385'

# support bundles should be .tar.gpg, process will fail and exit otherwise
ALLOWED_EXTENSIONS = {'gpg'}

# check if file has correct extension
try:
    logging.info("Trying to remove existing support bundle if still present")
    shutil.rmtree("support")

except:
    logging.info("No existing support bundle")
    pass

try:

    uploads = os.listdir('uploads')
    for upload in uploads:
        logging.info('Removing file: ' + upload + 'in uploads')
        os.remove(os.path.join('uploads', upload))
    os.remove('decrypted.tar')
except:
    pass


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS

# get Cisco favicon


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# render home landing page


@app.route("/")
def home():
    return render_template("home.html")

# path for upload page to upload file and enter gpg decryption key


@app.route("/upload", methods=['GET', 'POST'])
def upload():
    logging.info('Uploading file')
    form = UploadForm()

    if form.validate_on_submit():
        logging.info('File validated')

        filename = secure_filename(form.file.data.filename)
        if not allowed_file(filename):
            logging.info('Illegal file extension')

            flash('File must have extension .tar.gpg', 'danger')
        else:
            flash('File Accepted', 'success')

            # save accepted file to uploads folder
            try:
                form.file.data.save('uploads/' + filename)
                logging.info('File saved locally')
            except:
                logging.info('Failed to save file')
            # decrypt gpg to decrypted.tar and remove original gpg
            cwd = os.getcwd()
            gpg = gnupg.GPG(gnupghome=cwd+'/uploads/')

            with open(cwd+'/uploads/'+filename, 'rb') as f:
                logging.info('GPG decryption process beginning')
                try:
                    status = gpg.decrypt_file(
                        file=f, passphrase=form.password.data, output='decrypted.tar')
                    logging.info('GPG decryption successful')
                except:
                    logging.error('GPG decryption failed')
                if not status.ok:
                    os.remove('uploads/'+filename)
                    flash('GPG Decryption Failed', 'danger')
                    flash("Error Message: " +
                          status.stderr, 'danger')
                    return redirect(url_for('upload'))
                else:
                    os.remove('uploads/'+filename)
                    # redirect to next decompression step
                    return redirect(url_for('decompress'))

    return render_template('upload.html', form=form)

# page with button to decompress the already decrypted .tar file


# only true if you run script directly, if imported will be false
if __name__ == '__main__':
    logging.basicConfig(filename='kiosk.log', format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w', level=logging.INFO)
    logging.info('Started')
    app.run(debug=True)
    logging.info('Finished')
