# COVIDify

Hi, welcome to our entry for PenApps XXI! Our team for this year’s Pennapps decided to focus on one of the numerous problems presented by COVID-19, namely the identification of particular physiological changes caused by new diseases, both short-term and long-term.  To this end, we developed an application that uses a generative adversarial network (GAN) to overlay COVID-19 features onto X-ray images of healthy lungs. Using the GAN-generated lung X-rays and comparing them to pictures of healthy lungs allowed us to isolate particular COVID-19 feature “layers” that the GAN had recognized. We believe that, by isolating these additional features, we may potentially identify specific physiological changes both during the disease and post-recovery, a space which has been cause for concern but not robustly studied. We also hope to extend this platform to novel diseases as they appear in order to more quickly assess their long-term effects on internal organ tissues.

The structure of this repo is as follows:

Current layout of project
- static
    - virus.png – icon
    - main.css – CSS formatting
    - uploads – temporary storage space for uploaded files
- templates
    - home.html – home page that extends layout.html
    - layout.html – standardized layout, contains the top level option menu and color scheme
    - upload.html – form to upload image
    - results.html - presents a side-by-side view of uploaded and generated x-ray images
    - info.html - general information about this web app and additional redirect links about COVID-19
- .gitignore – standard .gitignore file for commits
- lung_GAN_Model.ipynb - GAN model training and testing in Jupyter Notebook
- forms.py – outlines flask_wtf forms to be used 
- app.py – main Flask project folder
- lung_generator.pth - exported GAN model
- README.md – this!
- requirements.txt - Python dependencies and packages
