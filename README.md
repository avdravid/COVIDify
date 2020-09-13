# COVID-19 Lung GAN Generator

Hi, welcome to our entry for PenApps XXI!

The structure of this repo is as follows:

Current layout of project
- app.py – main project folder
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
- forms.py – outlines flask_wtf forms to be used 
- app.py – main Flask application file
- README.md – this!
