# Cisco Log Analysis & Remediation Kiosk

Hi, welcome to my Cisco software engineer intern repository for the Summer of 2020! This is the Cisco Log Analysis & Remediation Kiosk, a project I worked on for the Identity Services Engine (ISE) team during the last few weeks of my internship. This Flask application is designed to assist escalation engineers in troubleshooting common ISE configuration issues. 

The structure of this repo is as follows:

Current layout of project
- static
    - favicon.ico – icon
    - main.css – CSS formatting

- templates
    - home.html – home page that extends layout.html
    - layout.html – standardized layout, contains the top level option menu and color scheme
    - upload.html – form to upload image
- uploads – temporary storage space for uploaded files
- .gitignore – standard .gitignore file for commits
- forms.py – outlines flask_wtf forms to be used 
- app.py – main Flask application file
- README.md – this!