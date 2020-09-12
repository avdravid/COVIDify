# Cisco Log Analysis & Remediation Kiosk

Hi, welcome to my Cisco software engineer intern repository for the Summer of 2020! This is the Cisco Log Analysis & Remediation Kiosk, a project I worked on for the Identity Services Engine (ISE) team during the last few weeks of my internship. This Flask application is designed to assist escalation engineers in troubleshooting common ISE configuration issues. 

The structure of this repo is as follows:

Current layout of project
- Kiosk – main project folder
    - compliance_check.py – parses /support/showtech/showtech.out and checks various parameters for compliance, output is recorded on dashboard.html
    - dashboard_generator.py – runs all other scripts and produces dashboard.html, converts Plotly figures produced to .html code and embeds them in the dashboard
    - elasticsearch_indexer.py – uses Elasticsearch API to provide methods for clearing indexes to prevent duplicate documents and bulk-indexing documents
    - interim_accounting.py – parses /support/logs/localStore/* files and collects parameters on all instances of Interim Accounting Watchdog Updates
    - sar_parser.py – parses system /support/adeos/sa/*.sar files for various types of data pertaining to system disk usage, memory, network, etc., stores in a dictionary of arrays and indexes data to Elasticsearch
- static
    - favicon.ico – Cisco icon
    - main.css – CSS formatting

- templates
    - dashboard.html – HTML file containing Plotly graphs, generated by dashboard_generator.py and rendered
    -	dashboard_generate.html – intermediate page with a button to prompt user to run dashboard_generator.py
    - dashboard_layout.html – HTML layout for dashboard
    - dashboard_placeholder.html – simple placeholder for when no dashboard.html file has been generated yet
    - decompress_layout.html – intermediate page with a button to prompt user to decompress the file
    - home.html – home page that extends layout.html
    - kibana.html – extends layout.html, contains embedded iframe of Kibana dashboard
    - layout.html – standardized layout, contains the top level option menu and color scheme
    - upload.html – form to upload file and enter GPG decryption password
- uploads – temporary storage space for uploaded files (automatically deleted during processing)
- .gitignore – standard .gitignore file for commits
- Dockerfile – Docker file configuration to run this app in a Docker container
- forms.py – outlines flask_wtf forms to be used 
- kiosk.py – main Flask application file
- README.md – this!
- requirements.txt – dependencies required for kiosk.py to run
- unit_test.py – a file to unit test new features added to the scripts, feel free to delete
