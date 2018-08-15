# coding: utf-8
import requests
import os
import config
import rarfile

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def unrar(dpath, xpath):
    size = 0
    n = 0
    with rarfile.RarFile(dpath) as opened_rar:
        for f in opened_rar.infolist():
            size +=f.file_size
            n+=1
        opened_rar.extractall(xpath)
        print("extracted {} files, with total size of {}".format(n,size))
