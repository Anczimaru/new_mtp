# coding: utf-8
import os
import config


x = input("Do you want to install all neccessary elements?[y/n]")
if x =="y":
    os.system("pip3 install tensorflow")
    os.system("pip3 install Pillow")
    os.system("pip3 install imageio")
    os.system("pip3 install rarfile")
    os.system('pip3 install opencv-python')
    os.system('pip3 install matplotlib')


print("Creating directories")
data_dir = config.DATA_DIR
if not os.path.exists(data_dir): #if dir not present, make it
    os.mkdir(data_dir)


#Download dataset from google drive and unpack it
x = input("Do you want to download pictures?[y/n]")
if x =="y":
    import install_helper
    if not os.path.exists(config.BG_INSTALL_DIR):
        print("Downloading dataset")
        file_id = '1FbDCH8nvaGMoRLYxW1Rqn5-gmE45K1q8'
        rar_dest = 'data/images.rar'
        install_helper.download_file_from_google_drive(file_id, rar_dest)
        file_id = "1fsc1GgkgxQn57v0ej5a4AEea1BzgLh_S"
        install_helper.download_file_from_google_drive(file_id,
                                                        os.path.join(config.DATA_DIR,
                                                                    "injection.png"))

    x1 = input("Do you want to upack archive?[y/n]")
    if x1 =="y":
        import shutil
        if os.path.exists(config.BG_INSTALL_DIR):
            shutil.rmtree(config.BG_INSTALL_DIR, ignore_errors=True)
        rar_dest = 'data/images.rar'
        install_helper.unrar(rar_dest, config.BG_INSTALL_DIR)
print("Installation compleated")
