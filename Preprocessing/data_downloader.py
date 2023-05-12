import os
import shutil
import urllib
import zipfile
import glob
import urllib.request

def download_dataset(name):
    os.chdir('./data')
    """Download the dataset into current working directory.
    The labeled dataset is ESC-50, the unlabeld are ESC-US-00,ESC-US-01, ... , ESC-US-25 
    but I'm not able to download them automatically from https://dataverse.harvard.edu/dataverse/karol-piczak?q=&types=files&sort=dateSort&order=desc&page=1"""

    if name=='ESC-50' and not os.path.exists(f'./{name}'):

        if not os.path.exists(f'./{name}-master.zip') and not os.path.exists(f'./{name}-master'):
            urllib.request.urlretrieve(f'https://github.com/karoldvl/{name}/archive/master.zip', f'{name}-master.zip')

        if not os.path.exists(f'./{name}-master'):
            with zipfile.ZipFile(f'{name}-master.zip','r') as package:
                package.extractall(f'{name}-master')

        os.remove(f'{name}-master.zip') 
        original = f'./{name}-master/{name}-master/audio'
        target = f'./{name}'
        shutil.move(original,target)
        original = f'./{name}-master/{name}-master/meta'
        target = f'./meta'
        shutil.move(original,target)

    if os.path.exists(f'./{name}-master'):
        shutil.rmtree(f'./{name}-master')

    else:
        print('donwload it from https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/YDEPUT/YE0CVE&version=2.0#')
        pass 
    os.chdir('../')

def main():
    download_dataset('ESC-50')
if __name__='__main__':
    main() # Call main() if this module is run, but not when imported.
            