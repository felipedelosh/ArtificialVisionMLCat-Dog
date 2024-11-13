"""
FelipedelosH

Read all images .png y .jpg in folder and delete img with errors

python -m pip install pillow
"""
import os
from PIL import Image, UnidentifiedImageError

_path = "validation\\dogs"
_errors = 0

def getAllFilesInFolderPath(path):
    data = []

    try:
        _files = os.listdir(path)
        for file in _files:
            _complete_path = os.path.join(path, file)

            if os.path.isfile(_complete_path):
                data.append(file)
    except:
        pass

    return data


def validateImage(path):
    global _errors
    try:
        img = Image.open(path)
        img.verify()
    except:
        _errors = _errors + 1
        try:
            os.remove(path)
        except:
            pass

_img_paths = getAllFilesInFolderPath(_path)

for i in _img_paths:
    validateImage(f"{_path}\\{i}")

print("==========VALIDATION===========")
print(f"Total images: {len(_img_paths)}")
print(f"Total ERRORS images: {_errors}")

