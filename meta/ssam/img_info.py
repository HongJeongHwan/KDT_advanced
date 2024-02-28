'''
명령창(cmd)에서 아래 명령을 실행하여 image, pillow 라이브러리를 설치하세요
pip install image
pip install pillow
'''
from PIL import Image
from PIL.ExifTags import TAGS
#
def get_image_exif(image):
    img = Image.open(image)

    img_info = img._getexif()
    if img_info is not None:
        print(image)
    #print(img_info)
    for tag_id in img_info:
        tag = TAGS.get(tag_id, tag_id)
        data = img_info.get(tag_id)
        print(f'{tag} : {data}')
       # if tag == 'DateTime' or tag == 'DateTimeOriginal':
       #     print(f'{tag} : {data}')
    img.close()

image = './mouse/10.png'

#print(type(image)) 
get_image_exif(image)