from PIL import Image
from PIL.ExifTags import TAGS
import os

image_path = []
test_dir = './mouse'

for name in sorted(os.listdir(test_dir)):
    if name.find('.') > 0:
        image_path.append(test_dir + '/' + name)
        # print(test_dir + '/' + name)
        # print(f'평가용 이미지 경로 확인 :\n{image_path}')

# 평가용 이미지를 이용한 평가 진행
for image in image_path:
    img = Image.open(image)
    img_info = img._getexif()

    if img_info is not None:
        print(image)
    else:
        print('=======================')