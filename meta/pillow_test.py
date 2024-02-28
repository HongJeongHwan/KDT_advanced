# 이미지 파일 경로
image_path = 'test.jpg'

# =============================================================
# 1. pillow를 이용한 예제 ======================================
# =============================================================
# from PIL import Image

# image = Image.open(image_path)

# # 이미지 제작자 정보 갱신
# image.info["author"] = "John Smith"

# # 저작권 정보 갱신
# image.info["copyright"] = "Copyright (c) 2021 My Company"

# # # 수정된 메타데이터 저장
# image.save("test_updated.jpg")


# =============================================================
# 2. imageio를 이용한 예제 =====================================
# =============================================================
# import imageio
import imageio.v2 as imageio

# 이미지 파일의 메타데이터 읽기
meta_data = imageio.imread(image_path).meta

# 메타데이터 출력
print(meta_data)

# 특정 메타데이터 접근
height = meta_data.get('height')
width = meta_data.get('width')
colorspace = meta_data.get('colorspace')

# 메타데이터 수정
meta_data['width'] = 1000
meta_data['height'] = 800

# 수정된 메타데이터 적용하여 이미지 파일 저장
new_image_path = 'new_image.jpg'
imageio.imwrite(new_image_path, imageio.imread(image_path), meta=meta_data)


# =============================================================
# PIL을 이용한 예제    =========================================
# =============================================================
# image = Image.open(image_path)
# # extract EXIF data
# metadata = image._getexif()
# print(metadata)
# print(image.height, image.width)

# from PIL import Image
# from PIL.ExifTags import TAGS

# def get_image_exif(image):
#     img = Image.open(image)

#     img_info = img._getexif()
#     for tag_id in img_info:
#         tag = TAGS.get(tag_id, tag_id)
#         data = img_info.get(tag_id)
#         print(f'{tag} : {data}')
#     img.close()

# # image ='20150404_095727.jpg'
# print(get_image_exif(image_path))

# =============================================================
# Exifread를 이용한 예제 ======================================= 에러남
# =============================================================
# import exifread

# with open(image_path, 'rb') as f:
#     tags = exifread.process_file(f)
#     print(tags)
# print(tags['Image Model'])


# =============================================================
# Exif를 이용한 예제     =======================================
# =============================================================
# from exif import Image

# with open(image_path, 'rb') as f:
#     img = Image(f)
#     # print(img)
# print(img.has_exif)
# print(img.gps_altitude_ref, img.model, img.datetime)


