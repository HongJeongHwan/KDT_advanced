'''
명령창(cmd)에서 아래 명령을 실행하여 image, pillow 라이브러리를 설치하세요
pip install image
pip install pillow
'''
from PIL import Image
from PIL.ExifTags import TAGS

# 이미지의 meta 정보를 확인
def get_image_exif(image):
    img = Image.open(image)
    # print(f'img = {img}')
    img_info = img._getexif()
    # print(f'img_info = {img_info}')
    for tag_id in img_info:
        tag = TAGS.get(tag_id, tag_id)
        data = img_info.get(tag_id)
        # print(f'{tag} : {data}')
        if tag == 'Artist':
            print(f'분류결과 : {data}')
        elif tag == 'XResolution':
            print(f'분류확률 : {data}')
    img.close()

# 이미지의 meta 정보를 수정
def change_exif_date(image_path, classified_result, classified_percent):

    myTags = {}  ### TAGS.get() 이 자꾸 에러라서 만들어서 사용
    try:
        image = Image.open(image_path)
        exif = image._getexif()
        exif_data = {}

        # Extract EXIF data
        if exif:
            for tag, value in exif.items():
                decoded = TAGS.get(tag, tag)
                #print(tag)
                if decoded != 'ExifOffset':  #에러부분 제외
                    exif_data[decoded] = value
                    myTags[decoded] = tag  ## tag 에 해당하는 tag_id 설정

            # Update DateTimeOriginal and DateTime tags
            exif_data['Artist'] = classified_result
            exif_data['XResolution'] = classified_percent
            
            # print(f'type(myTags) =\n{myTags}')

            # Encode EXIF data
            exif_bytes = Image.Exif()
            #print(exif_bytes)
            #print(exif_data)
            for tag, value in exif_data.items():
                #tag_id = TAGS.get(tag)  리턴이 None임. 왕 짜증
                tag_id = myTags.get(tag)  ##만든 dictionary에서 받아 사용
                exif_bytes[tag_id] = value
                #print(tag_id)

            # print(f'exif_bytes =\n{exif_bytes}')
            # Insert modified EXIF data back into the image
            image.save(image_path, exif=exif_bytes)
            print("Exif date updated successfully.")
        else:
            print("No EXIF data found in the image.")
    except Exception as e:
        print(f"An error occurred: {e}")


# meta 정보를 확인할 이미지 지정
image = './mouse/1.png'

# meta 정보 수정전 이미지의 meta 정보 확인
print(f'{image} 파일의 meta정보를 출력을 시작합니다.')
print('-' * 50)
get_image_exif(image)
print('-' * 50)
print(f'{image} 파일의 meta정보를 출력을 종료합니다.\n')


# # tag = 'DateTimeOriginal'
# # tag_id = TAGS.get(tag)
# # print(f"Tag: {tag}, Tag ID: {tag_id}")

# # new date에 원하는 날짜로 변경해 보세요.
classified_result = 'bodan'
classified_percent = 2.5
# new_date = "2024:02:22 14:51:00"   # New date in the format YYYY:MM:DD HH:MM:SS
change_exif_date(image, classified_result, classified_percent)


# meta 정보 수정후 이미지의 meta 정보 확인
print(f'{image} 파일의 meta정보를 출력을 시작합니다.')
print('-' * 50)
get_image_exif(image)
print('-' * 50)
print(f'{image} 파일의 meta정보를 출력을 종료합니다.\n')