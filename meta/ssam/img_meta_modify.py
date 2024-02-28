from PIL import Image
from PIL.ExifTags import TAGS


def change_exif_date(image_path, new_date):

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
            exif_data['DateTimeOriginal'] = new_date
            exif_data['DateTime'] = new_date

            print(myTags)
            # Encode EXIF data
            exif_bytes = Image.Exif()
            #print(exif_bytes)
            #print(exif_data)
            for tag, value in exif_data.items():
                #tag_id = TAGS.get(tag)  리턴이 None임. 왕 짜증
                tag_id = myTags.get(tag)  ##만든 dictionary에서 받아 사용
                exif_bytes[tag_id] = value
                #print(tag_id)


            print(exif_bytes)
            # Insert modified EXIF data back into the image
            image.save(image_path, exif=exif_bytes)
            print("Exif date updated successfully.")
        else:
            print("No EXIF data found in the image.")
    except Exception as e:
        print(f"An error occurred: {e}")






tag = 'DateTimeOriginal'
tag_id = TAGS.get(tag)
print(f"Tag: {tag}, Tag ID: {tag_id}")

# Example usage:
image_path = "fff/그네1.jpg"
new_date = "2024:02:19 12:12:12"   # New date in the format YYYY:MM:DD HH:MM:SS
#new_date = "2024:12:32 10:10:10"   # New date in the format YYYY:MM:DD HH:MM:SS
change_exif_date(image_path, new_date)