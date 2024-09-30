from PIL import Image
img =Image.open("abc.JPEG")
img.show()
img.rotate(180).show()
img.save("temp/img2.jpg")