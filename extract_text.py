from PIL import Image
import pytesseract

img = Image.open('out.jpg')
print (pytesseract.image_to_string(img))
