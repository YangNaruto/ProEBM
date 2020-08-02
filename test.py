from PIL import Image

img = Image.open("one-piece-anime-hd-wallpaper-1920x1080-43893.jpg")

img = img.resize((512, 512), Image.LANCZOS)
img.save('lanczos.png')
img.show()