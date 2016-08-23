from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import matplotlib.pyplot as plt

import numpy.random as rnd
import datetime as dt
import time
import os



def genDigitsImg(numbers,font,img_size=(64,32), colorBackground = "white",    colorText = "black"):
    '''
    Generates one image with random digits with specified font
    
    numbers - numpy array with digits
    img_size - tuple of img width and height
    
    font - PIL font object
    
    Returns
    ===========
    img - PIL img object
    
    '''
    digit_offset=5
    dh=-5 #height offset
    angle_var=25


    img = Image.new('RGBA', img_size, colorBackground)

    for i,nr in enumerate(numbers):
        
        digit_str = str(nr)
        fw, fh=font.getsize(digit_str)
        im1 = Image.new('RGBA',(fw,fh+15),colorBackground)
        
        d1  = ImageDraw.Draw(im1)
        d1.text( (0,dh),digit_str,font=font, fill=colorText)
        #d1.rectangle((0, 0, fw-1, fh-1), outline='red')
        
        im1sz = im1.size
        #d1.rectangle((0, 0, im1sz[0]-1, im1sz[1]-1), outline='green')
        
        angle = rnd.randint(-angle_var,angle_var)    
        #im1_rot=im1.rotate(angle,  expand=1)
        im1_rot=im1.rotate(angle, resample=Image.BILINEAR,  expand=1)
        #im1_rot=im1.rotate(angle, resample=Image.BICUBIC,  expand=1)
        
        pad_w = rnd.randint(-5,5)
        pad_h = rnd.randint(4)
        
        pos_w = digit_offset+pad_w
        
        #img.paste(im1_rot,(pos_w,pad_h))
        img.paste(im1_rot,(pos_w,pad_h),im1_rot)
        
        
        digit_offset=pos_w+im1_rot.size[0]
        
    return img
    

font_size = 26 
#font_size = 24


font_names = ["OpenSans-Regular.ttf", "Mothproof_Script.ttf", "Calligraffiti.ttf"]
font_names = ["OpenSans-Regular.ttf"]
font_path = "fonts/{}"
folder='shared/Digits_4'

#how many images with one type of font, final dataset has size number_of_images*number_of_fonts
number_of_images=4

#image size
img_size=(104,32)#width, height
#img_size=(32,28)#width, height

#how many digits to generate
random_digits=4
dispaly_count=1


#plt.figure(figsize=(20,10))
for font_name in font_names:

    
    font = ImageFont.truetype(font_path.format(font_name), font_size)
    font_folder = os.path.splitext(font_name)[0]
    
    img_save_folder = '{}/{}/'.format(folder,font_folder)
    
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)


    for a in range(number_of_images):
    
        numbers = rnd.choice(10,random_digits, replace=True)    
        numbers_str = ''.join([str(x) for x in numbers])
       
        img = genDigitsImg(numbers,font,img_size=img_size)
        
        digit_file = '{}{}_{}.png'.format(img_save_folder,numbers_str,int(time.time()*1000))
        
        #convert to grayscale
        img = img.convert('L')
        #img.save(digit_file)
        
        if a % dispaly_count ==0:
            #plt.axis('off')
            plt.imshow(img,cmap=plt.cm.gray, interpolation='bicubic')
            plt.show()
            
