import numpy as np
import os
from scipy.misc import imread


def load_dataset(digits_dir,img_size, digits_count, max_files=float('inf')):
    '''
    loads digit dataset, folder structure:
    Digit_2/
        Font_name1/
            00_timestamp.png
            ...
            99_timestamp.png
        Font_name2/
            00_timestamp.png
            ...
            99_timestamp.png
        Font_name3/
            00_timestamp.png
            ...
            99_timestamp.png
    
    Params
    ============
    digits_dir - root dir, containning the font folders with digits
    img_size - tuple, size of each image
    digits_count - how many digits is on the image
    
    Return
    ============
    x - numpy array with data, Nx(img_size[0]*img_size[1]), images are flatten
    Y - numpy array with labels,
    
    '''
    img_files = []
    for root, dirs, files in os.walk(digits_dir):
     for file in files:
        img_files.append(os.path.join(root, file))
    
    N= len(img_files)
    
    imgN= img_size[0]*img_size[1]
    X = np.zeros([N,imgN])
    # how many digits?
    Y = np.zeros([N,digits_count*10])
    
    
    for i,file in enumerate(img_files):
        img = imread(file)
        
        #take number from file name
        number_str = os.path.basename(file).split('_')[0]
        
        X[i,:] = img.flatten()
        Y[i,:] = encode2vector(number_str)
    
    
    return (X,Y)
    
    
def encode2vector(number_str):
    '''
    Encode number(string) in to one hot encoding, each
    
    Params
    =========
    number_str - string containning number with N digits
    
    '''
    
    N = len(number_str)
    
    vec = np.zeros(N*10)
    
    for i,char in enumerate(number_str):
        
        digit = int(char)
        
        pos= i*10+digit
        
        vec[pos]=1
        
        
    return vec
    
    