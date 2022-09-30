import numpy as np
import nibabel as nib
from dipy.segment.tissue import TissueClassifierHMRF
import matplotlib.pyplot as plt
from skimage import exposure as ex
from scipy.spatial import distance
from skimage import morphology
from scipy.ndimage import binary_dilation 
from PIL import  ImageFilter,Image
import time




def load_data(dir):
    nii = nib.load(dir)
    neuro_data = np.array(nii.get_fdata(), dtype=np.uint8)
    return neuro_data

def filter_and_enhancement(img):
    

    arr_eq = ex.equalize_adapthist(img)
    slice_enhanced = ex.rescale_intensity(arr_eq, out_range=(0, 255))
    arr_ = np.uint8(slice_enhanced) 
    
    return arr_


def gray_and_white_segment(data,nb_classes,beta):
    hmrf = TissueClassifierHMRF()
    initial_segmentation, final_segmentation, PVE = hmrf.classify(data, nb_classes, beta)
    return initial_segmentation, final_segmentation, PVE



def white_gray_matter(img,treshold_1,treshold_2):
 
    white_matter = img > treshold_1
    gray_matter = np.zeros((256,256))

    for i in range(256):
        for j in range(256):
            if (img[i,j] <= treshold_1) and (img[i,j] > treshold_2):
                val = True
            else:
                val = False
            gray_matter[i,j] = val
    return white_matter , gray_matter
    #gray_matter = np.logical_and(img > treshold_gray, img <= treshold_white)

def noise_removal(arr_1,arr_2):
    w = morphology.area_opening(arr_1,130, connectivity=5) 
    g = morphology.area_opening(arr_2,130, connectivity=5) 

    white = binary_dilation(w, morphology.disk(4))
    gray_ = np.logical_and(white, g) 
    
    gray = np.zeros(white.shape)
    for row in range(256):
        for col in range(256):
            if (white[row,col]) and (g[row,col]):
                gray[row,col] = True
                
            else:
                gray[row,col] = False     
    return w, gray

def euclidean_distance(x_1,x_2,y_1,y_2):
    dist = np.sqrt(np.square(x_1-x_2) + np.square(y_1-y_2)) 
    return dist


fig, ax = plt.subplots(1, 2) 
ax = ax.flatten()    
figures = []
for i in range(2):
    im = ax[i].imshow((np.zeros((256,256))), cmap='gray')
    figures.append(im)
    ax[i].axis('off')



neuro_data = load_data('raw_t1_subject_02.nii')
 
width, height, count = neuro_data.shape

data_arr = np.zeros((count,width,height))

for index in  range(count):    
    data_arr[index] = neuro_data[:,:,index]
res_arr = np.zeros((256, 256, 256), dtype=np.float32)
for index in range(256):
 
    im =  neuro_data[:,:,index]
     
    
    #plt.imshow(image,cmap = 'gray')
    #plt.show()
    image = filter_and_enhancement(im)
    white, gray = white_gray_matter(image,170,100)
    white_matter, gray_matter = noise_removal(white,gray)

    x_white, y_white = np.where(white_matter==True)
    x_gray, y_gray = np.where(gray_matter==True)


    res_shape = (256,256)
    dtype_ = np.float32
    result = np.zeros(res_shape, dtype=dtype_)

    for cord in range(x_gray.shape[0]):
        eclu_dist = np.sqrt(
                            np.square(x_white - x_gray[cord]) + 
                            np.square(y_white - y_gray[cord]))
        distance_ = euclidean_distance(x_white,x_gray[cord],y_white,y_gray[cord])
        shortest_distance = np.min(distance_)
        result[x_gray[cord], y_gray[cord]] = shortest_distance
    res_arr[:, :, index] = np.float32(result)
    cortical_thickness_map = Image.fromarray(np.uint8(result)).filter(ImageFilter.MaxFilter(size = 1))
        

    figures[0].set_data(im)
    figures[0].autoscale()
    ax[0].set_title(f'original sample {index}', fontsize=10)

    figures[1].set_data(cortical_thickness_map)
    figures[1].autoscale()
    ax[1].set_title(f'cortical_thickness_map sample {index}', fontsize=10)
    plt.pause(0.0001)

res_nii = nib.Nifti1Image(res_arr, affine=None)
nib.save(res_nii, 'thickness_map_subject_02.nii') #saving as nii image format


        
        
        