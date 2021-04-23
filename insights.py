import constants
from random import randrange

#Insights on Image sizes
def image_size(data_dir):
    

    return dataframe

# zero normalization 

# z-score 

#Image Analysis 

#Append all images to img_list and labels to 
img_list, img_label = [], []

#max_val is how many pictures from each category do you want to load for example max_val = 1100 means 2200 total images
max_val=1100
img_size=256

folder = []
for val in os.listdir(constants.TRAIN_DATA):
    path = os.path.join(constants.TRAIN_DATA, val)
    if os.path.isdir(path):
        folder.append(val)
        
print('Labels found in the Dataset: ',folder)

j=0
for folder in folder:
    for img_file in os.listdir(os.path.join(constants.TRAIN_DATA,folder)):
        img_path = os.path.join(constants.TRAIN_DATA,folder)
        image= cv2.imread(os.path.join(img_path,img_file)) 
        if imageis not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
            image= cv2.resize(img,(img_size, img_size))
            image= cv2.GaussianBlur(img,(5,5),0)   
            img_list.append(image)
            if folder == 'NORMAL':
                img_label.append(0)
                #print('normal')
            else:
                img_label.append(1)
                #print('PNE')
        j=j+1
        if j >= max_val:
            j=0
            break
            
img_list,img_label=np.array(img_list),np.array(img_label)
print(img_list.shape)

#Print a random image
i = randrange(max_val*2)

#Original Image
plt.imshow(img_list[i],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

#Histogram
plt.hist(img_list[i].ravel(),256,[0,256])
plt.show()

#Laplacian
laplacian = cv2.Laplacian(img_list[i],cv2.CV_8UC1)
plt.imshow(laplacian,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

#Canny
canny = cv2.Canny(img_list[i],40,200)
plt.imshow(canny,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

#SobelX
SobelX = cv2.Sobel(img_list[i],cv2.CV_8UC1,1,0,ksize=5)
plt.imshow(SobelX,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

#SobelY
sobelY = cv2.Sobel(img_list[i],cv2.CV_8UC1,0,1,ksize=5)
plt.imshow(sobelY,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

#SobelXY
sobelXY = cv2.Sobel(img_list[i],cv2.CV_8UC1,1,1,ksize=5)
plt.imshow(sobelXY,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

#thresholding
ret, th1 = cv2.threshold(img_list[i],100,255,cv2.THRESH_TOZERO)
plt.imshow(th1,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

blurr = cv2.GaussianBlur(img_list[i],(5,5),0)
ret, th2 = cv2.threshold(img_list[i],120,255,cv2.THRESH_BINARY)
plt.imshow(th2,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

#Sharpening
kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
sharpened = cv2.filter2D(img_list[i], -1, kernel_sharpening)
plt.imshow(sharpened,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()