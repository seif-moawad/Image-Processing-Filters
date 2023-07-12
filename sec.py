from tkinter import * 
from PIL import ImageTk,Image,ImageFilter
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import filters, data
from skimage.color import rgb2gray
import math 

root = Tk()
root.title("Home")
root.geometry("1500x1200")
root.configure(bg="#344D67")



def Spatial():
    my_w = Toplevel()
    my_w.title("Spatial Domain")
    my_w.geometry("1500x1200")
    my_w.configure(bg="#344D67")
    global iop
    iop = []
    f_types = [('All Files', '*.*'),('png Files', '*.png'),('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = ImageTk.PhotoImage(file=filename)
    b2 =Button(my_w,image=img) # using Button 
    b2.place(x= 250, y= 300)
    iop = filename
    #messagebox.askquestion("hdhdh", iop)
    
   

    #smoothing
    label1 = Label(my_w, text= "Smoothing Filters", font=("Arial",18), foreground= "white", border= 10, bg="#344D67").place(x= 150, y=5)

    m_btn = Button(my_w, text="Median Filter", bg="#6ECCAF", command = lambda:Median()).place(x=200, y= 70)
    a_btn = Button(my_w, text="Adaptive Filter", bg="#6ECCAF", command= lambda:Adaptive()).place(x=200, y= 120)
    aa_btn = Button(my_w, text="Averaging Filter", bg="#6ECCAF", command = lambda:Averaging()).place(x=200, y= 180)
    g_btn = Button(my_w, text="Gussian Filter", bg="#6ECCAF", command= lambda:Gussian_f()).place(x=200, y= 240)

    #sharpiniing
    label2 = Label(my_w, text= "Sharpining Filters", font=("Arial",18),foreground= "white", bg= "#344D67", border= 10).place(x=400, y= 5)

    l_btn = Button(my_w, text="Laplacian Operator",bg="#6ECCAF", command= lambda:Laplacian()).place(x=450, y= 60)
    u_btn =Button(my_w, text="Highboost Filtering",bg="#6ECCAF", command= lambda:Unsharp()).place(x=450, y= 120)
    r_btn =Button(my_w, text="Roberts Cross-Gradient Operators",bg="#6ECCAF", command= lambda:Robert()).place(x=450, y= 180)
    s_btn = Button(my_w, text="Sobel Operators",bg="#6ECCAF", command= lambda:sobel()).place(x=450, y= 240)

    #noise
    label3 = Label(my_w, text= "Noise Filters",font=("Arial",18), foreground= "white", bg= "#344D67", border= 10).place(x=650, y= 5)

    i_btn = Button(my_w, text="Impulse noise",bg="#6ECCAF", command= lambda:Impulse()).place(x=700, y= 60)
    g_btn = Button(my_w, text="Gaussian noise",bg="#6ECCAF", command= lambda:Gussian_n()).place(x=700, y= 120)
    uu_btn = Button(my_w, text="Uniform noise",bg="#6ECCAF").place(x=700, y= 180)

    
    
    global s
    s = []
    e = Entry(my_w)
    s = e
    e.grid(row=1, column= 0)
    
    my_w.mainloop()

def Frequency():
    my_ww =Toplevel()
    my_ww.title("Frequency Domain")
    my_ww.geometry("1500x1200")
    my_ww.configure(bg="#344D67")
    global ioc
    ioc = []
    f_types = [('All Files', '*.*'),('png Files', '*.png'),('Jpg Files', '*.jpg')]
    filename1 = filedialog.askopenfilename(filetypes=f_types)
    imgg = ImageTk.PhotoImage(file=filename1)
    ioc = filename1
    b3 =Button(my_ww,image=imgg) # using Button 
    b3.place(x= 250, y= 300)
    #messagebox.showinfo(imgg)

    label5 = Label(my_ww, text= "Frequency Domain", font=("Arial" , 18 ),foreground= "white", bg= "#344D67").place(x= 50, y=60)
    h_btn = Button(my_ww,text="Histogram Equalization",bg="#6ECCAF", command= lambda:Histogram_e()).place(x=400, y= 70)
    hh_btn = Button(my_ww,text="Histogram Specification",bg="#6ECCAF", command= lambda:Histogram_s()).place(x=650, y= 70)
    f_btn = Button(my_ww, text="Fourier transform",bg="#6ECCAF").place(x=900, y= 70)
    ii_btn = Button(my_ww, text="Interpolation",bg="#6ECCAF", command= lambda:Interpolation()).place(x=1150, y= 70)
    
    my_ww.mainloop()

def Other():
    my_www = Toplevel()
    my_www.title("Other Filters")
    my_www.geometry("1500x1200")
    my_www.configure(bg="#344D67")
    
    global iod
    iod = []
    f_types = [('All Files', '*.*'),('png Files', '*.png'),('Jpg Files', '*.jpg')]
    filename3 = filedialog.askopenfilename(filetypes=f_types)
    imgs = ImageTk.PhotoImage(file=filename3)
    b5 =Button(my_www,image=imgs) # using Button 
    b5.place(x= 250, y= 300)
    iod = filename3


    h_btn = Button(my_www,text="Image Negative",bg="#6ECCAF", command= lambda:Negative()).place(x=400, y= 70)
    o_btn = Button(my_www,text="Log Transformation",bg="#6ECCAF", command= lambda:LogTrans()).place(x=600, y= 70)
    t_btn = Button(my_www,text="Gamma Transformation",bg="#6ECCAF", command= lambda:Gamma()).place(x=800, y= 70)

    my_www.mainloop()

def Averaging():
    messagebox.askokcancel("good")
    global img_new
    img = cv2.imread(iop,0)
    # Obtain number of rows and columns
    # of the image
    #cv2.imshow("image",img)
    x, y = img.shape
    # Develop Averaging filter(3, 3) mask
    mask = np.ones([3, 3], dtype = int)
    mask = mask / 9

    # Convolve the 3X3 mask over the image
    img_new = np.zeros([x, y])

    for i in range(1, x-1):
        for j in range(1, y-1):
            temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]
            img_new[i, j]= temp       
    img_new = img_new.astype(np.uint8)

    ios = img_new
    cv2.imshow("image",img)
    cv2.imshow("img",img_new)
    cv2.waitKey(0)
    
    #lol = Button(mo,image= img_new).place(x=500, y = 500)

def Median():
    #messagebox.askokcancel("wow")
        ###########################################  MEDIAN   ##########################################################
    # Obtain the number of rows and columns
    # of the image
    
    #imm = rgb2gray(img)
    img = cv2.imread(iop,0)
    img_w=img.shape[0]
    img_h=img.shape[1]

 

    # Traverse the image. For every 3X3 area,
    # find the median of the pixels and
    # replace the center pixel by the median



    img_new1 = np.ones([img_w, img_h])



    for i in range(1,img_w - 1):
        for j in range(1,img_h - 1):
            
            temp =[img[i-1, j-1],
                   img[i-1, j],
                   img[i-1, j + 1],
                   img[i, j-1],
                   img[i, j],
                   img[i, j + 1],
                   img[i + 1, j-1],
                   img[i + 1, j],
                   img[i + 1, j + 1]]

            temp = sorted(temp)
            img_new1[i, j]= temp[4]
            #img_new1[i, j]= temp

    img_new1 = img_new1.astype(np.uint8)

    cv2.imshow("ee",img)
    cv2.imshow("dd",img_new1)
    cv2.waitKey(0)

def sobel():
    ################################################################## Sobel.################################################
    imm = cv2.imread(iop)
    img = rgb2gray(imm)
    filterV = [[-1,0,1],[-2,0,2],[-1,0,1]]
    filterH = [[-1,-2,-1],[0,0,0],[1,2,1]]
    s, v = img.shape
    newimggg = np.zeros([s,v])
    newimgggg = np.zeros([s,v])
    result = np.zeros([s,v])
    for i in range(1, s-1):
        for j in range(1,v-1):
            sobelH = img[i - 1][j - 1] * filterH[0][0] + img[i - 1][j] * filterH[0][1] + img[i - 1][j + 1] * filterH[0][2] + img[i][j - 1] * filterH[1][0] + img[i][j] * filterH[1][1] + img[i][j + 1] * filterH[1][2] + img[i + 1][j - 1] * filterH[2][0] + img[i + 1][j] * filterH[2][1] + img[i + 1][j + 1] * filterH[2][2]
            sobelV = img[i - 1][j - 1] * filterV[0][0] + img[i - 1][j] * filterV[0][1] + img[i - 1][j + 1] * filterV[0][2] + img[i][j - 1] * filterV[1][0] + img[i][j] * filterV[1][1] + img[i][j + 1] * filterV[1][2] + img[i + 1][j - 1] * filterV[2][0] + img[i + 1][j] * filterV[2][1] + img[i + 1][j + 1] * filterV[2][2]
            newimggg[i][j] = sobelV
            newimgggg[i][j] = sobelH
            result[i][j] = math.sqrt(pow(newimggg[i,j],2)+pow(newimgggg[i,j],2))
    cv2.imshow("Original",img)
    #cv2_imshow(newimggg)
    #cv2_imshow(newimgggg)
    cv2.imshow("Sobel",result)

    #llk = Label(root, text= s ).grid()
    
def Robert():
        ###########################################  Robert-cross   ##########################################################
    # Obtain the number of rows and columns
    # of the image
    global result2
    imm = cv2.imread(iop)
    img = rgb2gray(imm)
    m, n = img.shape
    filtercv = [[1,0],[0,-1]]
    filtervc = [[0,1],[-1,0]]

    # Traverse the image. For every 3X3 area,
    # find the median of the pixels and
    # replace the center pixel by the median
    img_new1 = np.zeros([m, n])
    img_new2 = np.zeros([m, n])
    result2 = np.zeros([m, n])
    for i in range(1, m):
        for j in range(1, n):
            roberth = img[i - 1][j - 1] * filtercv[0][0] + img[i - 1][j] * filtercv[0][1] + img[i][j - 1] * filtercv[1][0] + img[i][j] * filtercv[1][1]
            robertv = img[i - 1][j - 1] * filtervc[0][0] + img[i - 1][j] * filtervc[0][1] + img[i][j - 1] * filtervc[1][0] + img[i][j] * filtervc[1][1]
            img_new1[i][j] = roberth
            img_new2[i][j] = robertv
            result2[i][j] = math.sqrt(pow(img_new1[i,j],2)+pow(img_new2[i,j],2))

    cv2.imshow("Original",img)
    #cv2_imshow(img_new1)
    #cv2_imshow(img_new2)
    cv2.imshow("Robert-cross",result2)
    
def Laplacian():
    ################################################################## Laplacian  ################################################
    filter1 = [[0,-1,0],[-1,4,-1],[0,-1,0]]
    filter2 = [[1,1,1],[1,-8,1],[1,1,1]]
    filter3 = [[0,1,0],[1,-4,1],[0,1,0]]
    filter4 = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]

    imm = cv2.imread(iop)
    img = rgb2gray(imm)

    s, v = img.shape
    imagemask1 = np.zeros([s,v])
    imagemask2 = np.zeros([s,v])
    imagemask3 = np.zeros([s,v])
    imagemask4 = np.zeros([s,v])

    Lap_image1 = np.zeros([s,v])
    Lap_image2 = np.zeros([s,v])
    Lap_image3 = np.zeros([s,v])
    Lap_image4 = np.zeros([s,v])

    for i in range(1, s-1):
        for j in range(1,v-1):

            xc1  = img[i - 1][j - 1] * filter1[0][0]  +  img[i - 1][j] * filter1[0][1] + img[i - 1][j + 1] * filter1[0][2] + img[i][j - 1] * filter1[1][0] + img[i][j] * filter1[1][1] + img[i][j + 1] * filter1[1][2] + img[i + 1][j - 1] * filter1[2][0] + img[i + 1][j] * filter1[2][1] + img[i + 1][j + 1] * filter1[2][2]
            xc2 = img[i - 1][j - 1] * filter2[0][0] + img[i - 1][j] * filter2[0][1] + img[i - 1][j + 1] * filter2[0][2] + img[i][j - 1] * filter2[1][0] + img[i][j] * filter2[1][1] + img[i][j + 1] * filter2[1][2] + img[i + 1][j - 1] * filter2[2][0] + img[i + 1][j] * filter2[2][1] + img[i + 1][j + 1] * filter2[2][2]
            xc3 = img[i - 1][j - 1] * filter3[0][0] + img[i - 1][j] * filter3[0][1] + img[i - 1][j + 1] * filter3[0][2] + img[i][j - 1] * filter3[1][0] + img[i][j] * filter3[1][1] + img[i][j + 1] * filter3[1][2] + img[i + 1][j - 1] * filter3[2][0] + img[i + 1][j] * filter3[2][1] + img[i + 1][j + 1] * filter3[2][2]
            xc4 = img[i - 1][j - 1] * filter4[0][0] + img[i - 1][j] * filter4[0][1] + img[i - 1][j + 1] * filter4[0][2] + img[i][j - 1] * filter4[1][0] + img[i][j] * filter4[1][1] + img[i][j + 1] * filter4[1][2] + img[i + 1][j - 1] * filter4[2][0] + img[i + 1][j] * filter4[2][1] + img[i + 1][j + 1] * filter4[2][2]

            imagemask1[i][j] = xc1
            imagemask2[i][j] = xc2
            imagemask3[i][j] = xc3
            imagemask4[i][j] = xc4

            Lap_image1[i][j] = imagemask1[i][j] + img[i][j] 
            Lap_image2[i][j] = imagemask2[i][j] + img[i][j] 
            Lap_image3[i][j] = imagemask3[i][j] + img[i][j] 
            Lap_image4[i][j] = imagemask4[i][j] + img[i][j] 


    cv2.imshow("one",img)
    #cv2_imshow(imagemask1)
    #cv2_imshow(imagemask2)
    #cv2_imshow(imagemask3)
    #cv2_imshow(imagemask4)

    cv2.imshow("two",Lap_image1)
    cv2.imshow("three",Lap_image2)
    cv2.imshow("four",Lap_image3)
    cv2.imshow("five",Lap_image4)

    llk = Label(root, text= s ).grid()


def Unsharp():
    ################################################################## UNsharp ################################################
    imgg = cv2.imread(iop,0)
    
    x, y = imgg.shape
    # Develop Averaging filter(3, 3) mask
    mask = np.ones([3, 3], dtype = int)
    mask = mask / 9

    # Convolve the 3X3 mask over the image
    img_new = np.zeros([x, y])

    for i in range(1, x-1):
        for j in range(1, y-1):
            temp = imgg[i-1, j-1]*mask[0, 0]+imgg[i-1, j]*mask[0, 1]+imgg[i-1, j + 1]*mask[0, 2]+imgg[i, j-1]*mask[1, 0]+ imgg[i, j]*mask[1, 1]+imgg[i, j + 1]*mask[1, 2]+imgg[i + 1, j-1]*mask[2, 0]+imgg[i + 1, j]*mask[2, 1]+imgg[i + 1, j + 1]*mask[2, 2]
            img_new[i, j]= temp       

    img_new = img_new.astype(np.uint8)

    r, c = imgg.shape
    unsharped =np.zeros([r,c])
    sharped=np.zeros([r,c])

    for i in range(1, r-1):
        for j in range(1,c-1):
            unsharped[i][j] = imgg[(i,j)] - img_new[(i,j)]
            sharped[i][j] = imgg[(i,j)] + unsharped[i][j]

    cv2.imshow("Original",imgg)
    cv2.imshow("avg",img_new)
    cv2.imshow("UNsharp",unsharped)
    cv2.imshow("Sharp",sharped)

    k = 5
    boosting=np.zeros([r,c])
    for i in range(0, r-1):
        for j in range(0,c-1):
            boosting[i][j] = imgg[(i,j)] + k *  unsharped[i][j]
    cv2.imshow("boost",boosting)

def Gussian_n():

    img = cv2.imread(iop,0)
    #img = rgb2gray(imm)
    
    img = img/255
    r = img.shape[0]
    c = img.shape[1]

    mean = 0
    std = 0.01

    noise = np.multiply(np.random.normal(mean, std, img.shape), 255)
    img_gau = np.clip(img.astype(int) + noise, 0, 255)

    RES = img_gau - img

    aws = np.zeros([r, c])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            aws[i,j] =img[i,j] - RES[i,j]  



    cv2.imshow('Gray_Image',img)
    cv2.imshow('Gaussian_Noise',RES)
    cv2.imshow("finalllllly", aws)
    cv2.waitKey(0)

def Impulse():

    imm = cv2.imread(iop)
    img = rgb2gray(imm)
    p = 0.01
    s, v = img.shape
    ni = np.zeros([s,v])
    for i in range(1, s-1):
        for j in range(1,v-1):
            ni[i][j] = img[i][j]
    for x in range(1, s-1):
        for y in range(1, v-1):
            pix = np.random.random()
            if pix < p:
                sp = np.random.random()
                if sp > 0.5:
                    ni[x,y] = 0
                else:
                     ni[x,y] = 255
            
    cv2.imshow("ori",img)            
    cv2.imshow("res",ni)
    

def Gussian_f():

    img = cv2.imread(iop,0)

    def corr(img,mask):
        

        row,col=img.shape
        m,n=mask.shape
        new=np.zeros((row+m-1,col+n-1))
        n=n//2
        m=m//2
        filtered_img=np.zeros(img.shape)
        iu = img
        new[m:new.shape[0]-m,n:new.shape[1]-n]=img
        for i in range (m,new.shape[0]-m):
            for j in range(n,new.shape[1]-n):
                temp=new[i-m:i+m+1,j-m:j+m+1]
                result=temp*mask
                filtered_img[i-m,j-n]=result.sum()
        return filtered_img


    
    def gaussian(m,n,sigma):
        gaussian=np.zeros((m,n))
        m=m//2
        n=n//2
        for x in range (-m,m+1):
            for y in range(-n,n+1):
                x1=sigma*(2*np.pi)**2
                x2=np.exp(-(x*2+y*2)/(2*sigma*2))
                gaussian[x+m,y+n]=(1/x1)*x2
        return gaussian


    g=gaussian(5,5,2)
    n=corr(img,g)  
    plt.imshow(img)
    cv2.imshow("res", img)
    cv2.waitKey(0)

def Adaptive():

    img = cv2.imread(iop,0)

    width = img.shape[0]
    height = img.shape[1]

    Updated_Image = np.zeros([width, height])

    noise_variance = 10.0
    Local_variance = 5.0
    Local_mean = 7.0

    #Convert to integer
    floor_value1 = (noise_variance) 
    floor_value2 = (Local_variance)
    floor_value3 = (Local_mean)

    for i in range(width - 1):
        for j in range(height - 1):
            val1 = np.square(floor_value1) // np.square(floor_value2)
            val = (img[i][j] - (val1 * (img[i][j] - Local_mean) ) )
            Updated_Image[i][j] = val

    cv2.imshow("rr",Updated_Image)
    cv2.waitKey(0)

def Histogram_e():
    image = cv2.imread(ioc,0)
    def Get_the_Histogram(image):
        Get_the_Histogram = np.zeros(shape = (256,1))  
        for i in range(image.shape[0]):  
            for j in range(image.shape[1]):
                 k = image[i,j]
                 Get_the_Histogram[k,0] = Get_the_Histogram[k,0] + 1
        return Get_the_Histogram


    x = Get_the_Histogram.reshape(1,256)
    y = np.array([])
    y = np.append(y,x[0,0])

    for i in range(255):
        k = x[0,i+1] + y[i]
        y = np.append(y,k)
        y = np.round((y/(image.shape[0]*image.shape[1]))*255)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            k = image[i,j]
            image[i,j] = y[k]

    def Get_the_Histogram(image):
        Get_the_Histogram = np.zeros(shape = (256,1))  
        for i in range(image.shape[0]):  
            for j in range(image.shape[1]):
                k = image[i,j]
                Get_the_Histogram[k,0] = Get_the_Histogram[k,0] + 1
        return Get_the_Histogram

    Get_the_Histogram = Get_the_Histogram(image)
    plt.plot(Get_the_Histogram)
    #cv2.imshow(image)
    Get_the_Histogram = Get_the_Histogram(image)
    plt.plot(Get_the_Histogram)

def Histogram_s():
    
    def find_nearest_above(my_array, target):
        diff = my_array - target
        mask = np.ma.less_equal(diff, -1)
        # We need to mask the negative differences
        # since we are looking for values above
        if np.all(mask):
            c = np.abs(diff).argmin()
            return c # returns min index of the nearest if target is greater than any value
        masked_diff = np.ma.masked_array(diff, mask)
        return masked_diff.argmin()


    def hist_match(original, specified):

        oldshape = original.shape()
        original = original.ravel()
        specified = specified.ravel()

        # get the set of unique pixel values and their corresponding indices and counts
        s_values, bin_idx, s_counts = np.unique(original, return_inverse=True,return_counts=True)
        t_values, t_counts = np.unique(specified, return_counts=True)

        # Calculate s_k for original image
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        
        # Calculate s_k for specified image
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # Round the values
        sour = np.around(s_quantiles*255)
        temp = np.around(t_quantiles*255)
        
        # Map the rounded values
        b=[]
        for data in sour[:]:
            b.append(find_nearest_above(temp,data))
        b= np.array(b,dtype='uint8')

        return b[bin_idx].reshape(oldshape)
        

        # Load the images in greyscale
    original = cv2.imread('ioc',0)
    specified = cv2.imread(r'C:\Users\seifm\Downloads\Image_created_with_a_mobile_phone.png',0)

    # perform Histogram Matching
    a = hist_match(original, specified)

    # Display the image
    cv2.imshow('a',np.array(a,dtype='uint8'))
    cv2.imshow('a1',original)
    cv2.imshow('a2',specified)
    cv2.waitKey(0)
    
def Interpolation():
    ###################################################################### NEAREST NEIBOUR ################################################
    imm = cv2.imread(ioc)
    img = rgb2gray(imm)
    
    img_w=img.shape[0]
    img_h=img.shape[1]
    nn = np.zeros([img_w,img_h])

    x = int(img_w*1/2)
    y = int(img_h*1/2)

    xs = x/img_w
    ys = y/img_h

    for i in range(x-5):
        for j in range(y-5):
            nn[i-1][j-1]=img[int(i/xs),int(j/ys)]

    cv2.imshow("ori",img)            
    cv2.imshow("res",nn)

def Compression():

    img = Image.open(r'C:\Users\seifm\Downloads\x-ray.jpg')

    plt.figure(figsize=(9,6))
    plt.imshow(img)


    imggray=img.convert('LA')
    plt.figure(figsize=(9,6))
    plt.imshow(imggray)
    #cv2.imshow("first",imggray)
    print(imggray)

    imgmat=np.array(list(imggray.getdata(band=0)), float)
    imgmat.shape=(imggray.size[1], imggray.size[0])
    imgmat=np.matrix(imgmat)
    plt.figure(figsize=(9,6))
    plt.imshow(imgmat, cmap='gray')
    cv2.imshow("second",imgmat)

    imgmat.shape


    u, sigma, v = np.linalg.svd(imgmat)


    sigma[0]


    for term_number in range(1,41,5):
        reconsting = np.matrix(u[:, :term_number])* np.diag(sigma[:term_number])* np.matrix(v[:term_number, :])
        plt.figure(figsize=(9,6))
        cv2.imshow("third",reconsting)
        plt.imshow(reconsting, cmap='gray')

    plt.plot(sigma,'ob')
    plt.xlabel("i")
    plt.ylabel("sigma_i")
    
    cv2.imshow("fourth", sigma)

def Negative():

    img = cv2.imread(iod,0)
    for i in range (img.shape[0]-1):
        for j in range (img.shape[1]-1):
            img[(i,j)] = 255 - img[(i,j)]
    
    cv2.imshow("res",img)

def LogTrans():

    img = cv2.imread(iod,0)
    for i in range (img.shape[0]-1):
        for j in range (img.shape[1]-1):
            img[(i,j)] = 22 * math.log(1 + img[(i,j)])
    
    cv2.imshow("res",img)

def Gamma():

    img = cv2.imread(iod,0)

    for i in range (img.shape[0]-1):
        for j in range (img.shape[1]-1):

            img[(i,j)] = 10 * math.pow(img[(i,j)], 3)
    
    cv2.imshow("res",img)

#my_label2 = Label(root, height=5, width= 50)
#my_label2.grid(row=1, column=0)
my_button = Button(root, text = "Spatial Domain", command = lambda:Spatial(), height= 2, width= 15, bg="#6ECCAF")
my_button1 = Button(root, text = "Frequency Domain", command = lambda:Frequency(), height= 2, width= 15, bg="#6ECCAF")
my_button3 = Button(root, text= "Other Filters", command= lambda:Other(),height= 2, width= 15, bg="#6ECCAF").place(x= 700 , y = 300)
my_button2 = Button(root, text = "compression", command= lambda:Compression(), height= 2, width= 15, bg="#6ECCAF").place( x = 850, y=300)
my_button.place(x= 400, y= 300)
my_button1.place(x= 550, y=300 )
root.mainloop()