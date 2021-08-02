# DRA-UNet

## Abstract
According to statistics in Taiwan, **breast cancer is the third-commonest reason which causes death**. This fact really pose a significant threat to women’s health. From 2009 to 2019, with the increment of death rate, medical ultrasound imaging has been widely employed to segment breast lumps because its **safety, painless characteristics, noninvasive diagnosis and non-ionized radiation**. Furthermore, compared with other clinical medical imaging such as CT and MRI, ultrasound is relatively **cheaper, portable and general-used**. Nevertheless, it requires the subjective judgement from radiologists with relevant experience and the annotation is **laborious and time-consuming, resulting in scarcity of data** and bringing more challenges for implementing deep learning technologies on analyzing ultrasound images.   

In recent years, deep learning in computer vision has demonstrated the potential in a vast repertoire of biomedical image segmentation tasks. With the development of medical equipment, professionals prefer to miniaturize the ultrasound devices to leverage the efficiency and portability. Therefore, the deep learning models should also be as lightweight as possible to strike a balance among time consumption, accuracy and stability. In this thesis, I used **six existing neural networks for biomedical images** and took breast ultrasound raw images and their ground truths to derive the corresponding weights through end-to-end training and categorize the lesions pixel by pixel in the end. Thanks to other brilliant structures that the great researchers proposed before, I combined them and developed a novel architecture, **DRA-UNet (Dense-Res-Attention UNet)**, as a solution of assisting the professionals to delineate the tumor area.    

For the sake of objectively analyzing the results of tumor segmentation, the six error metrics of JSI, DSC, ACC, TPR, TNR, and Precision were used to evaluate the goodness of the model. **The DRA-UNet has the highest JSI of 78.10%, DSC of 85.79%, ACC of 97.81% and Precision of 89.72%, while TPR of 89.47% and TNR of 98.79% are the second best.** Thus, the proposed method really can be improved based on existing methods and provide appropriate detection of tumors and lesions at early stages of breast cancer. In summary, the proposed method has the following advantages. First, the model does not require any manual adjustment, which saves valuable healthcare manpower and time; second, it has excellent pixel-level segmenting ability even with a small number of parameters and sparse training data; third, it can maintain a stable level in ultrasound images with high difficulty, such as very tiny lesions and severe acoustic shadowing.    

---

## How to use
* main_code.py -> for training model
* figure.py -> for predicting the segmentation image and performance evaluation
* printmodel.py -> for plotting the model to help realizing the whole structure
* data_augment.py -> for data augmentation using tensorflow(keras) ImageDataGenerator
* keras_unet -> thera are some tools, metrics, losses
 - models -> including DRA-UNet which we proposed and the others
 - utils -> data augment, color mask, plot figure, save images, data type transformation...
 - metrics -> JSI(IoU), DSC(F1-score), Accuracy, Precision, Sensitivity, Specificity...
 - losses


---

### * Preprocessing(load dataset) 
if you need the dataset, please contact with me: a0956525116@gmail.com
```py
 ### Load Data ###
    mypath = "D:/datasets/trainV1"
    sid = os.listdir(mypath)
    
    x_data = []
    y_data = []
    for s in sid:
        img = cv2.imread('D:/datasets/trainV1/'+ s +'/images/'+ s + '.jpg', cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img,(256, 256), interpolation = cv2.INTER_CUBIC)
        x_data.append(resized_img)
    
        msk = cv2.imread('D:/datasets/trainV1/'+ s +'/masks/'+ s + '_gt.jpg', cv2.IMREAD_GRAYSCALE)
        resized_msk = cv2.resize(msk,(256, 256), interpolation = cv2.INTER_CUBIC)
        y_data.append(resized_msk)
        
    imgs_np = np.asarray(x_data)
    masks_np = np.asarray(y_data)
    
    x = np.asarray(imgs_np, dtype=np.float32)/255
    y = np.asarray(masks_np, dtype=np.float32)/255 
    y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
```

---

## Model & Results
**Our proposed model**

![](https://i.imgur.com/QUd0F8T.png)

**Example about confusion matrix**

![](https://i.imgur.com/sAlZu98.png)
