import numpy as np
import matplotlib.pyplot as plt
import os

from keras_unet import TF
if TF:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
else:
    from keras.preprocessing.image import ImageDataGenerator

MASK_COLORS = [
    "red", "green", "blue",
    "yellow", "magenta", "cyan"
]

def combine_generator(gen1, gen2):
    while True:
        yield(gen1.next(), gen2.next())

# Runtime data augmentation
def get_augmented(
    X_train,
    Y_train,
    X_val=None,
    Y_val=None,
    batch_size=32,
    seed=0,
    data_gen_args=dict(
        rotation_range=10.0,
        # width_shift_range=0.02,
        height_shift_range=0.02,
        shear_range=5,
        # zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="constant",
    ),
):
    """[summary]
    
    Args:
        X_train (numpy.ndarray): [description]
        Y_train (numpy.ndarray): [description]
        X_val (numpy.ndarray, optional): [description]. Defaults to None.
        Y_val (numpy.ndarray, optional): [description]. Defaults to None.
        batch_size (int, optional): [description]. Defaults to 32.
        seed (int, optional): [description]. Defaults to 0.
        data_gen_args ([type], optional): [description]. Defaults to dict(rotation_range=10.0,# width_shift_range=0.02,height_shift_range=0.02,shear_range=5,# zoom_range=0.3,horizontal_flip=True,vertical_flip=False,fill_mode="constant",).
    
    Returns:
        [type]: [description]
    """

    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(
        X_train, batch_size=batch_size, shuffle=True, seed=seed
    )
    Y_train_augmented = Y_datagen.flow(
        Y_train, batch_size=batch_size, shuffle=True, seed=seed
    )

    train_generator = combine_generator(X_train_augmented, Y_train_augmented)

    if not (X_val is None) and not (Y_val is None):
        # Validation data, no data augmentation, but we create a generator anyway
        X_datagen_val = ImageDataGenerator(**data_gen_args)
        Y_datagen_val = ImageDataGenerator(**data_gen_args)
        X_datagen_val.fit(X_val, augment=False, seed=seed)
        Y_datagen_val.fit(Y_val, augment=False, seed=seed)
        X_val_augmented = X_datagen_val.flow(
            X_val, batch_size=batch_size, shuffle=False, seed=seed
        )
        Y_val_augmented = Y_datagen_val.flow(
            Y_val, batch_size=batch_size, shuffle=False, seed=seed
        )

        # combine generators into one which yields image and masks
        val_generator = combine_generator(X_val_augmented, Y_val_augmented)

        return train_generator, val_generator
    else:
        return train_generator


def plot_segm_history_iou(history, modelname , metrics=["iou", "val_iou"], losses=["loss", "val_loss"]):
    """[summary]
    
    Args:
        history ([type]): [description]
        metrics (list, optional): [description]. Defaults to ["iou", "val_iou"].
        losses (list, optional): [description]. Defaults to ["loss", "val_loss"].
    """
    # summarize history for iou
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle("IOU(Jacard) over epochs", fontsize=20)
    plt.ylabel("IOU(Jacard)", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    # plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    # plt.xticks(fontsize=35)
    plt.xticks([0.0,10,20,30,40,50,60,70,80,90,100])
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
    plt.legend(metrics, loc="lower right", fontsize=15)
    plt.show()
    plt.savefig(modelname + '/IOU.png',format='png')
    plt.close()
    # summarize history for loss
    plt.figure(figsize=(12, 6))
    for loss in losses:
        plt.plot(history.history[loss], linewidth=3)
    plt.suptitle("loss over epochs", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    # plt.yticks(np.arange(0, 0.2, step=0.005), fontsize=35)
    # plt.xticks(fontsize=35)
    plt.xticks([0.0,10,20,30,40,50,60,70,80,90,100])
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
    plt.legend(losses, loc="upper right", fontsize=15)
    plt.show()
    plt.savefig(modelname + '/loss.png',format='png')
    plt.close()
    
def plot_segm_history_acc(history, modelname, metrics=["acc", "val_acc"]):
    # summarize history for acc
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle("ACC over epochs", fontsize=20)
    plt.ylabel("ACC", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    # plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    # plt.xticks(fontsize=35)
    plt.xticks([0.0,10,20,30,40,50,60,70,80,90,100])
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
    plt.legend(metrics, loc="lower right", fontsize=15)
    plt.show()
    plt.savefig(modelname + '/Acc.png',format='png')
    plt.close()
    # summarize history for loss
    # plt.figure(figsize=(12, 6))
    # for loss in losses:
    #     plt.plot(history.history[loss], linewidth=3)
    # plt.suptitle("loss over epochs", fontsize=20)
    # plt.ylabel("loss", fontsize=20)
    # plt.xlabel("epoch", fontsize=20)
    # # plt.yticks(np.arange(0, 0.2, step=0.005), fontsize=35)
    # # plt.xticks(fontsize=35)
    # plt.xticks([0.0,10,20,30,40,50,60,70,80,90,100])
    # plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    # plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
    # plt.legend(losses, loc="upper right", fontsize=15)
    # plt.show()
    
def plot_segm_history_dice(history, modelname, metrics=["dice_coef", "val_dice_coef"]):
    # summarize history for dice_coef
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle("Dice coef over epochs", fontsize=20)
    plt.ylabel("Dice coef", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    # plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    # plt.xticks(fontsize=35)
    plt.xticks([0.0,10,20,30,40,50,60,70,80,90,100])
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
    plt.legend(metrics, loc="lower right", fontsize=15)
    plt.show()
    plt.savefig(modelname + '/Dice.png',format='png')
    plt.close()
    # summarize history for loss
    # plt.figure(figsize=(12, 6))
    # for loss in losses:
    #     plt.plot(history.history[loss], linewidth=3)
    # plt.suptitle("loss over epochs", fontsize=20)
    # plt.ylabel("loss", fontsize=20)
    # plt.xlabel("epoch", fontsize=20)
    # # plt.yticks(np.arange(0, 0.2, step=0.005), fontsize=35)
    # # plt.xticks(fontsize=35)
    # plt.xticks([0.0,10,20,30,40,50,60,70,80,90,100])
    # plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    # plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
    # plt.legend(losses, loc="upper right", fontsize=15)
    # plt.show()
    
def plot_segm_history_SE(history, modelname, metrics=["sensitivity", "val_sensitivity"]):
    # summarize history for dice_coef
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle("Sensitivity over epochs", fontsize=20)
    plt.ylabel("Sensitivity", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    # plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    # plt.xticks(fontsize=35)
    plt.xticks([0.0,10,20,30,40,50,60,70,80,90,100])
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
    plt.legend(metrics, loc="lower right", fontsize=15)
    plt.show()
    plt.savefig(modelname + '/SE.png',format='png')
    plt.close()
    
def plot_segm_history_PC(history, modelname, metrics=["precision", "val_precision"]):
    # summarize history for dice_coef
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle("Precision over epochs", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    # plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    # plt.xticks(fontsize=35)
    plt.xticks([0.0,10,20,30,40,50,60,70,80,90,100])
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
    plt.legend(metrics, loc="lower right", fontsize=15)
    plt.show()
    plt.savefig(modelname + '/PC.png',format='png')
    plt.close()
    
def plot_segm_history_SP(history, modelname, metrics=["specificity", "val_specificity"]):
    # summarize history for dice_coef
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle("Specificity over epochs", fontsize=20)
    plt.ylabel("Specificity", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    # plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    # plt.xticks(fontsize=35)
    plt.xticks([0.0,10,20,30,40,50,60,70,80,90,100])
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
    plt.legend(metrics, loc="lower right", fontsize=15)
    plt.show()
    plt.savefig(modelname + '/SP.png',format='png')
    plt.close()
    


def mask_to_red(mask):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    
    Args:
        mask (numpy.ndarray): [description]
    
    Returns:
        numpy.ndarray: [description]
    """
    img_size = mask.shape[0]
    c1 = mask.reshape(img_size, img_size)
    c2 = np.zeros((img_size, img_size))
    c3 = np.zeros((img_size, img_size))
    c4 = mask.reshape(img_size, img_size)
    return np.stack((c1, c2, c3, c4), axis=-1)


def mask_to_rgba(mask, color="red"):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    
    Args:
        mask (numpy.ndarray): [description]
        color (str, optional): Check `MASK_COLORS` for available colors. Defaults to "red".
    
    Returns:
        numpy.ndarray: [description]
    """    
    assert(color in MASK_COLORS)
    assert(mask.ndim==3 or mask.ndim==2)

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones, ones), axis=-1)


def plot_imgs(
        org_imgs,
        mask_imgs,
        pred_imgs=None,
        nm_img_to_plot=10,
        figsize=4,
        alpha=0.5,
        color="red"):
    """
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.

    Args:
        org_imgs (numpy.ndarray): Array of arrays representing a collection of original images.
        mask_imgs (numpy.ndarray): Array of arrays representing a collection of mask images (grayscale).
        pred_imgs (numpy.ndarray, optional): Array of arrays representing a collection of prediction masks images.. Defaults to None.
        nm_img_to_plot (int, optional): How many images to display. Takes first N images. Defaults to 10.
        figsize (int, optional): Matplotlib figsize. Defaults to 4.
        alpha (float, optional): Transparency for mask overlay on original image. Defaults to 0.5.
        color (str, optional): Color for mask overlay. Defaults to "red".
    """ # NOQA E501
    assert(color in MASK_COLORS)

    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

    fig, axes = plt.subplots(
        nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize), squeeze=False
    )
    axes[0, 0].set_title("original", fontsize=15)
    axes[0, 1].set_title("ground truth", fontsize=15)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=15)
        axes[0, 3].set_title("overlay", fontsize=15)
    else:
        axes[0, 2].set_title("overlay", fontsize=15)
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 3].imshow(
                mask_to_rgba(
                    zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size),
                    color=color,
                ),
                cmap=get_cmap(pred_imgs),
                alpha=alpha,
            )
            # new ground truth overlay
            axes[m,3].imshow(mask_to_rgba(
                    zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size),
                    color="yellow",
                ),
                cmap=get_cmap(mask_imgs),
                alpha=0.3,)
            # -------------------------
            axes[m, 3].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(
                mask_to_rgba(
                    zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size),
                    color=color,
                ),
                cmap=get_cmap(mask_imgs),
                alpha=alpha,
            )
            axes[m, 2].set_axis_off()
        im_id += 1

    plt.show()


def zero_pad_mask(mask, desired_size):
    """[summary]
    
    Args:
        mask (numpy.ndarray): [description]
        desired_size ([type]): [description]
    
    Returns:
        numpy.ndarray: [description]
    """
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask


def reshape_arr(arr):
    """[summary]
    
    Args:
        arr (numpy.ndarray): [description]
    
    Returns:
        numpy.ndarray: [description]
    """
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])


def get_cmap(arr):
    """[summary]
    
    Args:
        arr (numpy.ndarray): [description]
    
    Returns:
        string: [description]
    """
    if arr.ndim == 3:
        return "gray"
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return "jet"
        elif arr.shape[3] == 1:
            return "gray"


def get_patches(img_arr, size=256, stride=256):
    """
    Takes single image or array of images and returns
    crops using sliding window method.
    If stride < size it will do overlapping.
    
    Args:
        img_arr (numpy.ndarray): [description]
        size (int, optional): [description]. Defaults to 256.
        stride (int, optional): [description]. Defaults to 256.
    
    Raises:
        ValueError: [description]
        ValueError: [description]
    
    Returns:
        numpy.ndarray: [description]
    """    
    # check size and stride
    if size % stride != 0:
        raise ValueError("size % stride must be equal 0")

    patches_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1

    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride - overlapping

        for i in range(i_max):
            for j in range(i_max):
                # print(i*stride, i*stride+size)
                # print(j*stride, j*stride+size)
                patches_list.append(
                    img_arr[
                        i * stride : i * stride + size,
                        j * stride : j * stride + size
                    ]
                )

    elif img_arr.ndim == 4:
        i_max = img_arr.shape[1] // stride - overlapping
        for im in img_arr:
            for i in range(i_max):
                for j in range(i_max):
                    # print(i*stride, i*stride+size)
                    # print(j*stride, j*stride+size)
                    patches_list.append(
                        im[
                            i * stride : i * stride + size,
                            j * stride : j * stride + size,
                        ]
                    )

    else:
        raise ValueError("img_arr.ndim must be equal 3 or 4")

    return np.stack(patches_list)


def plot_patches(img_arr, org_img_size, stride=None, size=None):
    """
    Plots all the patches for the first image in 'img_arr' trying to reconstruct the original image

    Args:
        img_arr (numpy.ndarray): [description]
        org_img_size (tuple): [description]
        stride ([type], optional): [description]. Defaults to None.
        size ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]
    """

    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    fig, axes = plt.subplots(i_max, j_max, figsize=(i_max * 2, j_max * 2))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    jj = 0
    for i in range(i_max):
        for j in range(j_max):
            axes[i, j].imshow(img_arr[jj])
            axes[i, j].set_axis_off()
            jj += 1


def reconstruct_from_patches(img_arr, org_img_size, stride=None, size=None):
    """[summary]
    
    Args:
        img_arr (numpy.ndarray): [description]
        org_img_size (tuple): [description]
        stride ([type], optional): [description]. Defaults to None.
        size ([type], optional): [description]. Defaults to None.
    
    Raises:
        ValueError: [description]
    
    Returns:
        numpy.ndarray: [description]
    """
    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    nm_layers = img_arr.shape[3]

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    total_nm_images = img_arr.shape[0] // (i_max ** 2)
    nm_images = img_arr.shape[0]

    averaging_value = size // stride
    images_list = []
    kk = 0
    for img_count in range(total_nm_images):
        img_bg = np.zeros(
            (org_img_size[0], org_img_size[1], nm_layers), dtype=img_arr[0].dtype
        )

        for i in range(i_max):
            for j in range(j_max):
                for layer in range(nm_layers):
                    img_bg[
                        i * stride : i * stride + size,
                        j * stride : j * stride + size,
                        layer,
                    ] = img_arr[kk, :, :, layer]

                kk += 1
        # TODO add averaging for masks - right now it's just overwritting

        #         for layer in range(nm_layers):
        #             # average some more because overlapping 4 patches
        #             img_bg[stride:i_max*stride, stride:i_max*stride, layer] //= averaging_value
        #             # corners:
        #             img_bg[0:stride, 0:stride, layer] *= averaging_value
        #             img_bg[i_max*stride:i_max*stride+stride, 0:stride, layer] *= averaging_value
        #             img_bg[i_max*stride:i_max*stride+stride, i_max*stride:i_max*stride+stride, layer] *= averaging_value
        #             img_bg[0:stride, i_max*stride:i_max*stride+stride, layer] *= averaging_value

        images_list.append(img_bg)

    return np.stack(images_list)


###　Save model ###
def saveModel(model, modelname):

    model_json = model.to_json()
    
    try:
        os.makedirs(modelname)
    except:
        pass
    
    fp = open(modelname + '/modelP.json','w')
    fp.write(model_json)
    model.save_weights(modelname + '/modelW.h5')
    
### Evaluate the Model ###
def evaluateModel(model, X_test, Y_test, batchSize, modelname, dirname):

    try:
        os.makedirs(dirname)
    except:
        pass 
    

    yp = model.predict(x=X_test, batch_size=batchSize, verbose=0)

    yp = np.round(yp,0)

    for i in range(10):
        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(X_test[i])
        plt.title('Input')
        plt.subplot(1,3,2)
        plt.imshow(Y_test[i].reshape(Y_test[i].shape[0],Y_test[i].shape[1]), cmap="gray")
        plt.title('Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]), cmap="gray")
        plt.title('Prediction')

        intersection = yp[i].ravel() * Y_test[i].ravel()
        union = yp[i].ravel() + Y_test[i].ravel() - intersection

        jacard = (np.sum(intersection)/np.sum(union))  
        plt.suptitle('Jacard Index'+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jacard))

        plt.savefig(dirname + '/'+str(i)+'.png',format='png')
        plt.close()


    jacard = 0
    dice = 0
    
    
    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        
        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jacard += (np.sum(intersection)/np.sum(union))  

        dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))

    
    jacard /= len(Y_test)
    dice /= len(Y_test)
    


    print('Jacard Index : '+str(jacard))
    print('Dice Coefficient : '+str(dice))
    

    fp = open(modelname + '/log.txt','a')
    fp.write(str(jacard)+'/'+str(dice)+'\n')
    fp.close()

    fp = open(modelname + '/best.txt','r')
    best = fp.read()
    fp.close()

    if(jacard>float(best)):
        print('***********************************************')
        print('Jacard Index improved from '+str(best)+' to '+str(jacard))
        print('***********************************************')
        fp = open(modelname + '/best.txt','w') 
        fp.write(str(jacard))
        fp.close()

        saveModel(model, modelname)
        
### Training the Model ###
def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize, modelname, dirname):
    
    for epoch in range(epochs):
        print('Epoch : {}'.format(epoch+1))
        model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=1, verbose=1)     

        evaluateModel(model,X_test, Y_test,batchSize, modelname ,dirname)

    return model

def cal_base(y_true, y_pred):
    y_pred_positive = np.round(np.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive

    y_positive = np.round(np.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = np.sum(y_positive * y_pred_positive)
    TN = np.sum(y_negative * y_pred_negative)

    FP = np.sum(y_negative * y_pred_positive)
    FN = np.sum(y_positive * y_pred_negative)

    return TP, TN, FP, FN

def acc(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    ACC = (TP + TN) / (TP + FP + FN + TN + np.finfo(np.float32).eps)
    return ACC


def sensitivity(y_true, y_pred): # = recall
    """ recall """
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SE = TP/(TP + FN + np.finfo(np.float32).eps)
    return SE


def precision(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    PC = TP/(TP + FP + np.finfo(np.float32).eps)
    return PC


def specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP + np.finfo(np.float32).eps)
    return SP


def evaluate_result(model, X_test, Y_test, modelname, dirname):

    try:
        os.makedirs(dirname)
    except:
        pass 
    fp = open(modelname + '/log.txt','w')
    fp.close()
    fp = open(modelname + '/best.txt','w')
    fp.write('-1.0')
    fp.close()

    yp = model.predict(x=X_test, verbose=0)

    yp = np.round(yp,0)

    for i in range(len(Y_test)):
        
        fig, ax = plt.subplots(1, 4, figsize=(20,10), squeeze=False)

        ax[0,0].imshow(X_test[i])
        ax[0,0].set_title('Input')
        ax[0,0].set_axis_off()

        # ax[0,1].imshow(Y_test[i].reshape(Y_test[i].shape[0],Y_test[i].shape[1]), cmap="gray")
        # ax[0,1].set_title('Ground Truth')
        # ax[0,1].set_axis_off()
        ax[0,1].imshow(X_test[i], cmap = "gray")
        ax[0,1].imshow(mask_to_rgba(zero_pad_mask(Y_test[i].reshape(Y_test[i].shape[0],Y_test[i].shape[1]), desired_size = X_test[i].shape[1]),
                   color = "yellow"), cmap = get_cmap(Y_test[i]), alpha = 0.5,)
        ax[0,1].set_title('Ground Truth')
        ax[0,1].set_axis_off()
        

        # ax[0,2].imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]), cmap="gray")
        # ax[0,2].set_title('Prediction')
        # ax[0,2].set_axis_off()
        ax[0,2].imshow(X_test[i], cmap = "gray")
        ax[0,2].imshow(mask_to_rgba(zero_pad_mask(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]), desired_size = X_test[i].shape[1]),
                   color = "red"), cmap = get_cmap(yp[i]), alpha = 0.5,)
        ax[0,2].set_title('Prediction')
        ax[0,2].set_axis_off()

        ax[0,3].imshow(X_test[i], cmap = "gray")
        ax[0,3].imshow(mask_to_rgba(zero_pad_mask(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]), desired_size = X_test[i].shape[1]),
                   color = "red"), cmap = get_cmap(yp[i]), alpha = 0.5,)
        ax[0,3].imshow(mask_to_rgba(zero_pad_mask(Y_test[i].reshape(Y_test[i].shape[0],Y_test[i].shape[1]), desired_size = X_test[i].shape[1]),
                   color = "yellow"), cmap = get_cmap(Y_test[i]), alpha = 0.3,)
        ax[0,3].set_title('Overlay')
        ax[0,3].set_axis_off()
        

        yp1 = yp[i].ravel()
        y1 = Y_test[i].ravel()
        intersection = yp1 * y1
        union = yp1 + y1 - intersection

        # intersection = yp[i].ravel() * Y_test[i].ravel()
        # union = yp[i].ravel() + Y_test[i].ravel() - intersection

        jacard = (np.sum(intersection)/np.sum(union)) 
        dice = (2. *np.sum(intersection) ) / (np.sum(yp1) + np.sum(y1))
        plt.suptitle('Jacard Index = ' + str(jacard)+'\n'
                     +'Dice Coef = ' + str(dice))
        


        plt.savefig(dirname + '/'+str(i)+'.png',format='png')
        plt.close()


    jacard = 0.
    dice = 0.
    acc_test = 0.
    se_t = 0.
    pc_t = 0.
    sp_t = 0.
    
    
    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        
        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jacard += (np.sum(intersection)/np.sum(union))  

        dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))
        acc_test += acc(y2, yp_2)
        se_t += sensitivity(y2, yp_2)
        pc_t += precision(y2, yp_2)
        sp_t += specificity(y2, yp_2)     

    
    jacard /= len(Y_test)
    dice /= len(Y_test)
    acc_test /= len(Y_test) # name it because it collide with the function name
    se_t /= len(Y_test)
    pc_t /= len(Y_test)
    sp_t /= len(Y_test)


    print('Jacard Index : '+str(jacard))
    print('Dice Coefficient : '+str(dice))
    print('Accuracy : '+str(acc_test))
    print('Sensitivity : '+str(se_t))
    print('Precision : '+str(pc_t))
    print('Specificity : '+str(sp_t))
    

    fp = open(modelname + '/log.txt','a')
    fp.write('Jacard: '+str(jacard)+'\n'+
             'Dice Coefficient: '+str(dice)+'\n'+
             'Accuracy: '+str(acc_test)+'\n'+
             'Sensitivity: '+str(se_t)+'\n'+
             'Precision: '+str(pc_t)+'\n'+
             'Specificity: '+str(sp_t)+'\n')
    fp.close()

    fp = open(modelname + '/best.txt','r')
    best = fp.read()
    fp.close()

    if(jacard>float(best)):
        print('***********************************************')
        print('Jacard Index improved from '+str(best)+' to '+str(jacard))
        print('***********************************************')
        fp = open(modelname + '/best.txt','w') 
        fp.write(str(jacard))
        fp.close()

#return only the pixels contained in the FOV, for both images and masks
def pred_only_FOV(data_imgs,data_masks,original_imgs_border_masks):
    assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)  #4D arrays
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[1]==data_masks.shape[1])
    assert (data_imgs.shape[2]==data_masks.shape[2])
    assert (data_imgs.shape[3]==1 and data_masks.shape[3]==1)  #check the channel is 1
    height = data_imgs.shape[1]
    width = data_imgs.shape[2]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                # if inside_FOV_DRIVE(i,x,y,original_imgs_border_masks)==True:
                    new_pred_imgs.append(data_imgs[i,y,x,:])
                    new_pred_masks.append(data_masks[i,y,x,:])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks

def inside_FOV_DRIVE(i, x, y, DRIVE_masks):
    # assert (len(DRIVE_masks.shape)==4)  #4D arrays
    # assert (DRIVE_masks.shape[3]==1)  #DRIVE masks is black and white
    # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!

    if (x >= DRIVE_masks.shape[3] or y >= DRIVE_masks.shape[2]): #my image bigger than the original
        return False

    if (DRIVE_masks[i,0,y,x]>0):  #0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    else:
        return False
    
#function to set to black everything outside the FOV, in a full image
def kill_border(data, original_imgs_border_masks):
    assert (len(data.shape)==4)  #4D arrays
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,x,y,original_imgs_border_masks)==False:
                    data[i,:,y,x]=0.0