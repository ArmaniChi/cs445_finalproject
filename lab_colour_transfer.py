import cv2
import numpy as np

def color_transfer_lab_mean(source_img_path, target_img_path):
    """
    Color transfer using Reinhard's method
    
    @param source_img_path: path to source image
    @param target_img_path: path to target image
    @return: color transferred image
    """
    source = cv2.imread(source_img_path)
    target = cv2.imread(target_img_path)
    
    # convert images from RGB to L*a*b* color space
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # split the images into their respective color channels
    (lSource, aSource, bSource) = cv2.split(source)
    (lTarget, aTarget, bTarget) = cv2.split(target)

    # compute the mean and standard deviation of each color channel
    (lMeanSrc, lStdSrc) = (lSource.mean(), lSource.std())
    (lMeanTar, lStdTar) = (lTarget.mean(), lTarget.std())
    (aMeanSrc, aStdSrc) = (aSource.mean(), aSource.std())
    (aMeanTar, aStdTar) = (aTarget.mean(), aTarget.std())
    (bMeanSrc, bStdSrc) = (bSource.mean(), bSource.std())
    (bMeanTar, bStdTar) = (bTarget.mean(), bTarget.std())

    # subtract the means from the target image
    lTarget -= lMeanTar
    aTarget -= aMeanTar
    bTarget -= bMeanTar

    # Scale the target channels 
    lTarget = (lStdSrc / lStdTar) * lTarget
    aTarget = (aStdSrc / aStdTar) * aTarget
    bTarget = (bStdSrc / bStdTar) * bTarget

    # add the source means
    lTarget += lMeanSrc
    aTarget += aMeanSrc
    bTarget += bMeanSrc

    # clip the pixel intensities to [0, 255]
    lTarget = np.clip(lTarget, 0, 255)
    aTarget = np.clip(aTarget, 0, 255)
    bTarget = np.clip(bTarget, 0, 255)

    # merge the channels together and convert back to RGB color space
    output = cv2.merge([lTarget, aTarget, bTarget])
    output = cv2.cvtColor(output.astype("uint8"), cv2.COLOR_LAB2BGR)

    return output