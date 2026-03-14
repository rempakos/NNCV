import numpy as np
import torch
import albumentations as A
import config

def fast_fourier_transform(img1, img2, alpha=config.FOURIER_ALPHA):
    """
    Inspired by the paper "Fourier Domain Adaptation for Sematic Segmentation" this function aims to utilize
    the Fourier Transform not for domain adaptation but data augmentation in training instead
    params:
    img1 - the first image to transform
    img2 - the second image to transform
    alpha - factor for blending the low/high frequency components of the two images
    return:
    result - blended image
    """
    # Normalize img1 and img2 to range [0, 1]
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    # do fourier transform in height and width dimensions (spatial to frequency domain)
    fft1 = np.fft.fftn(img1, axes=(0, 1))
    fft2 = np.fft.fftn(img2, axes=(0, 1))

    # get the magnitude and the phase (color/structure)
    magnitude_img1 = np.abs(fft1)
    phase_img1 = np.angle(fft1)
    magnitude_img2 = np.abs(fft2)
    phase_img2 = np.angle(fft2)

    # magnitude blend with the given alpha factor

    blended_magnitude = alpha * magnitude_img1 + (1 - alpha) * magnitude_img2
    # using the belnded magnitude  but the same phase to preserve the structure of the original image
    fft_blended = blended_magnitude * np.exp(1j * phase_img1)

    #Finally do the inverse f.transform to reach the spatial domain again and normalize back to [0, 255]
    result = np.fft.ifftn(fft_blended, axes=(0, 1)).real
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    return result


class CityscapeAlbumentations(torch.utils.data.Dataset):
    """
    This class applies transformations to the cityscape dataset such as crops, resizing, flips to both the 
    image and mask. In this way the mask /image duos are always synced.
    """
    def __init__(self, dataset, transform=None, apply_fourier=False):
        """
        params: 
        dataset - the cityscape dataset
        transform -  the transformations to apply to the dataset
        apply_fourier - flag to indicate if Fourier augmentation should be applied(only in training and not validation)
        """
        self.dataset = dataset
        self.transform = transform
        self.apply_fourier = apply_fourier

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        
        #convert pil image to n.array for the albumentations lib
        image = np.array(image) # height,width,rgb
        mask = np.array(target) #0-255 values

        if self.apply_fourier and np.random.rand() < config.FOURIER_PROBABILITY: # apply fourier augmentation with configured probability
            #pick a random image from the dataset to blend with the currently processed image
            img2_index = np.random.randint(0, len(self.dataset))
            img2, _ = self.dataset[img2_index]
            img2 = np.array(img2)

            #swap the low/high frequency components and make the blended image using fast_fourier_transform()
            image = fast_fourier_transform(image, img2, alpha=0.3)

        #if transform is required apply it to the image and mask
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = torch.from_numpy(image).permute(2, 0, 1) #rearrange 
        mask = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0).long()
                
        return image, mask