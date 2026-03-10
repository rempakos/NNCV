import numpy as np
import torch
import albumentations as A


class CityscapeAlbumentations(torch.utils.data.Dataset):
    """
    This class applies transformations to the cityscape dataset such as crops, resizing, flips to both the 
    image and mask. In this way the mask /image duos are always synced.
    """
    def __init__(self, dataset, transform=None):
        """
        params: 
        dataset - the cityscape dataset
        transform -  the transformations to apply to the dataset
        """
        self.dataset = dataset
        self.transform = transform 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        
        #convert pil image to n.array for the albumentations lib
        image = np.array(image) # height,width,rgb
        mask = np.array(target) #0-255 values

        #if transform is required apply it to the image and mask
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = torch.from_numpy(image).permute(2, 0, 1) #rearrange 
        mask = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0).long()
                
        return image, mask