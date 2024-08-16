import os
import cv2
import numpy as np 


class ImageSlicer:
    def __init__(self, slice_size):
        self.slice_size = slice_size

    def slice_image(self, image, mask):
        img_height, img_width, _ = image.shape
        # Calculate padding needed to make the image dimensions a multiple of slice_size
        pad_height = (self.slice_size - img_height % self.slice_size) % self.slice_size
        pad_width = (self.slice_size - img_width % self.slice_size) % self.slice_size

        # Apply padding to both image and mask
        padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
        padded_mask = np.pad(mask, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

        slices = []
        mask_slices = []
        # Iterate over the padded image and mask to extract slices
        for i in range(0, padded_image.shape[0], self.slice_size):
            for j in range(0, padded_image.shape[1], self.slice_size):
                img_slice = padded_image[i:i + self.slice_size, j:j + self.slice_size]
                mask_slice = padded_mask[i:i + self.slice_size, j:j + self.slice_size]
                slices.append(img_slice)
                mask_slices.append(mask_slice)

        return slices, mask_slices

    def process_directories(self, original_dir, mask_dir, output_image_dir, output_mask_dir):
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)
        if not os.path.exists(output_mask_dir):
            os.makedirs(output_mask_dir)
        
        original_images = sorted(os.listdir(original_dir))
        mask_images = sorted(os.listdir(mask_dir))

        for img_file, mask_file in zip(original_images, mask_images):
            img_path = os.path.join(original_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            
            slices, mask_slices = self.slice_image(image, mask)
            
            for index, (slice_img, slice_mask) in enumerate(zip(slices, mask_slices)):
                slice_img_path = os.path.join(output_image_dir, f"{img_file[:-4]}_slice_{index}.png")
                slice_mask_path = os.path.join(output_mask_dir, f"{mask_file[:-4]}_slice_{index}.png")
                
                cv2.imwrite(slice_img_path, slice_img)
                cv2.imwrite(slice_mask_path, slice_mask)

if __name__ == "__main__":
    slice_size = 512  # Define the slice size as needed
    base_dir = "../Massachusetts Buildings Dataset/png"
    input_image_dir, input_mask_dir = os.path.join(base_dir, "val"), os.path.join(base_dir, "val_labels") 
    output_image_dir, output_mask_dir = os.path.join(base_dir, "sliced", "val"), os.path.join(base_dir, "sliced", "val_labels") 

    slicer = ImageSlicer(slice_size)
    slicer.process_directories(input_image_dir, input_mask_dir, output_image_dir, output_mask_dir)
