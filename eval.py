import numpy as np
import math
import cv2
import os
import glob

test_low_images = glob.glob("/home/shimizu/CV_project/dataset/test/low_4/*")
test_output_images = glob.glob("/home/shimizu/CV_project/dataset/test/572_4_epoch50/*")
test_high_images = glob.glob("/home/shimizu/CV_project//dataset/test/high/*")

sum_low_psnr = 0
sum_low_datasize = 0
sum_output_psnr = 0
sum_output_datasize = 0
sum_high_datasize = 0
for i in range(len(test_low_images)):
    low_img = cv2.imread(test_low_images[i])
    output_img = cv2.imread(test_output_images[i])
    high_img = cv2.imread(test_high_images[i])

    low_psnr = cv2.PSNR(low_img, high_img)
    output_psnr = cv2.PSNR(output_img, high_img)

    low_datasize = os.path.getsize(test_low_images[i])
    output_datasize = os.path.getsize(test_output_images[i])
    high_datasize = os.path.getsize(test_high_images[i])

    sum_low_psnr += round(low_psnr, 3)
    sum_output_psnr += round(output_psnr, 3)
    
    sum_low_datasize += round(low_datasize / 1024, 2)
    sum_output_datasize += round(output_datasize / 1024, 2)
    sum_high_datasize += round(high_datasize / 1024, 2)

print("mean low PSNR: ", round(sum_low_psnr / len(test_low_images), 3))
print("mean output PSNR: ", round(sum_output_psnr / len(test_output_images), 3))
print("mean low Data size: ", round(sum_low_datasize / len(test_low_images), 2))
print("mean output Data size: ", round(sum_output_datasize / len(test_output_images), 2))
print("mean high Data size: ", round(sum_high_datasize / len(test_high_images), 2))
