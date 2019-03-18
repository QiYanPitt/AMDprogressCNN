# AMDprogressCNN
# Use fundus image to classify whether the late AMD progression time is before or after 3 years
python code_img2amdState2time.py --cutoff_yr 3 --input_image 3980_20_LE_F2_LS.jpg --out_file output.txt --weights "/directory/weights"

# To generate saliency map to visualize what part of the image is focused by CNN to make the classification decision
python code_img2amdState2time_saliency.py --cutoff_yr 3 --input_image 3980_20_LE_F2_LS.jpg --out_image output.png --weights "/directory/weights"

# Use fundus image and 52 reported SNPs to predict whether the late AMD progression time is before or after 3 years
python code_imgNgeno2amdState2time.py --cutoff_yr 3 --input_image 3980_20_LE_F2_LS.jpg --input_geno input.geno --out_file output.txt --weights "/directory/weights"