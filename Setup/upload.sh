find C:/Users/Trisss/Desktop/PjA/Data/SAD_data/Mild -name "*.nii" > nii_Mild_gitig_files.txt
while read line; do aws s3 cp "$line" s3://thesis-ad/ori/; done < nii_Mild_gitig_files.txt

