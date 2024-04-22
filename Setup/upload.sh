find base_dir -name "*.nii" > nii_gitig_files.txt
while read line; do aws s3 cp "$line" s3://thesis-ad/ori/; done < nii_gitig_files.txt

