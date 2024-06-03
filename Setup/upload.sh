find dir -name "*.nii" > MCI_nii_gitig_files.txt
while read line; do aws s3 cp "$line" s3://thesis-ad/ori/; done < MCI_nii_gitig_files.txt

