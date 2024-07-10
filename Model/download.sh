filenames=$(grep -oE '[A-Za-z0-9_]+\.nii\.gz' test_files_.json)

for filename in $filenames; do
    echo "Processing: $filename"
    aws s3 cp "s3://thesis-ad/fsl/$filename" ""
done
