aws ecr get-login-password --region your_region| docker login --username AWS --password-stdin your_account_id.dkr.ecr.ap-southeast-1.amazonaws.com
aws ecr create-repository --repository-name some_repo_name --region your_region --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE
docker tag name:tag your_account_id.dkr.ecr.ap-southeast-1.amazonaws.com/some_repo_name:latest
docker push your_account_id.dkr.ecr.ap-southeast-1.amazonaws.com/some_repo_name:latest
