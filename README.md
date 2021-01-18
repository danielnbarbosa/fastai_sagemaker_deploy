Deploys a fastai model to a sagemaker endpoint using torchserve.  This notebook can be run on a CPU based Sagemaker notebook instance.

This is based on other guides on the internet that use [pytorch 1.0](https://course19.fast.ai/deployment_amzn_sagemaker.html) and [pytorch 1.4](https://github.com/mattmcclean/fastai2-sagemaker-deployment-demo/blob/master/fastai2_deploy_sagemaker_demo.ipynb).  This guide uses the newer deployment mechanism of torchserve which is only available in pytorch >= 1.6.

Feel free to use this as a template for deploying your own models.  I suffered through a lot of issues getting this working so hopefully I can save you some of the pain.
