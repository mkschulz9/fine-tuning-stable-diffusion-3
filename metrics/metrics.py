import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from functools import partial
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlashFlowMatchEulerDiscreteScheduler
from peft import PeftModel
from huggingface_hub import notebook_login
import numpy as np
import pandas as pd

"""
Metrics class: Computes the CLIP, 
"""
class Metrics:
    """
    """
    def __init__(self, ground_truth_images, generated_images, prompts):
        self.ground_truth_images = ground_truth_images
        self.generated_images = generated_images
        self.prompts = prompts
        self.generated_images_stack = None
        self.ground_truth_images_stack = None
        self.resized_generated_images_stack_transform = None
    
    """
    """
    def create_generated_images_stack(self):
        generated_img_tensors = [torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in self.generated_images]
        self.generated_images_stack = torch.stack(generated_img_tensors).to(torch.uint8)
    
    """
    """
    def create_ground_truth_images_stack(self):
        ground_truth_img_tensors = [torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in self.ground_truth_images]
        self.ground_truth_images_stack = torch.stack(ground_truth_img_tensors).to(torch.uint8)
    
    """
    """
    def resize_generated_images_stack(self):
        if self.generated_images_stack == None:
            self.create_generated_images_stack()
        if self.ground_truth_images_stack == None:
            self.create_ground_truth_images_stack()
        resize_transform = transforms.Resize((self.ground_truth_images_stack.shape[2:4]))
        self.resized_generated_images_stack_transform = resize_transform(self.generated_images_stack)
    
    """
    """
    def compute_clip_score(self):
        clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
        self.clip_score = clip_score(torch.from_numpy(np.array(self.generated_images)).to(torch.uint8), self.prompts).item()
        print(f"CLIP Score: {self.clip_score}")
    
    """
    """
    def compute_inception_score(self):
        if self.generated_images_stack == None:
            self.create_generated_images_stack()
        inception_score = InceptionScore()
        inception_score.update(self.generated_images_stack)
        is_compute = inception_score.compute()
        self.inception_score_mean = is_compute[0].item()
        self.inception_score_stdev = is_compute[1].item()
        print(f"Inception Score (IS) - Mean: {self.inception_score_mean}")
        print(f"Inception Score (IS) - Standard Deviation: {self.inception_score_stdev}")
    
    """
    """
    def compute_fid_score(self):
        if self.generated_images_stack == None:
            self.create_generated_images_stack()
        if self.ground_truth_images_stack == None:
            self.create_ground_truth_images_stack()
        fid = FrechetInceptionDistance()
        fid.update(self.generated_images_stack, real=False)
        fid.update(self.ground_truth_images_stack, real=True)
        self.fid_score = fid.compute().item()
        print(f"FID Score: {self.fid_score}")
    
    """
    """
    def compute_kid_score(self):
        if self.generated_images_stack == None:
            self.create_generated_images_stack()
        if self.ground_truth_images_stack == None:
            self.create_ground_truth_images_stack()
        kid = KernelInceptionDistance(subset_size=2)
        kid.update(self.generated_images_stack, real=False)
        kid.update(self.ground_truth_images_stack, real=True)
        kid_compute = kid.compute()
        self.kid_score_mean = kid_compute[0].item()
        self.kid_score_stdev = kid_compute[1].item()
        print(f"Kernel Inception Distance (KID) Score - Mean: {self.kid_score_mean}")
        print(f"Kernel Inception Distance (KID) Score - Standard Deviation: {self.kid_score_stdev}")
    
    """
    """
    def compute_psnr(self):
        if self.resized_generated_images_stack_transform == None:
            self.resize_generated_images_stack()
        psnr = PeakSignalNoiseRatio()
        self.psnr_score = psnr(self.resized_generated_images_stack_transform, self.ground_truth_images_stack).item()
        print(f"Peak Signal to Noise Ratio (PSNR) Score: {self.psnr_score}")
    
    """

    """
    def compute_ssim(self, image_value_range = 255.0):
        if self.resized_generated_images_stack_transform == None:
            self.resize_generated_images_stack()
        ssim = StructuralSimilarityIndexMeasure(data_range = image_value_range)
        self.ssim_score = ssim(self.resized_generated_images_stack_transform.to(torch.float),
                  self.ground_truth_images_stack.to(torch.float)).item()
        print(f"Sturctural Similarity Index Measure (SSIM) Score: {self.ssim_score}")
    
    """
    compute_lpips: Method computes the Learned Perceptual Image Patch Similarity (LPIPS) metric
    for text-to-image generation. 

    Arguments:
        - self: Metrics class
        - network_type: Type of network architecture used for computing the LPIPS metric. Defaults to the
        "vgg" network.
    
    Return:
        - self.lpips_score: Field containing the LPIPS score computed
    """
    def compute_lpips(self, network_type = "vgg"):
        if self.resized_generated_images_stack_transform == None:
            self.resize_generated_images_stack()
        lpips = LearnedPerceptualImagePatchSimilarity(net_type = network_type, normalize = True)
        self.lpips_score = lpips(self.resized_generated_images_stack_transform.to(torch.float) / 255.0,
                            self.ground_truth_images_stack.to(torch.float) / 255.0).item()
        print(f"Learned Perceptual Image Patch Similarity (LPIPS) Score: {self.lpips_score}")
    
    """
    compute_all_metrics: Method computes 

    Arguments:
        - self: Metrics class
        - image_value_range:
        - network_type: 
        - output_csv:
    
    Returns:
    """
    def compute_all_metrics(self, image_value_range = 255.0, network_type = "vgg", output_csv = 'metrics.csv'):
        self.compute_clip_score()
        self.compute_inception_score()
        self.compute_fid_score()
        self.compute_kid_score()
        self.compute_psnr()
        self.compute_ssim(image_value_range)
        self.compute_lpips(network_type)
        metrics = ['CLIP', 'Inception Score - Mean', 'Inception Score - Standard Deviation', 'FID', 'KID - Mean', 'KID - Standard Deviation', 'PSNR', 'SSIM', 'LPIPS']
        metric_scores = [self.clip_score, self.inception_score_mean, self.inception_score_stdev, self.fid_score, 
        self.kid_score_mean, self.kid_score_stdev, self.psnr_score, self.ssim_score, self.lpips_score]
        metric_scores_df = pd.DataFrame({'Metric': metrics, 'Score': metric_scores})
        metric_scores_df.to_csv(output_csv, index=False)


