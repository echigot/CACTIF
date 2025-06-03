import numpy as np
import torch
from scipy.ndimage import binary_erosion
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)


def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def adain_pixel(content_feat, content_mu, content_sigma, style_mu, style_sigma):
    size = content_feat.size()
    normalized_feat = (content_feat - content_mu.expand(size)) / content_sigma.expand(size)
    return normalized_feat * style_sigma.expand(size) + style_mu.expand(size)


def process_mask(feats, binary_mask, nb_class):
    binary_mask = torch.from_numpy(binary_mask).cpu()
    # extract one class from segmentation map
    processed_mask = np.where(binary_mask == nb_class, 1, 0)
    
    processed_mask = binary_erosion(processed_mask)
    # resize mask to match the features
    processed_mask = processed_mask.astype(np.float32)
    
    height = feats.shape[1]
    width = feats.shape[2]
    processed_mask = np.array(Image.fromarray(processed_mask).resize((width, height)))

    processed_mask = torch.from_numpy(processed_mask).cuda()

    return feats, processed_mask


def custom_adain_pixel(content_feat, style_feat, content_label = None, style_label = None):
    torch.manual_seed(42)
    np.random.seed(42)
    
    style_mu = torch.zeros((4, *style_label.shape), dtype=torch.float32).cuda()
    style_sigma = torch.ones((4, *style_label.shape), dtype=torch.float32).cuda()
    
    content_mu = torch.zeros((4, *content_label.shape), dtype=torch.float32).cuda()
    content_sigma = torch.ones((4, *content_label.shape), dtype=torch.float32).cuda()
    
    # 19 classes of cityscapes
    for c in range(19):
        content_feat, content_mask = process_mask(content_feat, content_label, c)
        content_mean, content_std = calc_mean_std_smooth(content_feat, mask=content_mask)
        
        mu, sigma = None, None
        
        mask = content_label == c
        mask = np.repeat(mask[np.newaxis,:,:], 4, axis=0)
        
        # check if content has meaningful statistics about the current class
        if not (torch.isnan(content_mean).any() or torch.isnan(content_std).any()):
            style_feat, style_mask = process_mask(style_feat, style_label, c)
            if style_mask.max() > 0. :
                mu, sigma = calc_mean_std_smooth(style_feat, mask=style_mask)
                
            if mu is not None and sigma is not None:
                style_mu = style_mu + mu * content_mask
                style_sigma = style_sigma + sigma * content_mask
                
                content_mu = content_mu + content_mean * content_mask
                content_sigma = content_sigma + content_std * content_mask
    
    content_feat = adain_pixel(content_feat, content_mu, content_sigma, style_mu, style_sigma)
    
    # content feat is squeezed during the preprocessing 
    content_feat = content_feat.unsqueeze(0)
    return content_feat

def calc_mean_std_smooth(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 3)
    C = size[0]
    
    mask = mask.view(-1)  # Flatten the mask
    feat_flat = feat.view(C, -1)  # Flatten the feature tensor

    # Compute weighted mean
    weighted_sum = (feat_flat * mask).sum(dim=1)
    sum_of_weights = mask.sum()
    weighted_mean = (weighted_sum / sum_of_weights).view(C, 1, 1)

    # Compute weighted variance
    squared_diff = (feat_flat - weighted_mean.view(C, -1)) ** 2
    weighted_variance = (squared_diff * mask).sum(dim=1) / sum_of_weights
    weighted_variance += eps  # Add epsilon for numerical stability
    weighted_std = weighted_variance.sqrt().view(C, 1, 1)

    return weighted_mean, weighted_std

def calc_mean_std(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    if len(size) == 2:
        return calc_mean_std_2d(feat, eps, mask)
    
    assert (len(size) == 3)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)

    return feat_mean, feat_std


def calc_mean_std_2d(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 2)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1)

    return feat_mean, feat_std
