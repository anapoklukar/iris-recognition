import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import math
from math import pi
from torchvision import models
from modules.network import *

class irisRecognition(object):
    def __init__(self, cfg):
        # cParsing the config file
        self.cuda = cfg["cuda"]
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.polar_height = cfg["polar_height"]
        self.polar_width = cfg["polar_width"]
        self.filter_sizes = [int(filter_size) for filter_size in cfg["recog_filter_size"].split(',')]
        self.num_filters_per_size = [int(num_filter) for num_filter in cfg["recog_num_filters"].split(',')]
        self.total_num_filters = sum(self.num_filters_per_size)
        self.torch_filters = self.load_filters(cfg["recog_bsif_dir"], self.filter_sizes, self.num_filters_per_size)
        self.max_shift = cfg["recog_max_shift"]
        self.score_norm = cfg["score_norm"]
        self.threshold_frac_avg_bits = cfg["threshold_frac_avg_bits"]
        
        self.mask_model_path = cfg["mask_model_path"]
        self.circle_model_path = cfg["circle_model_path"]
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,5))
        
        
        self.NET_INPUT_SIZE = (320,240)

        with torch.inference_mode():
            self.circle_model = models.resnet18()
            self.circle_model.avgpool = conv(in_channels=512, out_n=6)
            self.circle_model.fc = fclayer(out_n=6)
            try:
                self.circle_model.load_state_dict(torch.load(self.circle_model_path, map_location=self.device))
            except AssertionError:
                    print("assertion error")
                    self.circle_model.load_state_dict(torch.load(self.circle_model_path,
                        map_location = lambda storage, loc: storage))
            self.circle_model = self.circle_model.to(self.device)
            self.circle_model.eval()
            self.mask_model = NestedSharedAtrousResUNet(1, 1, width=32, resolution=(240,320))
            try:
                self.mask_model.load_state_dict(torch.load(self.mask_model_path, map_location=self.device))
            except AssertionError:
                    print("assertion error")
                    self.mask_model.load_state_dict(torch.load(self.mask_model_path,
                        map_location = lambda storage, loc: storage))
            self.mask_model = self.mask_model.to(self.device)
            self.mask_model.eval()
            self.input_transform_mask = Compose([
                ToTensor(),
                Normalize(mean=(0.5,), std=(0.5,))
            ])
            self.input_transform_circ = Compose([
                ToTensor(),
                Normalize(mean=(0.5,), std=(0.5,))
            ])          
        
        self.avg_bits_by_filter_size = {5: 25056, 7: 24463, 9: 23764, 11: 23010, 13: 22225, 15: 21420, 17: 20603, 19: 19777, 21: 18945, 27: 16419, 33: 13864, 39: 11289}
        self.avg_num_bits = 0
        for filter_size in self.filter_sizes:
            self.avg_num_bits += self.avg_bits_by_filter_size[filter_size]
        self.avg_num_bits /= len(self.filter_sizes)
        self.fixed_num_bits = 26770 # based on masks statistics from 'VII-Q', 'ND3DIris', 'PostMortem-Iris-NIJ', and 'Q-FIRE'
        self.ISO_RES = (640,480)

    @torch.inference_mode()
    def load_filters(self, recog_bsif_dir, filter_sizes, num_filters_per_size):
        torch_filters = []
        for filter_size, num_filters in zip(filter_sizes, num_filters_per_size):
            mat_file_path = recog_bsif_dir+'ICAtextureFilters_{0}x{1}_{2}bit.pt'.format(filter_size, filter_size, num_filters)
            filter_mat = torch.jit.load(mat_file_path, torch.device('cpu')).ICAtextureFilters.detach().numpy()
            torch_filter = torch.FloatTensor(filter_mat).to(self.device)
            torch_filter = torch.moveaxis(torch_filter.unsqueeze(0), 3, 0).detach().requires_grad_(False)
            torch_filters.append(torch_filter.clone().detach())
        return torch_filters

    # converts non-ISO images into ISO dimensions
    def fix_image(self, image):
        w, h = image.size
        aspect_ratio = float(w)/float(h)
        if aspect_ratio >= 1.333 and aspect_ratio <= 1.334:
            result_im = image.copy().resize(self.ISO_RES)
        elif aspect_ratio < 1.333:
            w_new = h * (4.0/3.0)
            w_pad = (w_new - w) / 2
            result_im = Image.new(image.mode, (int(w_new), h), 127)
            result_im.paste(image, (int(w_pad), 0))
            result_im = result_im.resize(self.ISO_RES)
        else:
            h_new = w * (3.0/4.0)
            h_pad = (h_new - h) / 2
            result_im = Image.new(image.mode, (w, int(h_new)), 127)
            result_im.paste(image, (0, int(h_pad)))
            result_im = result_im.resize(self.ISO_RES)
        return result_im
    
    def segment_and_circApprox(self, image):
        pred = self.segment(image)
        pupil_xyr, iris_xyr = self.circApprox(image)
        return pred, pupil_xyr, iris_xyr

    @torch.inference_mode()
    def segment(self,image):

        w,h = image.size
        image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
        mask_logit_t = self.mask_model(Variable(self.input_transform_mask(image).unsqueeze(0).to(self.device)))[0]
        mask_t = torch.where(torch.sigmoid(mask_logit_t) > 0.5, 255, 0)
        mask = mask_t.cpu().numpy()[0]
        mask = cv2.resize(np.uint8(mask), (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
        #print('Mask Shape: ', mask.shape)

        return mask

    def segmentVis(self,im,mask,pupil_xyr,iris_xyr):
        
        pupil_xyr = np.around(pupil_xyr).astype(np.int32)
        iris_xyr = np.around(iris_xyr).astype(np.int32)
        imVis = np.stack((np.array(im),)*3, axis=-1)
        imVis[:,:,1] = np.clip(imVis[:,:,1] + (96/255)*mask,0,255)
        imVis = cv2.circle(imVis, (pupil_xyr[0],pupil_xyr[1]), pupil_xyr[2], (0, 0, 255), 2)
        imVis = cv2.circle(imVis, (iris_xyr[0],iris_xyr[1]), iris_xyr[2], (255, 0, 0), 2)

        return imVis

    @torch.inference_mode()
    def circApprox(self,image):

        w,h = image.size

        image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
        inp_xyr_t = self.circle_model(Variable(self.input_transform_circ(image).unsqueeze(0).repeat(1,3,1,1).to(self.device)))

        #Circle params
        diag = math.sqrt(w**2 + h**2)
        inp_xyr = inp_xyr_t.tolist()[0]
        pupil_x = (inp_xyr[0] * w)
        pupil_y = (inp_xyr[1] * h)
        pupil_r = (inp_xyr[2] * 0.5 * 0.8 * diag)
        iris_x = (inp_xyr[3] * w)
        iris_y = (inp_xyr[4] * h)
        iris_r = (inp_xyr[5] * 0.5 * diag)

        return np.array([pupil_x,pupil_y,pupil_r]), np.array([iris_x,iris_y,iris_r])
    
    # New Rubbersheet model-based Cartesian-to-polar transformation. It has support for bilinear interpolation
    @torch.inference_mode()
    def grid_sample(self, input, grid, interp_mode):  #helper function for new interpolation
        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode, align_corners=True, padding_mode='border')

    @torch.inference_mode()
    def cartToPol_torch(self, image, mask, pupil_xyr, iris_xyr, interpolation='bilinear'): # 
        if pupil_xyr is None or iris_xyr is None:
            return None, None
        
        image = torch.tensor(np.array(image)).float().unsqueeze(0).unsqueeze(0).to(self.device)
        mask = torch.tensor(np.array(mask)).float().unsqueeze(0).unsqueeze(0).to(self.device)
        width = image.shape[3]
        height = image.shape[2]

        polar_height = self.polar_height
        polar_width = self.polar_width
        pupil_xyr = torch.tensor(pupil_xyr).unsqueeze(0).float().to(self.device)
        iris_xyr = torch.tensor(iris_xyr).unsqueeze(0).float().to(self.device)
        
        theta = (2*pi*torch.linspace(0,polar_width-1,polar_width)/polar_width).to(self.device)
        pxCirclePoints = (pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)).to(self.device) #b x 512
        pyCirclePoints = (pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)).to(self.device)  #b x 512
        
        ixCirclePoints = (iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)).to(self.device)  #b x 512
        iyCirclePoints = (iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)).to(self.device) #b x 512

        radius = (torch.linspace(1,polar_height,polar_height)/polar_height).reshape(-1, 1).to(self.device)  #64 x 1
        
        pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        
        ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512

        x = (pxCoords + ixCoords).float()
        x_norm = ((x-1)/(width-1))*2 - 1 #b x 64 x 512

        y = (pyCoords + iyCoords).float()
        y_norm = ((y-1)/(height-1))*2 - 1  #b x 64 x 512

        grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1).to(self.device)

        image_polar = self.grid_sample(image, grid_sample_mat, interp_mode=interpolation)
        image_polar = torch.clamp(torch.round(image_polar), min=0, max=255)
        mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest') # always use nearest neighbor interpolation for mask
        mask_polar = (mask_polar>0.5).long() * 255

        return (image_polar[0][0].cpu().numpy()).astype(np.uint8), mask_polar[0][0].cpu().numpy().astype(np.uint8)

    # (Fixed) Old implementation of Rubbersheet model-based Cartesian-to-polar transformation that uses nearest neighbor interpolation
    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        
        if pupil_xyr is None:
            return None, None
       
        image = np.array(image)
        height, width = image.shape
        mask = np.array(mask)

        image_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)
        mask_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)

        theta = 2*pi*np.linspace(0,self.polar_width-1,self.polar_width)/self.polar_width
       
        pxCirclePoints = np.around(pupil_xyr[0] + pupil_xyr[2]*np.cos(theta))    
        ixCirclePoints = np.around(iris_xyr[0] + iris_xyr[2]*np.cos(theta))
        pyCirclePoints = np.around(pupil_xyr[1] + pupil_xyr[2]*np.sin(theta))
        iyCirclePoints = np.around(iris_xyr[1] + iris_xyr[2]*np.sin(theta))

        for j in range(1, self.polar_width+1):            
            for i in range(1, self.polar_height+1):

                radius = i/self.polar_height
                x = int(np.around((1-radius) * pxCirclePoints[j-1] + radius * ixCirclePoints[j-1]))
                y = int(np.around((1-radius) * pyCirclePoints[j-1] + radius * iyCirclePoints[j-1]))
                if (x > 0 and x <= width and y > 0 and y <= height): 
                    image_polar[i-1][j-1] = image[y-1][x-1]
                    mask_polar[i-1][j-1] = mask[y-1][x-1]

        return image_polar, mask_polar

    @torch.inference_mode()
    def extractCode(self, polar):
        if polar is None:
            return None
        codeBinaries = []
        for filter_size, torch_filter in zip(self.filter_sizes, self.torch_filters):
            r = int(np.floor(filter_size / 2))
            polar_t = torch.tensor(polar).float().unsqueeze(0).unsqueeze(0).to(self.device)
            padded_polar = nn.functional.pad(polar_t, (r, r, 0, 0), mode='circular')
            codeContinuous = nn.functional.conv2d(padded_polar, torch_filter)
            codeBinary = torch.where(codeContinuous.squeeze(0) > 0, True, False)
            codeBinaries.append(codeBinary.cpu().numpy())
        return codeBinaries
    
    """
    This module processes polar-transformed images to extract and match binary codes based on local intensity comparisons around circular patterns.

    Implemented features:
    1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances local contrast in the input polar image, helping improve feature extraction.
    2. **Dual Circle Code Extraction**:
        - Extracts binary codes around two concentric circles, defined by parameters:
            - Circle 1: 8 points (P=8) on a 1-pixel radius (R=1).
            - Circle 2: 16 points (P=16) on a 2-pixel radius (R=2).
    3. **Bilinear Interpolation**: Used to accurately sample pixel values along the circles, even if points don't align with pixel centers.
    4. **Rotation Invariance**: Adjusts each binary code to its lexicographically smallest rotation, ensuring the code is consistent regardless of orientation.
    5. **Local Variance Calculation**:
        - Computes the variance of the local pixel values for Circle 1 as an additional feature, which is then normalized.
    6. **Matching and Scoring**:
        - Uses masks to compare only valid bits in the binary codes.
        - Computes Hamming distances for both circles separately and combines them with equal weighting (0.5 each).
        - Calculates variance difference between the two code sets.
        - Computes a final similarity score, using 0.8 weight for the combined Hamming distance and 0.2 weight for the variance similarity.
    """
    
    @torch.inference_mode()
    def extractIBBCode(self, polar): #, mask):
        # Applying CLAHE to the polar image
        def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
            # Create CLAHE object
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            
            # Apply CLAHE to the image
            clahe_image = clahe.apply(image)
            return clahe_image
        
        polar = apply_clahe(polar)
        
        P = 8 # Point on circle, where the code is extracted
        R = 1 # Radius of the circle
        
        P2 = 16 # Point on second circle, where the code is extracted
        R2 = 2 # Radius of the second circle
        
        code = []
        
        # Shape of the polar image
        h,w = polar.shape
        
        # Iterating over the polar image
        for x in range(h):
            for y in range(w):
                
                # Center of the circle
                g0 = polar[x,y]
                
                # Array to store the code
                pixelcode = []
                local_values = []
                pixelcode2 = []
                local_values2 = []
                
                # Computing the coordinates of the points on the circle
                for p in range(P):
                    xp = x + R * math.cos(2*pi*p/P)
                    yp = y + R * math.sin(2*pi*p/P)
                    
                    # Checking if the point is inside the image
                    if xp >= 0 and xp < h and yp >= 0 and yp < w:
                        
                        # Interpolate the value at (xp, yp) using bilinear interpolation
                        x0, y0 = int(xp), int(yp)  # Top-left corner
                        x1, y1 = min(x0 + 1, h - 1), min(y0 + 1, w - 1)  # Bottom-right corner

                        # Interpolation weights
                        wx, wy = xp - x0, yp - y0
                        w00 = (1 - wx) * (1 - wy)
                        w01 = (1 - wx) * wy
                        w10 = wx * (1 - wy)
                        w11 = wx * wy

                        # Interpolated value at (xp, yp)
                        gp = (polar[x0, y0] * w00 +
                            polar[x0, y1] * w01 +
                            polar[x1, y0] * w10 +
                            polar[x1, y1] * w11)
                    
                       # Determine the binary code bit for this point
                        pixelcode.append(1 if gp > g0 else 0)
                        local_values.append(gp)
                    else:
                        # If the point is outside the image, append 0 by default
                        pixelcode.append(0)
                        local_values.append(0)
                        
                # Computing the coordinates of the points on the  second circle
                for p in range(P2):
                    xp = x + R2 * math.cos(2*pi*p/P2)
                    yp = y + R2 * math.sin(2*pi*p/P2)
                    
                    # Checking if the point is inside the image
                    if xp >= 0 and xp < h and yp >= 0 and yp < w:
                        
                        # Interpolate the value at (xp, yp) using bilinear interpolation
                        x0, y0 = int(xp), int(yp)  # Top-left corner
                        x1, y1 = min(x0 + 1, h - 1), min(y0 + 1, w - 1)  # Bottom-right corner

                        # Interpolation weights
                        wx, wy = xp - x0, yp - y0
                        w00 = (1 - wx) * (1 - wy)
                        w01 = (1 - wx) * wy
                        w10 = wx * (1 - wy)
                        w11 = wx * wy

                        # Interpolated value at (xp, yp)
                        gp = (polar[x0, y0] * w00 +
                            polar[x0, y1] * w01 +
                            polar[x1, y0] * w10 +
                            polar[x1, y1] * w11)
                    
                       # Determine the binary code bit for this point
                        pixelcode2.append(1 if gp > g0 else 0)
                        local_values2.append(gp)
                    else:
                        # If the point is outside the image, append 0 by default
                        pixelcode2.append(0)
                        local_values2.append(0)
                
                # Make the pixel code rotation invariant by shifting it to the smallest lexicographic representation
                def min_binary_rotation(arr):
                    # Convert array to a binary string
                    binary_str = ''.join(map(str, arr))
                    # Concatenate the string with itself to capture all rotations as substrings
                    doubled_str = binary_str + binary_str
                    # Find the lexicographically smallest substring of length n
                    n = len(binary_str)
                    min_rotation = min(doubled_str[i:i+n] for i in range(n))
                    # Convert the smallest binary string back to a list of ints
                    return [int(digit) for digit in min_rotation]
                
                min_code = min_binary_rotation(pixelcode)
                min_code2 = min_binary_rotation(pixelcode2)
                
                # Calculate the rotation-invariant variance
                variance = np.var(local_values) if local_values else 0
                normalized_variance = variance / (np.mean(local_values) + 1e-6)  # Avoid division by zero
                
                # Append binary code and normalized variance as a dictionary
                code.append({
                    "code": min_code,
                    "code2": min_code2,
                    "variance": normalized_variance
                })           
                
        return code
                

    @torch.inference_mode()
    def matchIBBCodes(self, code1, code2, mask1, mask2):
        # Separate binary codes and variance values
        var1 = [entry["variance"] for entry in code1]
        var2 = [entry["variance"] for entry in code2]
        code1_second = [entry["code2"] for entry in code1]
        code2_second = [entry["code2"] for entry in code2]
        code1 = [entry["code"] for entry in code1]
        code2 = [entry["code"] for entry in code2]
        
        # Flatten and repeat for the second code
        code1_second, code2_second = np.array(code1_second).flatten(), np.array(code2_second).flatten()
        mask1_second, mask2_second = np.repeat(mask1, 16), np.repeat(mask2, 16)
        
        # Flatten and repeat masks to match 8-bit structure
        code1, code2 = np.array(code1).flatten(), np.array(code2).flatten()
        mask1, mask2 = np.repeat(mask1, 8), np.repeat(mask2, 8)

        # Apply combined mask and calculate Hamming distance
        combined_mask_flat = np.bitwise_and(mask1, mask2)
        code = np.bitwise_xor(code1, code2)
        valid_code = code[combined_mask_flat != 0]
        hamm_dist_first = np.sum(valid_code) / len(valid_code.flatten()) 
        
        # Hamming distance for the second code
        combined_mask_flat_second = np.bitwise_and(mask1_second, mask2_second)
        code_second = np.bitwise_xor(code1_second, code2_second)
        valid_code_second = code_second[combined_mask_flat_second != 0]
        hamm_dist_second = np.sum(valid_code_second) / len(valid_code_second.flatten())
        
        # Combine Hamming distances
        hamm_dist = 0.5 * hamm_dist_first + 0.5 * hamm_dist_second
        
        # Variance difference
        variance_diff = np.abs(np.array(var1) - np.array(var2))
        variance_similarity = np.mean(variance_diff)

        # Combine binary Hamming distance and variance similarity
        total_similarity = 0.8 * hamm_dist + 0.2 * variance_similarity
        return total_similarity


    @torch.inference_mode()
    def matchCodes(self, codes1, codes2, mask1, mask2):
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        scoreC = []
        for xshift in range(-self.max_shift, self.max_shift+1):
            sumXorCodesMasked = 0
            sumBitsCompared = 0
            total_num_filters = 0
            for code1, code2 in zip(codes1, codes2):
                assert code1.shape == code2.shape # num_filter x filter_size x filter_size
                num_filters, code_size, _ = code1.shape
                r = int((mask1.shape[0] - code_size) / 2)
                mask1_binary = np.where(mask1[r:-r, :] > 127.5, True, False)
                mask2_binary = np.where(mask2[r:-r, :] > 127.5, True, False)
                andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
                if np.sum(andMasks) != 0:
                    xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                    xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (num_filters, 1, 1)))
                    sumXorCodesMasked += np.sum(xorCodesMasked)
                    sumBitsCompared += (np.sum(andMasks) * num_filters)
                    total_num_filters += num_filters
            if sumBitsCompared == 0:
                scoreC.append(float('inf'))
            else:
                scoreC.append(sumXorCodesMasked / sumBitsCompared)
                if self.score_norm:
                    scoreC[-1] = 0.5 - (0.5 - scoreC[-1]) * math.sqrt( sumBitsCompared / (self.avg_num_bits * total_num_filters) )
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_best = scoreC[scoreC_index]
        if scoreC_best == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0
                
        return scoreC_best, scoreC_index - self.max_shift
    

    @torch.inference_mode()
    def matchCodesEfficient(self, codes1, codes2, mask1, mask2):
        if (np.sum(mask1) <= self.threshold_frac_avg_bits * self.avg_num_bits * 255) or (np.sum(mask2) <= self.threshold_frac_avg_bits * self.avg_num_bits * 255):
            #print("Too small masks")
            return float('inf'), -1.0
        scoreC = []
        for xshift in range(-self.max_shift, self.max_shift+1, 2):
            sumXorCodesMasked = 0
            sumBitsCompared = 0
            total_num_filters = 0
            for code1, code2 in zip(codes1, codes2):
                assert code1.shape == code2.shape # num_filter x filter_size x filter_size
                num_filters, code_size, _ = code1.shape
                r = int((mask1.shape[0] - code_size) / 2)
                # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
                mask1_binary = np.where(mask1[r:-r, :] > 127.5, True, False)
                mask2_binary = np.where(mask2[r:-r, :] > 127.5, True, False)
                andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
                if np.sum(andMasks) != 0:
                    xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                    xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (num_filters, 1, 1)))
                    sumXorCodesMasked += np.sum(xorCodesMasked)
                    sumBitsCompared += (np.sum(andMasks) * num_filters)
                    total_num_filters += num_filters
            if sumBitsCompared == 0:
                scoreC.append(float('inf'))
            else:
                scoreC.append(sumXorCodesMasked / sumBitsCompared)
                if self.score_norm:
                    scoreC[-1] = 0.5 - (0.5 - scoreC[-1]) * math.sqrt( sumBitsCompared / (self.avg_num_bits * total_num_filters) )
                    
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_shift = scoreC_index * 2 - self.max_shift
        scoreC = scoreC[scoreC_index]
        if scoreC == float('inf'):
            print("Too small overlap between masks")
            return float('inf'), -1.0
        
        sumXorCodesMasked_left = 0
        sumBitsCompared_left = 0
        total_num_filters_left = 0
        for code1, code2 in zip(codes1, codes2):
            num_filters, code_size, _ = code1.shape
            r = int((mask1.shape[0] - code_size) / 2)
            mask1_binary = np.where(mask1[r:-r, :] > 127.5, True, False)
            mask2_binary = np.where(mask2[r:-r, :] > 127.5, True, False)
            andMasks_left = np.logical_and(mask1_binary, np.roll(mask2_binary, scoreC_shift-1, axis=1))
            if np.sum(andMasks_left) != 0:
                xorCodes = np.logical_xor(code1, np.roll(code2, scoreC_shift-1, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks_left,axis=0), (num_filters, 1, 1)))
                sumXorCodesMasked_left += np.sum(xorCodesMasked)
                sumBitsCompared_left += (np.sum(andMasks_left) * num_filters)
                total_num_filters_left += num_filters

        if sumBitsCompared_left == 0:
            scoreC_left = float('inf')
        else:
            scoreC_left = sumXorCodesMasked_left / sumBitsCompared_left
            if self.score_norm:
                scoreC_left = 0.5 - (0.5 - scoreC_left) * math.sqrt( sumBitsCompared_left / (self.avg_num_bits * total_num_filters_left) )
        
        sumXorCodesMasked_right = 0
        sumBitsCompared_right = 0
        total_num_filters_right = 0
        for code1, code2 in zip(codes1, codes2):
            num_filters, code_size, _ = code1.shape
            r = int((mask1.shape[0] - code_size) / 2)
            mask1_binary = np.where(mask1[r:-r, :] > 127.5, True, False)
            mask2_binary = np.where(mask2[r:-r, :] > 127.5, True, False)
            andMasks_right = np.logical_and(mask1_binary, np.roll(mask2_binary, scoreC_shift+1, axis=1))
            if np.sum(andMasks_right) != 0:
                xorCodes = np.logical_xor(code1, np.roll(code2, scoreC_shift+1, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks_right,axis=0), (num_filters, 1, 1)))
                sumXorCodesMasked_right += np.sum(xorCodesMasked)
                sumBitsCompared_right += (np.sum(andMasks_right) * num_filters)
                total_num_filters_right += num_filters

        if sumBitsCompared_right == 0:
            scoreC_right = float('inf')
        else:
            scoreC_right = sumXorCodesMasked_right / sumBitsCompared_right
            if self.score_norm:
                scoreC_right = 0.5 - (0.5 - scoreC_right) * math.sqrt( sumBitsCompared_right / (self.avg_num_bits * total_num_filters_right) )
        
        scoreC_best = min(scoreC, scoreC_left, scoreC_right)
        if scoreC_best == scoreC_left:
            scoreC_shift -= 1
        elif scoreC_best == scoreC_right:
            scoreC_shift += 1
        
        return scoreC_best, scoreC_shift






##########################################################
### Adam's extra functions -- no need to correct those ###
##########################################################

    @torch.inference_mode()
    def matchCodesEfficientRaw(self, code1, code2, mask1, mask2):
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1 > 127.5, True, False)
        mask2_binary = np.where(mask2 > 127.5, True, False)
        if (np.sum(mask1_binary) <= self.threshold_frac_avg_bits * self.avg_num_bits) or (np.sum(mask2_binary) <= self.threshold_frac_avg_bits * self.avg_num_bits):
            print("Too small masks")
            return -1.0, -1.0
        scoreC = []
        for xshift in range(-self.max_shift, self.max_shift+1, 2):
            andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
            if np.sum(andMasks) == 0:
                scoreC.append(float('inf'))
            else:
                xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.total_num_filters, 1, 1)))
                scoreC.append(np.sum(xorCodesMasked) / (np.sum(andMasks) * self.total_num_filters))
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_shift = scoreC_index * 2 - self.max_shift
        scoreC = scoreC[scoreC_index]
        if scoreC == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0
        
        andMasks_left = np.logical_and(mask1_binary, np.roll(mask2_binary, scoreC_shift-1, axis=1))
        if np.sum(andMasks_left) == 0:
            scoreC_left = float('inf')
        else:
            xorCodes_left = np.logical_xor(code1, np.roll(code2, scoreC_shift-1, axis=2))
            xorCodesMasked_left = np.logical_and(xorCodes_left, np.tile(np.expand_dims(andMasks_left,axis=0), (self.total_num_filters, 1, 1)))
            scoreC_left = np.sum(xorCodesMasked_left) / (np.sum(andMasks_left) * self.total_num_filters)
        
        andMasks_right = np.logical_and(mask1_binary, np.roll(mask2_binary, scoreC_shift+1, axis=1))
        if np.sum(andMasks_right) == 0:
            scoreC_right = float('inf')
        else:
            xorCodes_right = np.logical_xor(code1, np.roll(code2, scoreC_shift+1, axis=2))
            xorCodesMasked_right = np.logical_and(xorCodes_right, np.tile(np.expand_dims(andMasks_right,axis=0), (self.total_num_filters, 1, 1)))
            scoreC_right = np.sum(xorCodesMasked_right) / (np.sum(andMasks_right) * self.total_num_filters)
        
        scoreC_best = min(scoreC, scoreC_left, scoreC_right)
        if scoreC_best == scoreC_left:
            scoreC_shift -= 1
        elif scoreC_best == scoreC_right:
            scoreC_shift += 1
        
        return scoreC_best, scoreC_shift
    
    @torch.inference_mode()
    def matchCodesRaw(self, code1, code2, mask1, mask2):
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1 > 127.5, True, False)
        mask2_binary = np.where(mask2 > 127.5, True, False)
        scoreC = []
        for xshift in range(-self.max_shift, self.max_shift+1):
            andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
            if np.sum(andMasks) == 0:
                scoreC.append(float('inf'))
            else:
                xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.total_num_filters, 1, 1)))
                scoreC.append(np.sum(xorCodesMasked) / (np.sum(andMasks) * self.total_num_filters))
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_best = scoreC[scoreC_index]
        if scoreC_best == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0
                
        return scoreC_best, scoreC_index - self.max_shift
    
    @torch.inference_mode()
    def matchCodesKernelSubset(self, code1, code2, mask1, mask2, kernel_selection):
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1 > 127.5, True, False)
        mask2_binary = np.where(mask2 > 127.5, True, False)
        scoreC = []

        for xshift in range(-self.max_shift, self.max_shift+1):
            andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
            if np.sum(andMasks) == 0:
                scoreC.append(float('inf'))
            else:
                xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.total_num_filters, 1, 1)))

                _scoreC = []
                for s in kernel_selection:
                    _scoreC.append(np.sum(xorCodesMasked[s,:,:]) / (np.sum(andMasks)))
                
                scoreC.append(np.mean(np.array(_scoreC)))
                    
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_best = scoreC[scoreC_index]

        if scoreC_best == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0
                
            # DEBUG: 
            # full_path_code = f'temp_images/c-{c:02d}-{xshift+self.max_shift:02d}-{np.sum(xorCodeMasked_MorphOpened/np.sum(andMasks)):0.2f}.png'
            # Image.fromarray(xorCodeMasked_MorphOpened*255).save(full_path_code)

        return scoreC_best, scoreC_index - self.max_shift

    @torch.inference_mode()
    def matchCodesKernelWise(self, code1, code2, mask1, mask2):
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1 > 127.5, True, False)
        mask2_binary = np.where(mask2 > 127.5, True, False)
        scoreC = []
        scoreM = np.zeros([self.total_num_filters,2*self.max_shift+1],dtype=float)
        scoreM_Morph = np.zeros([self.total_num_filters,2*self.max_shift+1],dtype=float)

        for xshift in range(-self.max_shift, self.max_shift+1):
            andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
            if np.sum(andMasks) == 0:
                scoreC.append(float('inf'))
            else:
                xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.total_num_filters, 1, 1)))
                scoreC.append(np.sum(xorCodesMasked) / (np.sum(andMasks) * self.total_num_filters))

                for c in range(self.total_num_filters):
                    scoreM[c,xshift+self.max_shift] = np.sum(xorCodesMasked[c,:,:]/np.sum(andMasks))

                    xorCodeMasked_MorphOpened = cv2.morphologyEx(np.array(xorCodesMasked[c,:,:],dtype=np.uint8), cv2.MORPH_OPEN, self.morph_kernel)
                    scoreM_Morph[c,xshift+self.max_shift] = np.sum(xorCodeMasked_MorphOpened/np.sum(andMasks))
                    
                    # DEBUG: 
                    # full_path_code = f'temp_images/c-{c:02d}-{xshift+self.max_shift:02d}-{np.sum(xorCodeMasked_MorphOpened/np.sum(andMasks)):0.2f}.png'
                    # Image.fromarray(xorCodeMasked_MorphOpened*255).save(full_path_code)

        # raw / not-normalized / stadnard score
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_best = scoreC[scoreC_index]

        if scoreC_best == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0, -1.0, -1.0
        
        scoreM_best = np.min(scoreM,axis=1)
        scoreM_index = np.argmin(np.array(scoreM),axis=1)

        scoreM_Morph_best = np.min(scoreM_Morph,axis=1)
        scoreM_Morph_index = np.argmin(np.array(scoreM),axis=1)

        return scoreC_best, scoreC_index - self.max_shift, scoreM_best, scoreM_index - self.max_shift, scoreM_Morph_best, scoreM_Morph_index - self.max_shift
