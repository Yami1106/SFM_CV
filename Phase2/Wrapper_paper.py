import argparse
import glob
from html import parser
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
#import imageio
import imageio.v2 as imageio
import torch
import matplotlib.pyplot as plt
import os
import json
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim

from NeRFModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def loadDataset(data_path, mode):
    """
    Input:
        data_path: dataset path
        mode: train or test
    Outputs:
        camera_info: image width, height, camera matrix 
        images: images
        pose: corresponding camera pose in world frame
    """

    # read the transforms_train and test json files
    json_path = os.path.join(data_path, f"transforms_{mode}.json")

    with open(json_path, 'r') as f:
        meta = json.load(f)

    # read images and poses
    images = []
    poses = []

    for frame in meta['frames']:
        img_path = os.path.join(data_path, frame['file_path'] + ".png")

        # read the image and normalize it to [0,1]
        img = imageio.imread(img_path).astype(np.float32) / 255.0

        if img.shape[-1] == 4:
            rgb = img[..., :3]
            alpha = img[..., 3:4]
            img = rgb * alpha + (1.0 - alpha)

        # in case of RGBA, we only need RGB values so retain only that
        img[..., :3]

        pose = np.array(frame['transform_matrix'], dtype=np.float32)
        images.append(img)
        poses.append(pose)
    
    images = np.stack(images, axis=0)
    poses = np.stack(poses, axis=0)

    # calculate H,W and focal length 
    H = images.shape[1]
    W = images.shape[2]

    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / math.tan(0.5 * camera_angle_x)

    # make the intrinsic matrix
    K = np.array([[focal, 0, W/2],
                  [0, focal, H/2],
                  [0, 0, 1]], dtype=np.float32)
    
    camera_info = {
        "H": H,
        "W": W,
        "focal": focal,
        "K": K
    }

    return images, poses, camera_info


def PixelToRay(camera_info, pose, pixelPosition, args):
    """
    Input:
        camera_info: image width, height, camera matrix 
        pose: camera pose in world frame
        pixelPoition: pixel position in the image
        args: get near and far range, sample rate ...
    Outputs:
        ray origin and direction
    """

    # get the height and width of image and focal length from camera_info
    H = camera_info["H"]
    W = camera_info["W"]
    focal = camera_info["focal"]

    # get the pixel position 
    x = pixelPosition[:, 0]
    y = pixelPosition[:, 1]

    # convert the pixel position to camera direction
    dirs = np.stack([(x - W/2) / focal, -(y - H/2) / focal, -np.ones_like(x)], axis=-1)

    # camera to world transformation, rotation and translation
    R = pose[:3, :3]
    t = pose[:3, 3]

    # rotate the camera direction to world frame
    rays_direction = dirs @ R.T

    # all rays must origintae from the camera center 
    rays_origin = np.repeat(t[None, :], repeats=dirs.shape[0], axis=0)

    return rays_origin, rays_direction


def generateBatch(images, poses, camera_info, args):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays
    """

    # get the number of images, height and width of image from images and rays per batch from args
    n_imgs,H,W,_ = images.shape
    n_rays=int(args.n_rays_batch)

    rays_origin_list = []
    rays_direction_list = []
    rgb_list = []

    for _ in range(n_rays):
        # choose an image randomly from the dataset
        img_idx = np.random.randint(0, n_imgs)

        # now choose a random pixel from the image
        x = np.random.randint(0, W)
        y = np.random.randint(0, H)

        pixel_pos = np.array([[x, y]], dtype=np.float32)

        # get the ray for this pixel
        ray_or,ray_dir = PixelToRay(camera_info, poses[img_idx], pixel_pos, args)

        # get the rgb value ground truth for this pixel 
        rgb = images[img_idx, y, x]

        rays_origin_list.append(ray_or[0])
        rays_direction_list.append(ray_dir[0])
        rgb_list.append(rgb)
    
    rays_origin = torch.tensor(np.array(rays_origin_list), dtype=torch.float32, device=device)
    rays_direction = torch.tensor(np.array(rays_direction_list), dtype=torch.float32, device=device)
    rgb_gt = torch.tensor(np.array(rgb_list), dtype=torch.float32, device=device)

    return rays_origin, rays_direction, rgb_gt


def prob_dist(z_vals, weights, N_samples, device):
    # add small epsilon to prevent division by zero
    weights = weights + 1e-5

    # normalize weights to get a probability distribution (PDF)
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)

    # cumulative distribution function (CDF)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

    # get uniform random samples
    u = torch.rand(weights.shape[0], N_samples, device=device).contiguous()

    # invert the CDF — find where each random sample falls
    inds  = torch.searchsorted(cdf.contiguous(), u, right=True)
    below = torch.clamp(inds - 1, min=0, max=z_vals.shape[-1] - 1)
    above = torch.clamp(inds,     min=0, max=z_vals.shape[-1] - 1)

    inds_g = torch.stack([below, above], dim=-1)

    cdf_g = torch.gather(cdf,    1, inds_g.view(weights.shape[0], -1)).view(*inds_g.shape)
    z_g   = torch.gather(z_vals, 1, inds_g.view(weights.shape[0], -1)).view(*inds_g.shape)

    # linear interpolation to get exact sample position
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t     = (u - cdf_g[..., 0]) / denom

    samples = z_g[..., 0] + t * (z_g[..., 1] - z_g[..., 0])
    return samples


def volume_rendering(rgb, sigma, z_vals, rays_direction, N_rays, device):
    # get distance between adjacent sample points
    dists = z_vals[:, 1:] - z_vals[:, :-1]

    # final sample is very far away and can absorb any remaining transmittance
    dists_last = 1e10 * torch.ones_like(dists[:, :1])
    dists = torch.cat([dists, dists_last], dim=-1)

    # ray direction magnitude is also important so include that too 
    rays_normalized = torch.norm(rays_direction, dim=-1, keepdim=True)
    dists = dists * rays_normalized

    #volume rendering 
    alpha = 1.0 - torch.exp(-sigma * dists)
    transmittance = torch.cumprod(torch.cat([torch.ones((N_rays, 1), device=device), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]

    weights = alpha * transmittance

    #finaal rgb value will be weighted sum 
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
    acc_map = torch.sum(weights, dim=1, keepdim=True)
    rgb_final = rgb_final + (1.0 - acc_map)

    return rgb_final, weights


def render(model, rays_origin, rays_direction, args, model_fine=None):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
    Outputs:
        rgb values of input rays
    """

    N_rays = rays_origin.shape[0]
    N_samples = int(args.n_sample)

    # sample depth values between near and far bounds for each ray
    t_vals = torch.linspace(args.near, args.far, N_samples, device=device)
    t_vals = t_vals.unsqueeze(0).expand(N_rays, N_samples)

    # divide the ray into N bins and randomly sample in those bins
    midpoint = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
    upper = torch.cat([midpoint, t_vals[:, -1:]], dim=-1)
    lower = torch.cat([t_vals[:, :1], midpoint], dim=-1)

    rand = torch.rand(t_vals.shape, device=device)
    z_vals = lower + (upper - lower) * rand

    # calculate 3D sample points along each ray 
    # this is equivalent to o+td
    pts = rays_origin.unsqueeze(1) + rays_direction.unsqueeze(1) * z_vals.unsqueeze(-1)

    # viewing directions for input
    view_dir = rays_direction / torch.norm(rays_direction, dim=-1, keepdim=True)
    view_dir = view_dir.unsqueeze(1).expand_as(pts)

    # flatten points to help model process all points together
    pts_flat = pts.reshape(-1, 3)
    view_dir_flat = view_dir.reshape(-1, 3)

    output = model(pts_flat, view_dir_flat)
    output = output.reshape(N_rays, N_samples, 4)

    rgb = output[..., :3]
    sigma = output[..., 3]

    # coarse rgb and weights from volume rendering
    rgb_coarse, weights_coarse = volume_rendering(rgb, sigma, z_vals, rays_direction, N_rays, device)

    # print("sigma mean:", sigma.mean().item())
    # print("weights mean:", weights_coarse.mean().item())

    # if no fine model is provided return coarse result
    if model_fine is None:
        return rgb_coarse

    # use coarse weights as a pdf to sample new points concentrated near surfaces
    z_vals_fine = prob_dist(z_vals, weights_coarse.detach(), int(args.n_sample_fine), device)

    # combine coarse and fine samples and sort by depth
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_vals_fine], dim=-1), dim=-1)

    # calculate 3D sample points along each ray 
    # this is equivalent to o+td
    pts = rays_origin.unsqueeze(1) + rays_direction.unsqueeze(1) * z_vals_combined.unsqueeze(-1)

    # viewing directions for input
    view_dir = rays_direction / torch.norm(rays_direction, dim=-1, keepdim=True)
    view_dir = view_dir.unsqueeze(1).expand_as(pts)

    # flatten points to help model process all points together
    pts_flat = pts.reshape(-1, 3)
    view_dir_flat = view_dir.reshape(-1, 3)

    output = model_fine(pts_flat, view_dir_flat)
    output = output.reshape(N_rays, z_vals_combined.shape[1], 4)

    rgb = output[..., :3]
    sigma = output[..., 3]

    #finaal rgb value will be weighted sum 
    rgb_fine, _ = volume_rendering(rgb, sigma, z_vals_combined, rays_direction, N_rays, device)

    return rgb_coarse, rgb_fine


def loss(groundtruth, prediction):
    '''
    Input:
        groundtruth: rgb values of input rays
        prediction: predicted rgb values of input rays

    Output:
        MSE loss between groundtruth and prediction        
    '''
    return torch.mean((groundtruth - prediction) ** 2)


def compute_metrics(images, args):
    image_paths = sorted(glob.glob(os.path.join(args.images_path, "test_*.png")))

    psnr_list = []
    ssim_list = []

    for idx, pred_path in enumerate(image_paths):
        pred = imageio.imread(pred_path).astype(np.float32) / 255.0
        gt   = images[idx][..., :3]
        pred = pred[..., :3]

        mse  = np.mean((pred - gt) ** 2)
        psnr = -10.0 * np.log10(mse) if mse > 0 else 100.0
        psnr_list.append(psnr)

        ssim_val = ssim(gt, pred, data_range=1.0, channel_axis=-1)
        ssim_list.append(ssim_val)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    with open(os.path.join(args.images_path, "metrics.txt"), "w") as f:
        f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        for i, (p, s) in enumerate(zip(psnr_list, ssim_list)):
            f.write(f"  Image {i:03d}: PSNR={p:.4f}  SSIM={s:.4f}\n")

    return avg_psnr, avg_ssim, psnr_list


def save_comparison_images(images, args, n_best=3, n_worst=3):
    image_paths = sorted(glob.glob(os.path.join(args.images_path, "test_*.png")))

    psnr_list = []
    for idx, pred_path in enumerate(image_paths):
        pred = imageio.imread(pred_path).astype(np.float32) / 255.0
        gt   = images[idx][..., :3]
        mse  = np.mean((pred[..., :3] - gt) ** 2)
        psnr = -10.0 * np.log10(mse) if mse > 0 else 100.0
        psnr_list.append((psnr, idx, pred_path))

    psnr_list.sort(key=lambda x: x[0])

    worst = psnr_list[:n_worst]
    best  = psnr_list[-n_best:][::-1]

    for label, subset in [("best", best), ("worst", worst)]:
        fig, axes = plt.subplots(len(subset), 2, figsize=(8, 4 * len(subset)))
        for row, (psnr_val, idx, pred_path) in enumerate(subset):
            gt   = images[idx][..., :3]
            pred = imageio.imread(pred_path).astype(np.float32) / 255.0

            axes[row, 0].imshow(np.clip(gt,   0, 1))
            axes[row, 0].set_title(f"Ground Truth (idx {idx})")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(np.clip(pred, 0, 1))
            axes[row, 1].set_title(f"Predicted  PSNR={psnr_val:.2f}dB")
            axes[row, 1].axis("off")

        plt.tight_layout()
        save_path = os.path.join(args.images_path, f"comparison_{label}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")


def train(images, poses, camera_info, args):
    # loading coarse and fine models 
    model = NeRFmodel(args.n_pos_freq, args.n_dirc_freq).to(device)
    model_fine = NeRFmodel(args.n_pos_freq, args.n_dirc_freq).to(device)

    # both networks should to be optimized together
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(model_fine.parameters()),
        lr=args.lrate
    )

    # make the required folders to save respective files
    os.makedirs(args.logs_path, exist_ok=True)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.images_path, exist_ok=True)

    writer = SummaryWriter(args.logs_path)

    start_iter = 0
    if args.load_checkpoint:
        ckpts = sorted(glob.glob(os.path.join(args.checkpoint_path, "*.pth")))
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print("Loading checkpoint:", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model_fine.load_state_dict(ckpt["model_fine_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_iter = ckpt["iter"] + 1

    model.train()
    model_fine.train()

    # run the training loop
    for it in tqdm(range(start_iter, args.max_iters)):
        # training rays
        rays_origin, rays_direction, rgb_gt = generateBatch(images, poses, camera_info, args)

        # forward rendering
        rgb_coarse, rgb_fine = render(model, rays_origin, rays_direction, args, model_fine=model_fine)

        # get the loss
        train_loss = loss(rgb_gt, rgb_coarse) + loss(rgb_gt, rgb_fine)

        # backward prop and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # tensorboard log
        writer.add_scalar("train/loss", train_loss.item(), it)

        if it % 100 == 0:
            print(f"Iter {it}: Loss = {train_loss.item():.6f}")

        # save checkpoint
        if it % args.save_ckpt_iter == 0 and it > 0:
            ckpt_path = os.path.join(args.checkpoint_path, f"ckpt_{it:06d}.pth")
            torch.save({
                "iter": it,
                "model_state_dict": model.state_dict(),
                "model_fine_state_dict": model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    writer.close()


def create_360_gif(args):
    # get all rendered test images in order
    image_paths = sorted(glob.glob(os.path.join(args.images_path, "test_*.png")))
    
    if len(image_paths) == 0:
        print("No test images found to create GIF.")
        return

    gif_frames = []
    for img_path in image_paths:
        frame = imageio.imread(img_path)
        gif_frames.append(frame)

    # save as gif
    gif_path = os.path.join(args.images_path, "360_view.gif")
    imageio.mimwrite(gif_path, gif_frames, fps=10, loop=0)
    print(f"Saved 360 GIF to {gif_path}")


def test(images, poses, camera_info, args):
    # load both coarse and fine models
    model = NeRFmodel(args.n_pos_freq, args.n_dirc_freq).to(device)
    model_fine = NeRFmodel(args.n_pos_freq, args.n_dirc_freq).to(device)

    ckpts = sorted(glob.glob(os.path.join(args.checkpoint_path, "*.pth")))
    if len(ckpts) == 0:
        raise FileNotFoundError("No checkpoint found for testing.")

    # load the latest checkpoint
    ckpt_path = ckpts[-1]
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model_fine.load_state_dict(ckpt["model_fine_state_dict"])

    # set both models to evaluation mode
    model.eval()
    model_fine.eval()

    os.makedirs(args.images_path, exist_ok=True)

    H = camera_info["H"]
    W = camera_info["W"]

    with torch.no_grad():
        # loop through each test image and render it
        for idx in range(len(images)):
            pose = poses[idx]

            # create a grid of pixel coordinates for the image
            xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
            pixel_positions = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1).astype(np.float32)

            rays_origin, rays_direction = PixelToRay(camera_info, pose, pixel_positions, args)

            rays_origin = torch.tensor(rays_origin, dtype=torch.float32, device=device)
            rays_direction = torch.tensor(rays_direction, dtype=torch.float32, device=device)

            # render image in chunck to avoid running out of memory
            rgb_chunks = []
            total_rays = rays_origin.shape[0]

            for start in range(0, total_rays, args.chunk_size):
                end = min(start + args.chunk_size, total_rays)
                rays_origin_chunk = rays_origin[start:end]
                rays_direction_chunk = rays_direction[start:end]

                _, rgb_chunk = render(model, rays_origin_chunk, rays_direction_chunk, args, model_fine=model_fine)
                rgb_chunks.append(rgb_chunk.cpu())

            rgb_pred = torch.cat(rgb_chunks, dim=0)

            pred_img = rgb_pred.reshape(H, W, 3).numpy()
            pred_img = np.clip(pred_img, 0.0, 1.0)

            save_path = os.path.join(args.images_path, f"test_{idx:03d}.png")
            imageio.imwrite(save_path, (pred_img * 255).astype(np.uint8))

            print(f"Saved {save_path}")

    # generate 360 gif from saved test images
    print("Generating 360 GIF...")
    create_360_gif(args)

    # get metrics
    print("Computing PSNR and SSIM...")
    compute_metrics(images, args)

    # save best and worst comparisons
    print("Saving comparison images...")
    save_comparison_images(images, args)


def main(args):
    # load data
    print("Loading data...")
    images, poses, camera_info = loadDataset(args.data_path, args.mode)

    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)


def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Phase2/data/lego/",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',type=float,default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',type=int,default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',type=int,default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',type=int,default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',type=int,default=64,help="number of sample per ray")
    parser.add_argument('--n_sample_fine',type=int,default=128,help="number of fine samples per ray")
    parser.add_argument('--max_iters',type=int,default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./Phase2/example_checkpoint/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',type=bool,default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',type=int,default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    parser.add_argument('--chunk_size', type=int, default=4096, help="number of rays to render at once during testing")

    # bounds for ray sampling
    parser.add_argument('--near', default=2.0, type=float, help="near bound")
    parser.add_argument('--far', default=6.0, type=float, help="far bound")
    return parser


if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)