import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, model_geometry, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False, out_shape=(96, 96, 96)):
        if out_shape is None:
            out_shape = [32, 32, 32]
        self.model = model
        self.model_geometry = model_geometry
        self.optimizer = optimizer
        self.device = device
        self.out_shape = out_shape
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        self.model_geometry.eval()

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        points_colors = data.get('points.colors').to(device).permute((0, 2, 1))[:, :3, :]

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        inputs_color = data.get('inputs.colors', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        points_iou_colors = data.get('points_iou.colors').to(device).permute((0, 2, 1))[:, :3, :]

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, points_colors, inputs, inputs_color, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out = self.model(points_iou, inputs, inputs_color, sample=self.eval_sample, **kwargs)

        eval_dict['l1_color'] = torch.abs(points_iou_colors - p_out).mean().cpu().numpy()

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)
        input_c = data.get('inputs.colors', torch.empty(batch_size, 0)).to(device)

        shape = self.out_shape
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            p_r = self.model_geometry(p, inputs, sample=self.eval_sample, **kwargs)
            colors = self.model(p, inputs, input_c, sample=self.eval_sample, **kwargs).view(batch_size, 3, *shape)

        occ_hat = p_r.probs.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            if self.input_type == 'img':
                input_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            else:
                input_path = os.path.join(self.vis_dir, '%03d_in.obj' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_path)
            vis.visualize_colored_voxels_as_point_cloud(
                voxels_out[i], colors[i], os.path.join(self.vis_dir, '%03d.obj' % i), flip_axis=True)

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        colors = data.get('points.colors').to(device).permute((0, 2, 1))[:, :3, :]
        input_g = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        input_c = data.get('inputs.colors', torch.empty(p.size(0), 0)).to(device)

        kwargs = {}
        c0 = self.model.encode_input_g(input_g)
        c1 = self.model.encode_input_c(input_c)
        q_z = self.model.infer_z(p, None, c0, **kwargs)
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points
        pred = self.model.decode(p, z, torch.cat([c0, c1], dim=1), **kwargs)
        loss_i = torch.abs(colors - pred)
        loss = loss + loss_i.mean()

        return loss
