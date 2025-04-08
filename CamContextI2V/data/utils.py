import copy
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from torch import Tensor

def relative_pose(rt: Tensor, mode, ref_index) -> Tensor:
    '''
    :param rt: F,4,4
    :param mode: left or right
    :return:
    '''
    if mode == "left":
        rt = rt[ref_index].inverse() @ rt
    elif mode == "right":
        rt = rt @ rt[ref_index].inverse()
    return rt


def create_line_point_cloud(start_point, end_point, num_points=50, color=np.array([0, 0, 1.0])):
    points = np.linspace(start_point, end_point, num_points)
    colors = np.tile(color, (points.shape[0], 1))
    return points, colors


def remove_outliers(pcd):

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=3.0)
    # cl, ind = pcd.remove_radius_outlier(nb_points=3, radius=0.1)
    return pcd.select_by_index(ind)

def construct_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.transform([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return pcd


def camera_pose_lerp(c2w: Tensor, target_frames: int):
    weights = torch.linspace(0, c2w.size(0) - 1, target_frames, dtype=c2w.dtype, device=c2w.device)
    left_indices = weights.floor().long()
    right_indices = weights.ceil().long()

    return torch.lerp(c2w[left_indices], c2w[right_indices], weights.unsqueeze(-1).unsqueeze(-1).frac())


def apply_thresholded_conv(mask, kernel_size=5, threshold=0.9):

    b, f, h, w = mask.shape
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)
    mask_padded = F.pad(mask, pad=(kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
    mask_reshaped = mask_padded.view(-1, 1, h + 2*(kernel_size//2), w + 2*(kernel_size//2))
    summed_mask = F.conv2d(mask_reshaped, kernel)
    thresholded_mask = (summed_mask >= (kernel_size * kernel_size * threshold)).float()

    thresholded_mask = thresholded_mask.view(b, f, h, w)

    return thresholded_mask


def constrain_to_multiple_of(x, min_val=0, max_val=None, multiple_of=14):
    y = (np.round(x / multiple_of) * multiple_of).astype(int)

    if max_val is not None and y > max_val:
        y = (np.floor(x / multiple_of) * multiple_of).astype(int)

    if y < min_val:
        y = (np.ceil(x / multiple_of) * multiple_of).astype(int)

    return y


def add_camera_trace(points, colors, points_x, points_y):
    x, y = points_x, points_y
    for idx in [[0, 0], [0, -1], [-1, 0], [-1, -1]]:
        camera, camera_colors = create_line_point_cloud(
            start_point=np.array([0, 0, 0]),
            end_point=np.array([x[idx[0]][idx[1]], y[idx[0]][idx[1]], 1.0]),
            num_points=50,
        )
        points = np.concatenate((points, camera), axis=0)
        colors = np.concatenate((colors, camera_colors), axis=0)

    for start_idx, end_idx in [
        [[0, 0], [0, -1]],
        [[0, 0], [-1, 0]],
        [[-1, -1], [0, -1]],
        [[-1, -1], [-1, 0]],
    ]:
        camera, camera_colors = create_line_point_cloud(
            start_point=np.array([x[start_idx[0]][start_idx[1]], y[start_idx[0]][start_idx[1]], 1.0]),
            end_point=np.array([x[end_idx[0]][end_idx[1]], y[end_idx[0]][end_idx[1]], 1.0]),
            num_points=50,
        )
        points = np.concatenate((points, camera), axis=0)
        colors = np.concatenate((colors, camera_colors), axis=0)

    return points, colors


def create_relative(RT_list, K_1=4.7, dataset="syn"):
    scale_T = 1
    RT_list = [RT.reshape(3, 4) for RT in RT_list]
    temp = []
    first_frame_RT = copy.deepcopy(RT_list[0])
    # first_frame_R_inv = np.linalg.inv(first_frame_RT[:,:3])
    first_frame_R_inv = first_frame_RT[:, :3].T
    first_frame_T = first_frame_RT[:, -1]
    for RT in RT_list:
        RT[:, :3] = np.dot(RT[:, :3], first_frame_R_inv)
        RT[:, -1] = RT[:, -1] - np.dot(RT[:, :3], first_frame_T)
        RT[:, -1] = RT[:, -1] * scale_T
        temp.append(RT)
    if dataset == "realestate":
        temp = [RT.reshape(-1) for RT in temp]
    return temp


def sigma_matrix2(sig_x, sig_y, theta):
    """Calculate the rotated sigma matrix (two dimensional matrix).
    Args:
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
    Returns:
        ndarray: Rotated sigma matrix.
    """
    d_matrix = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))


def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.
    Args:
        kernel_size (int):
    Returns:
        xy (ndarray): with the shape (kernel_size, kernel_size, 2)
        xx (ndarray): with the shape (kernel_size, kernel_size)
        yy (ndarray): with the shape (kernel_size, kernel_size)
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size,
                                                                           1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def pdf2(sigma_matrix, grid):
    """Calculate PDF of the bivariate Gaussian distribution.
    Args:
        sigma_matrix (ndarray): with the shape (2, 2)
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.
    Returns:
        kernel (ndarrray): un-normalized kernel.
    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel


def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    """Generate a bivariate isotropic or anisotropic Gaussian kernel.
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool):
    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel
