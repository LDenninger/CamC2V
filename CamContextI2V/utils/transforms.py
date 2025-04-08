import torch
import math 

def matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix R into a quaternion (w, x, y, z).
    """
    trace = R.trace()
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        # Find the maximum diagonal element
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = torch.tensor([w, x, y, z], dtype=R.dtype, device=R.device)
    return q / q.norm()

def quaternion_to_matrix(q):
    """
    Convert a quaternion (w, x, y, z) into a 3x3 rotation matrix.
    """
    w, x, y, z = q
    R = torch.empty((3, 3), dtype=q.dtype, device=q.device)
    R[0, 0] = 1 - 2*y*y - 2*z*z
    R[0, 1] = 2*x*y - 2*w*z
    R[0, 2] = 2*x*z + 2*w*y
    R[1, 0] = 2*x*y + 2*w*z
    R[1, 1] = 1 - 2*x*x - 2*z*z
    R[1, 2] = 2*y*z - 2*w*x
    R[2, 0] = 2*x*z - 2*w*y
    R[2, 1] = 2*y*z + 2*w*x
    R[2, 2] = 1 - 2*x*x - 2*y*y
    return R

def slerp(q1, q2, fraction):
    """
    Spherical linear interpolation between two quaternions q1 and q2.
    fraction is a scalar between 0 and 1.
    """
    dot = torch.dot(q1, q2)
    # Ensure the shortest path is taken
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # Quaternions are nearly identical; use linear interpolation
        result = q1 + fraction * (q2 - q1)
        return result / result.norm()
    
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * fraction
    sin_theta = torch.sin(theta)
    
    s1 = torch.sin(theta_0 - theta) / sin_theta_0
    s2 = sin_theta / sin_theta_0
    result = s1 * q1 + s2 * q2
    return result / result.norm()

def resample_poses_slerp(poses, M):
    """
    Resample a trajectory of camera poses.

    Args:
        poses: Tensor of shape (N, 4, 4) representing N world-to-camera poses.
        M: Integer greater than N specifying the desired number of poses.

    Returns:
        Tensor of shape (M, 4, 4) containing the resampled poses.
    """
    N = poses.shape[0]
    
    # Extract translation and rotation components.
    translations = poses[:, :3, 3]         # shape (N, 3)
    rotations = poses[:, :3, :3]             # shape (N, 3, 3)
    
    # Convert each rotation matrix to a quaternion.
    quaternions = []
    for i in range(N):
        q = matrix_to_quaternion(rotations[i])
        quaternions.append(q)
    quaternions = torch.stack(quaternions, dim=0)  # shape (N, 4)
    
    new_poses = []
    # Create M uniformly spaced parameter values from 0 to N-1.
    t_vals = torch.linspace(0, N - 1, steps=M)
    for t in t_vals:
        i = int(torch.floor(t).item())
        f = t - i  # fractional part for interpolation
        j = min(i + 1, N - 1)
        
        # Linear interpolation for the translation.
        trans = (1 - f) * translations[i] + f * translations[j]
        
        # Spherical linear interpolation (slerp) for the rotation.
        q1 = quaternions[i]
        q2 = quaternions[j]
        q_interp = slerp(q1, q2, f)
        R_interp = quaternion_to_matrix(q_interp)
        
        # Reconstruct the 4x4 homogeneous transformation matrix.
        pose_interp = torch.eye(4, dtype=poses.dtype, device=poses.device)
        pose_interp[:3, :3] = R_interp
        pose_interp[:3, 3] = trans
        
        new_poses.append(pose_interp)
    
    return torch.stack(new_poses, dim=0)