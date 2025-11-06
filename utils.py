"""
:Author: Zachary T. Varley
:Year: 2025
:License: MIT License
:Description: monofile of EBSD Logic

"""

"""

Unit normal quaternions (points that sit on the surface of the 3-sphere with
unit radius in 4D Euclidean space) are used to represent 3D rotations. This
module provides a set of operations for working with quaternions in general.
Often times only the angle of the rotation is needed for comparison amongst
quaternions, so separate functions are provided for accelerating this common
operation. The quaternion (w, x, y, z) is used to represent a rotation that is
indistinguishable from the quaternion (-w, x, y, z), so the standardization
function is provided to make the real part non-negative by conjugation, limiting
the hypervolume we work with to the positive w hemisphere of the 3-sphere.

For more information on quaternions, see:

https://en.wikipedia.org/wiki/Quaternion

Adopted from PyTorch3D

https://github.com/facebookresearch/pytorch3d

"""

import math
import torch
from torch import Tensor
import sys
import time
from typing import Optional, Tuple, List, Union
import random
from torch.nn import Linear, Module
from torch.ao.quantization import quantize_dynamic


@torch.jit.script
def qu_std(qu: Tensor) -> Tensor:
    """
    Standardize unit quaternion to have non-negative real part.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(qu[..., 0:1] >= 0, qu, -qu)


@torch.jit.script
def qu_norm(qu: Tensor) -> Tensor:
    """
    Normalize quaternions to unit norm.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Tensor of normalized quaternions.
    """
    return qu / torch.norm(qu, dim=-1, keepdim=True)


@torch.jit.script
def qu_prod_raw(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


@torch.jit.script
def qu_prod(a: Tensor, b: Tensor) -> Tensor:
    """
    Quaternion multiplication, then make real part non-negative.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b Tensor shape (..., 4) of the quaternion product.

    """
    ab = qu_prod_raw(a, b)
    return qu_std(ab)


@torch.jit.script
def qu_slerp(a: Tensor, b: Tensor, t: float) -> Tensor:
    """
    Spherical linear interpolation between two quaternions.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)
        t: interpolation parameter between 0 and 1

    Returns:
        The interpolated quaternions, a tensor of shape (..., 4).
    """
    a = qu_norm(a)
    b = qu_norm(b)
    cos_theta = torch.sum(a * b, dim=-1)
    angle = torch.acos(cos_theta)
    sin_theta = torch.sin(angle)
    w1 = torch.sin((1 - t) * angle) / sin_theta
    w2 = torch.sin(t * angle) / sin_theta
    return (a.unsqueeze(-1) * w1 + b.unsqueeze(-1) * w2).squeeze(-1)


@torch.jit.script
def qu_prod_pos_real(a: Tensor, b: Tensor) -> Tensor:
    """
    Return only the magnitude of the real part of the quaternion product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b Tensor shape (..., ) of quaternion product real part magnitudes.
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    return ow.abs()


@torch.jit.script
def qu_triple_prod_pos_real(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """
    Return only the magnitude of the real part of the quaternion triple product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)
        c: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b*c Tensor shape (..., ) of quaternion triple product real part magnitudes.
    """
    return qu_prod_pos_real(a, qu_prod(b, c))


@torch.jit.script
def qu_prod_axis(a: Tensor, b: Tensor) -> Tensor:
    """
    Return the axis of the quaternion product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b Tensor shape (..., 3) of quaternion product axes.
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ox, oy, oz), -1)


@torch.jit.script
def qu_conj(qu: Tensor) -> Tensor:
    """
    Get the unit quaternions for the inverse action.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    scaling = torch.tensor([1, -1, -1, -1], device=qu.device, dtype=qu.dtype)
    return qu * scaling


@torch.jit.script
def qu_apply(qu: Tensor, point: Tensor) -> Tensor:
    """
    Rotate 3D points by unit quaternions.

    Args:
        qu: shape (..., 4) of quaternions in the form (w, x, y, z)
        point: shape (..., 3) of 3D points.

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    aw, ax, ay, az = qu[..., 0], qu[..., 1], qu[..., 2], qu[..., 3]
    bx, by, bz = point[..., 0], point[..., 1], point[..., 2]

    # need qu_prod_axis(qu_prod_raw(qu, point_as_quaternion), qu_conj(qu))
    # do qu_prod_raw(qu, point_as_quaternion) first to get intermediate values
    iw = aw - ax * bx - ay * by - az * bz
    ix = aw * bx + ax + ay * bz - az * by
    iy = aw * by - ax * bz + ay + az * bx
    iz = aw * bz + ax * by - ay * bx + az

    # next qu_prod_axis(qu_prod_raw(qu, point_as_quaternion), qu_conj(qu))
    ox = -iw * ax + ix * aw - iy * az + iz * ay
    oy = -iw * ay + ix * az + iy * aw - iz * ax
    oz = -iw * az - ix * ay + iy * ax + iz * aw

    return torch.stack((ox, oy, oz), -1)


@torch.jit.script
def qu_norm_std(qu: Tensor) -> Tensor:
    """
    Normalize a quaternion to unit norm and make real part non-negative.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Tensor of normalized and standardized quaternions.
    """
    return qu_std(qu_norm(qu))


@torch.jit.script
def quaternion_rotate_sets_sphere(points_start: Tensor, points_finish) -> Tensor:
    """
    Determine the quaternions that rotate the points_start to the points_finish.
    All points are assumed to be on the unit sphere. The cross product is used
    as the axis of rotation, but there are an infinite number of quaternions that
    fulfill the requirement as the points can be rotated around their axis by
    an arbitrary angle, and they will still have the same latitude and longitude.

    Args:
        points_start: Starting points as tensor of shape (..., 3).
        points_finish: Ending points as tensor of shape (..., 3).

    Returns:
        The quaternions, as tensor of shape (..., 4).

    """
    # determine mask for numerical stability
    valid = torch.abs(torch.sum(points_start * points_finish, dim=-1)) < 0.999999
    # get the cross product of the two sets of points
    cross = torch.cross(points_start[valid], points_finish[valid], dim=-1)
    # get the dot product of the two sets of points
    dot = torch.sum(points_start[valid] * points_finish[valid], dim=-1)
    # get the angle
    angle = torch.atan2(torch.norm(cross, dim=-1), dot)
    # add tau to the angle if the cross product is negative
    angle[angle < 0] += 2 * torch.pi
    # set the output
    out = torch.empty(
        (points_start.shape[0], 4), dtype=points_start.dtype, device=points_start.device
    )
    out[valid, 0] = torch.cos(angle / 2)
    out[valid, 1:] = torch.sin(angle / 2).unsqueeze(-1) * (
        cross / torch.norm(cross, dim=-1, keepdim=True)
    )
    out[~valid, 0] = 1
    out[~valid, 1:] = 0
    return out


@torch.jit.script
def qu_angle(qu: Tensor) -> Tensor:
    """
    Compute angles of rotation for quaternions.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        tensor of shape (..., ) of rotation angles.
    """
    return 2 * torch.acos(qu[..., 0])


"""

Routines for orientation representations adopted from PyTorch3D and from EMsoft

https://github.com/facebookresearch/pytorch3d

https://github.com/marcdegraef/3Drotations

Abbreviations used in the code:

cu: cubochoric
ho: homochoric
ax: axis-angle
qu: quaternion
om: orientation matrix
bu: Bunge ZXZ Euler angles
cl: Clifford Torus
ro: Rodrigues-Frank vector
zh: 6D continuous representation of orientation

"""


@torch.jit.script
def qu2ho(qu: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to homochoric coordinates.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4).

    Returns:
        Homochoric coordinates as tensor of shape (..., 3).
    """
    if qu.size(-1) != 4:
        raise ValueError(f"Invalid quaternion shape {qu.shape}.")
    ho = torch.empty_like(qu[..., :3])
    # get the angle
    angle = 2 * torch.acos(qu[..., 0:1].clamp_(min=-1.0, max=1.0))
    # get the unit vector
    unit = qu[..., 1:] / torch.norm(qu[..., 1:], dim=-1, keepdim=True)
    ho = unit * (3.0 * (angle - torch.sin(angle)) / 4.0) ** (1 / 3)
    # fix the case where the angle is zero
    ho[(angle.squeeze(-1) < 1e-8)] = 0.0
    return ho


@torch.jit.script
def om2qu(matrix: Tensor) -> Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).

    Notes:

    Farrell, J.A., 2015. Computation of the Quaternion from a Rotation Matrix.
    University of California, 2.

    "Converting a Rotation Matrix to a Quaternion" by Mike Day, Insomniac Games

    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]

    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m10 = matrix[..., 1, 0]

    mask_A = m22 < 0
    mask_B = m00 > m11
    mask_C = m00 < -m11
    branch_1 = mask_A & mask_B
    branch_2 = mask_A & ~mask_B
    branch_3 = ~mask_A & mask_C
    branch_4 = ~mask_A & ~mask_C

    branch_1_t = 1 + m00[branch_1] - m11[branch_1] - m22[branch_1]
    branch_1_t_rsqrt = 0.5 * torch.rsqrt(branch_1_t)
    branch_2_t = 1 - m00[branch_2] + m11[branch_2] - m22[branch_2]
    branch_2_t_rsqrt = 0.5 * torch.rsqrt(branch_2_t)
    branch_3_t = 1 - m00[branch_3] - m11[branch_3] + m22[branch_3]
    branch_3_t_rsqrt = 0.5 * torch.rsqrt(branch_3_t)
    branch_4_t = 1 + m00[branch_4] + m11[branch_4] + m22[branch_4]
    branch_4_t_rsqrt = 0.5 * torch.rsqrt(branch_4_t)

    qu = torch.empty(batch_dim + (4,), dtype=matrix.dtype, device=matrix.device)

    qu[branch_1, 1] = branch_1_t * branch_1_t_rsqrt
    qu[branch_1, 2] = (m01[branch_1] + m10[branch_1]) * branch_1_t_rsqrt
    qu[branch_1, 3] = (m20[branch_1] + m02[branch_1]) * branch_1_t_rsqrt
    qu[branch_1, 0] = (m12[branch_1] - m21[branch_1]) * branch_1_t_rsqrt

    qu[branch_2, 1] = (m01[branch_2] + m10[branch_2]) * branch_2_t_rsqrt
    qu[branch_2, 2] = branch_2_t * branch_2_t_rsqrt
    qu[branch_2, 3] = (m12[branch_2] + m21[branch_2]) * branch_2_t_rsqrt
    qu[branch_2, 0] = (m20[branch_2] - m02[branch_2]) * branch_2_t_rsqrt

    qu[branch_3, 1] = (m20[branch_3] + m02[branch_3]) * branch_3_t_rsqrt
    qu[branch_3, 2] = (m12[branch_3] + m21[branch_3]) * branch_3_t_rsqrt
    qu[branch_3, 3] = branch_3_t * branch_3_t_rsqrt
    qu[branch_3, 0] = (m01[branch_3] - m10[branch_3]) * branch_3_t_rsqrt

    qu[branch_4, 1] = (m12[branch_4] - m21[branch_4]) * branch_4_t_rsqrt
    qu[branch_4, 2] = (m20[branch_4] - m02[branch_4]) * branch_4_t_rsqrt
    qu[branch_4, 3] = (m01[branch_4] - m10[branch_4]) * branch_4_t_rsqrt
    qu[branch_4, 0] = branch_4_t * branch_4_t_rsqrt

    # guarantee the correct axis signs
    qu[..., 0] = torch.abs(qu[..., 0])
    qu[..., 1] = qu[..., 1].copysign((m21 - m12))
    qu[..., 2] = qu[..., 2].copysign((m02 - m20))
    qu[..., 3] = qu[..., 3].copysign((m10 - m01))

    return qu


@torch.jit.script
def om2ax(matrix: Tensor) -> Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        axis-angle representation as tensor of shape (..., 4).

    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]

    # set the output with the same batch dimensions as the input
    axis = torch.empty(batch_dim + (4,), dtype=matrix.dtype, device=matrix.device)

    # Get the trace of the matrix
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]

    # find the angles
    acos_arg = 0.5 * (trace - 1.0)
    acos_arg = torch.clamp(acos_arg, -1.0, 1.0)
    theta = torch.acos(acos_arg)

    # where the angle is small, treat theta/sin(theta) as 1
    stable = theta > 0.001
    axis[..., 0] = matrix[..., 2, 1] - matrix[..., 1, 2]
    axis[..., 1] = matrix[..., 0, 2] - matrix[..., 2, 0]
    axis[..., 2] = matrix[..., 1, 0] - matrix[..., 0, 1]
    factor = torch.where(stable, 0.5 / torch.sin(theta), 0.5)
    axis[..., :3] = factor[:, None] * axis[:, :3]

    # normalize the axis
    axis[..., :3] /= torch.norm(axis[:, :3], dim=-1, keepdim=True)

    # set the angle
    axis[..., 3] = theta

    return axis.view(batch_dim + (4,))


@torch.jit.script
def ax2om(axis_angle: Tensor) -> Tensor:
    """
    Convert axis-angle representation to rotation matrices.

    Args:
        axis_angle: Axis-angle representation as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if axis_angle.size(-1) != 4:
        raise ValueError(f"Invalid axis-angle shape {axis_angle.shape}.")

    batch_dim = axis_angle.shape[:-1]
    data_n = int(torch.prod(torch.tensor(batch_dim)))

    # set the output
    matrices = torch.empty(
        batch_dim + (3, 3), dtype=axis_angle.dtype, device=axis_angle.device
    )

    theta = axis_angle[..., 3:4]
    omega = axis_angle[..., :3] * theta

    matrices = torch.zeros((data_n, 3, 3), dtype=omega.dtype, device=omega.device)
    matrices[..., 0, 1] = -omega[..., 2]
    matrices[..., 0, 2] = omega[..., 1]
    matrices[..., 1, 2] = -omega[..., 0]
    matrices[..., 1, 0] = omega[..., 2]
    matrices[..., 2, 0] = -omega[..., 1]
    matrices[..., 2, 1] = omega[..., 0]

    skew_sq = torch.matmul(matrices, matrices)

    # Taylor expansion for small angles of each factor
    stable = (theta > 0.05).squeeze()

    theta_unstable = theta[~stable].unsqueeze(-1)

    # This prefactor is only used for the calculation of exp(skew)
    # sin(theta) / theta
    # expression: 1 - theta^2 / 6 + theta^4 / 120 - theta^6 / 5040 ...
    prefactor1 = 1 - theta_unstable**2 / 6

    # This prefactor is shared between calculations of exp(skew) and v
    # (1 - cos(theta)) / theta^2
    # expression: 1/2 - theta^2 / 24 + theta^4 / 720 - theta^6 / 40320 ...
    prefactor2 = 1 / 2 - theta_unstable**2 / 24

    theta_stable = theta[stable].unsqueeze(-1)
    matrices[stable] = (
        torch.eye(3, dtype=matrices.dtype, device=matrices.device)
        + (torch.sin(theta_stable) / theta_stable) * matrices[stable]
        + (1 - torch.cos(theta_stable)) / theta_stable**2 * skew_sq[stable]
    )
    matrices[~stable] = (
        torch.eye(3, dtype=matrices.dtype, device=matrices.device)
        + prefactor1 * matrices[~stable]
        + prefactor2 * skew_sq[~stable]
    )

    return matrices.view(batch_dim + (3, 3))


@torch.jit.script
def cu2ho(cu: torch.Tensor) -> torch.Tensor:
    """
    Converts cubochoric vector representation to homochoric vector representation.

    Args:
        cu: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Homochoric vectors as tensor of shape (..., 3).
    """

    # Sort components
    indices = torch.argsort(torch.abs(cu), dim=-1, descending=False)
    sorted = torch.gather(cu, -1, indices)
    s, m, b = sorted.unbind(dim=-1)

    # Calculate trigonometric argument and avoid indeterminate forms
    trig_arg_xy = s * torch.pi / (12.0 * m)
    trig_arg_xy[torch.isnan(trig_arg_xy)] = 0

    factor_xy = (
        2 ** (1 / 12)
        * 3 ** (1 / 3)
        * m
        * torch.sqrt(
            (
                4 * b**2 * (torch.cos(trig_arg_xy) - (2**0.5))
                + (2**0.5) * m**2 * (-2 * (2**0.5) * torch.cos(trig_arg_xy) + 3)
            )
            / (torch.cos(trig_arg_xy) - (2**0.5))
        )
        / (torch.pi ** (1 / 3) * b * torch.sqrt(-torch.cos(trig_arg_xy) + (2**0.5)))
    )

    # Compute x_s3, y_s3, and z_s3 using the provided equations
    x_s3 = factor_xy * torch.sin(trig_arg_xy)
    y_s3 = factor_xy * 2**-0.5 * ((2**0.5) * torch.cos(trig_arg_xy) - 1)
    z_s3 = (
        2 * 6 ** (1 / 3) * b**2 * (torch.cos(trig_arg_xy) - (2**0.5))
        + 2 ** (5 / 6)
        * 3 ** (1 / 3)
        * m**2
        * (-2 * (2**0.5) * torch.cos(trig_arg_xy) + 3)
    ) / (2 * torch.pi ** (1 / 3) * b * (torch.cos(trig_arg_xy) - (2**0.5)))

    # Reassemble the vector and undo the argsort
    ho = torch.stack((x_s3, y_s3, z_s3), dim=-1)
    ho = torch.scatter(ho, -1, indices, ho)

    # replace any nans with 0
    ho[torch.isnan(ho)] = 0

    # copy the sign of the original cubochoric vector
    ho.copysign_(cu)

    return ho


@torch.jit.script
def ho2cu(ho: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to cubochoric vector representation.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).

    """

    # inverse steps in reverse order of cu2ho
    # start with an argsort on the magnitudes
    indices = torch.argsort(torch.abs(ho), dim=-1, descending=False)
    sorted = torch.gather(ho, -1, indices)
    x_s3, y_s3, z_s3 = torch.abs(sorted).unbind(dim=-1)

    # step 3 inverse
    r_s = torch.norm(ho, dim=-1, keepdim=False)
    prefactor_xy_s3 = torch.sqrt(2 * r_s / (r_s + z_s3))
    x_s2 = x_s3 * prefactor_xy_s3
    y_s2 = y_s3 * prefactor_xy_s3
    z_s2 = (torch.pi / 6) ** 0.5 * r_s

    # # step 2 inverse from Appendix A eq (29)
    # prefactor_xy_s2 = (torch.pi / 6)**0.5 * torch.sqrt((x_s2**2 + 2 * y_s2**2) * (x_s2**2 + y_s2**2)) / (
    #     (2 ** 0.5) * torch.sqrt(x_s2**2 + 2 * y_s2**2 - (torch.abs(y_s2) * torch.sqrt(x_s2**2 + 2 * y_s2**2)))
    # )
    # the above equation in the publication can be dramatically simplified if you assume x and y are positive:
    # ((x^2 + 2 * y^2) * (x^2 + y^2)) / (x^2 + 2 * y^2 - |y| * sqrt(x^2 + 2*y^2)) is
    # the same as:
    # x^2 + y*(sqrt(x^2 + 2*y^2) + 2*y)
    # for x and y positive
    # these are the inverse squircle functions
    # this also avoids an annoying 0/0 case that slows down the calculation
    prefactor_xy_s2 = (
        (torch.pi / 6) ** 0.5
        * torch.sqrt(x_s2**2 + y_s2 * (torch.sqrt(x_s2**2 + 2 * y_s2**2) + 2 * y_s2))
        / (2**0.5)
    )
    # z_s1 is unchanged from z_s2 while x_s1 and y_s1 are found from inverse squircle function
    x_s1 = (prefactor_xy_s2 * 12.0 * torch.sign(x_s2) / torch.pi) * (
        torch.arccos(
            (
                (x_s2**2 + y_s2 * torch.sqrt(x_s2**2 + 2 * y_s2**2))
                / ((2**0.5) * (x_s2**2 + y_s2**2))
            ).clamp_(-1.0, 1.0)
        )
    )
    y_s1 = prefactor_xy_s2 * torch.sign(y_s2)

    # undo the argsort with an in-place scatter
    cu = torch.empty_like(ho)
    cu.scatter_(-1, indices, torch.stack((x_s1, y_s1, z_s2), dim=-1))
    cu /= (torch.pi / 6) ** (1 / 6)

    # copy the sign of the original homochoric vector
    cu.copysign_(ho)

    # replace any nans with 0
    cu[torch.isnan(cu)] = 0

    return cu


@torch.jit.script
def ho2ax(ho: Tensor, fast: bool = True) -> Tensor:
    """
    Converts a set of homochoric vectors to axis-angle representation.

    Args:
        ho (Tensor): shape (..., 3) homochoric coordinates (x, y, z)
        fast (bool): by default skip Newton iteration for FP64 only

    Returns:
        torch.Tensor: shape (..., 4) axis-angles (x, y, z, angle)


    Notes:

    These are Chebyshev fits on the modified homochoric inverse
    fitted by Zachary Varley on 05/24/2024. The modified homochoric
    inverse ties the square of the homochoric vector back to the
    cosine of the half rotation angle.

    """
    if ho.dtype == torch.float32 or ho.dtype == torch.float16:
        fit_parameters = torch.tensor(
            [
                # 8 terms to reach FP32 machine eps
                1.0000000000000009e00,
                -4.9999943403867775e-01,
                -2.5015165060149020e-02,
                -3.8120131548551729e-03,
                -1.2106188330642162e-03,
                4.9329295993155416e-04,
                -7.0089385526450620e-04,
                3.0979774923589078e-04,
                -7.3023474963298843e-05,
            ],
            dtype=ho.dtype,
            device=ho.device,
        ).to(ho.dtype)

    else:
        # 10 term loss mean abs error 5e-10 instead of 15 term's 1e-11
        # 1 iteration of Newton's method is needed for double precision
        # machine error... 20 terms somehow is stuck at 1e-11 error
        fit_parameters = torch.tensor(
            [
                # 10 term polyfit
                1.0000000000000000e00,
                -4.9999997124013285e-01,
                -2.5001181866044025e-02,
                -3.9144209820521038e-03,
                -8.9320268104539483e-04,
                3.1181024286083695e-05,
                -4.3961032788396477e-04,
                3.9657471727506439e-04,
                -2.6379945050586932e-04,
                9.1185355979587159e-05,
                -1.4875867805692529e-05,
            ],
            dtype=ho.dtype,
            device=ho.device,
        ).to(ho.dtype)

    # ho_norm_sq = torch.sum(ho**2, dim=-1, keepdim=True)
    # # makes out of memory error doing all at once
    # s = torch.zeros_like(ho_norm_sq[..., 0])
    # for i in range(len(fit_parameters)):
    #     s += fit_parameters[i] * ho_norm_sq[..., 0] ** i

    ho_norm_sq = torch.sum(ho**2, dim=-1, keepdim=False)
    # makes out of memory error doing all at once
    s = torch.zeros_like(ho_norm_sq)
    for i in range(len(fit_parameters)):
        s += fit_parameters[i] * ho_norm_sq**i

    if ho.dtype == torch.float64 and not fast:
        w = 2 * torch.arccos(torch.clamp(s, -1.0, 1.0))
        # do 1 iteration of Newton's method
        f_w = ((3 / 4) * (w - torch.sin(w))) ** (1 / 3) - torch.sqrt(ho_norm_sq)
        f_p_w = (1 - torch.cos(w)) / (6 ** (2 / 3) * (w - torch.sin(w)) ** (2 / 3))
        update = f_w / f_p_w
        # remove any nans
        update[torch.isnan(update)] = 0
        w -= update
    else:
        w = 2.0 * torch.arccos(torch.clamp(s, -1.0, 1.0))

    ax = torch.concat(
        [
            ho * torch.rsqrt(ho_norm_sq).unsqueeze(-1),
            w.unsqueeze(-1),
        ],
        dim=-1,
    )
    rot_is_identity = torch.abs(ho_norm_sq) < 1e-6
    # set the identity rotation
    ax[rot_is_identity] = 0
    ax[rot_is_identity, ..., 2] = 1.0
    return ax


@torch.jit.script
def ho2ax_reference(ho: Tensor, coeffs: str = "kikuchipy") -> Tensor:
    """
    Converts a set of homochoric vectors to axis-angle representation.

    I have seen two polynomial fits for this conversion, one from EMsoft
    and the other from Kikuchipy. The Kikuchipy one is used here.


    Args:
        ho (Tensor): shape (..., 3) homochoric coordinates (x, y, z)

    Returns:
        torch.Tensor: shape (..., 4) axis-angles (x, y, z, angle)


    Notes:

    f(w) = [(3/4) * (w - sin(w))]^(1/3) -> no inverse -> polynomial fit it

    """
    if coeffs == "kikuchipy":
        fit_parameters = torch.tensor(
            [
                # Kikuchipy polyfit coeffs
                1.0000000000018852,
                -0.5000000002194847,
                -0.024999992127593126,
                -0.003928701544781374,
                -0.0008152701535450438,
                -0.0002009500426119712,
                -0.00002397986776071756,
                -0.00008202868926605841,
                0.00012448715042090092,
                -0.0001749114214822577,
                0.0001703481934140054,
                -0.00012062065004116828,
                0.000059719705868660826,
                -0.00001980756723965647,
                0.000003953714684212874,
                -0.00000036555001439719544,
            ],
            dtype=ho.dtype,
            device=ho.device,
        ).to(ho.dtype)

    elif coeffs == "EMsoft":
        fit_parameters = torch.tensor(
            [
                # EMsoft polyfit coeffs
                0.9999999999999968,
                -0.49999999999986866,
                -0.025000000000632055,
                -0.003928571496460683,
                -0.0008164666077062752,
                -0.00019411896443261646,
                -0.00004985822229871769,
                -0.000014164962366386031,
                -1.9000248160936107e-6,
                -5.72184549898506e-6,
                7.772149920658778e-6,
                -0.00001053483452909705,
                9.528014229335313e-6,
                -5.660288876265125e-6,
                1.2844901692764126e-6,
                1.1255185726258763e-6,
                -1.3834391419956455e-6,
                7.513691751164847e-7,
                -2.401996891720091e-7,
                4.386887017466388e-8,
                -3.5917775353564864e-9,
            ],
            dtype=ho.dtype,
            device=ho.device,
        ).to(ho.dtype)
    else:
        raise ValueError(f"Invalid fit parameters {coeffs}.")

    ho_norm_sq = torch.sum(ho**2, dim=-1, keepdim=True)

    # makes out of memory error doing all at once
    s = torch.zeros_like(ho_norm_sq[..., 0])
    for i in range(len(fit_parameters)):
        s += fit_parameters[i] * ho_norm_sq[..., 0] ** i

    ax = torch.empty(ho.shape[:-1] + (4,), dtype=ho.dtype, device=ho.device)

    rot_is_identity = torch.abs(ho_norm_sq.squeeze(-1)) < 1e-8
    ax[rot_is_identity, 0:1] = 0.0
    ax[rot_is_identity, 1:2] = 0.0
    ax[rot_is_identity, 2:3] = 1.0

    ax[~rot_is_identity, :3] = ho[~rot_is_identity, :] * torch.rsqrt(
        ho_norm_sq[~rot_is_identity]
    )
    ax[..., 3] = torch.where(
        ~rot_is_identity,
        2.0 * torch.arccos(torch.clamp(s, -1.0, 1.0)),
        0,
    )
    return ax


@torch.jit.script
def ho2ax_newton(ho: Tensor) -> Tensor:
    """
    Converts homochoric coordinates to axis-angle representation.

    Args:
        ho (Tensor): shape (..., 3) homochoric coordinates (x, y, z)

    Returns:
        torch.Tensor: shape (..., 4) axis-angles (x, y, z, angle)

    Notes:
        Newton's method

    """

    # initial guess for ang given h
    h = torch.norm(ho, dim=-1)

    # where zero return zero
    mask_zero = h == 0
    ax = torch.empty(ho.shape[:-1] + (4,), dtype=ho.dtype, device=ho.device)
    ax[mask_zero, 0] = 0.0
    ax[mask_zero, 1] = 0.0
    ax[mask_zero, 2] = 1.0
    ax[mask_zero, 3] = 0.0

    # Newton's method
    # initial guess for w given h is an inverted Pade approximation
    w_newton = (15 - torch.sqrt(225 - 60 * h[~mask_zero] ** 2)) / h[~mask_zero]

    # Newton's method
    for _ in range(2 if h.dtype == torch.float32 else 3):
        f_w = ((3 / 4) * (w_newton - torch.sin(w_newton))) ** (1 / 3) - h[~mask_zero]
        f_p_w = (1 - torch.cos(w_newton)) / (
            6 ** (2 / 3) * (w_newton - torch.sin(w_newton)) ** (2 / 3)
        )
        update = f_w / f_p_w
        # remove any nans
        update[torch.isnan(update)] = 0
        w_newton -= update

    ax[~mask_zero, 0:3] = ho[~mask_zero] * torch.rsqrt(h[~mask_zero].unsqueeze(-1))
    ax[~mask_zero, 3] = w_newton

    return ax


@torch.jit.script
def ax2ho(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to homochoric vector representation.

    Args:
        ax: Axis-angle representation as tensor of shape (..., 4).

    Returns:
        Homochoric vectors as tensor of shape (..., 3).

    """
    return (0.75 * (ax[..., 3:4] - torch.sin(ax[..., 3:4]))) ** (1.0 / 3.0) * ax[
        ..., :3
    ]


@torch.jit.script
def ax2ro(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to Rodrigues vector representation.

    Args:
        ax (Tensor): shape (..., 4) axis-angle (x, y, z, angle)

    Returns:
        torch.Tensor: shape (..., 4) Rodrigues-Frank (x, y, z, tan(angle/2))
    """
    ro = ax.clone()
    ro[..., 3] = torch.tan(ax[..., 3] / 2)
    return ro


@torch.jit.script
def ro2ax(ro: Tensor) -> Tensor:
    """
    Converts a rotation vector to an axis-angle representation.

    Args:
        ro (Tensor): shape (..., 4) Rodrigues-Frank (x, y, z, tan(angle/2)).

    Returns:
        torch.Tensor: shape (..., 4) axis-angles (x, y, z, angle).
    """
    ax = torch.empty_like(ro)
    mask_zero_ro = torch.abs(ro[..., 3]) == 0
    ax[mask_zero_ro] = torch.tensor([0, 0, 1, 0], dtype=ro.dtype, device=ro.device)

    mask_inf_ro = torch.isinf(ro[..., 3])
    ax[mask_inf_ro, :3] = ro[mask_inf_ro, :3]
    ax[mask_inf_ro, 3] = torch.pi

    mask_else = ~(mask_zero_ro | mask_inf_ro)
    ax[mask_else, :3] = ro[mask_else, :3] / torch.norm(
        ro[mask_else, :3], dim=-1, keepdim=True
    )
    ax[mask_else, 3] = 2 * torch.atan(ro[mask_else, 3])
    return ax


@torch.jit.script
def ax2qu(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to quaternion representation.

    Args:
        ax (Tensor): shape (..., 4) axis-angle in the format (x, y, z, angle).

    Returns:
        torch.Tensor: shape (..., 4) quaternions in the format (w, x, y, z).
    """
    qu = torch.empty_like(ax)
    cos_half_ang = torch.cos(ax[..., 3] / 2.0)
    sin_half_ang = torch.sin(ax[..., 3:4] / 2.0)
    qu[..., 0] = cos_half_ang
    qu[..., 1:] = ax[..., :3] * sin_half_ang
    return qu


@torch.jit.script
def qu2ax(qu: Tensor) -> Tensor:
    """
    Converts quaternion representation to axis-angle representation.

    Args:
        qu (Tensor): shape (..., 4) quaternions in the format (w, x, y, z).

    Returns:
        torch.Tensor: shape (..., 4) axis-angle in the format (x, y, z, angle).
    """

    ax = torch.empty_like(qu)
    angle = 2 * torch.acos(torch.clamp(qu[..., 0], min=-1.0, max=1.0))

    s = torch.where(
        qu[..., 0:1] != 0,
        torch.sign(qu[..., 0:1]) / torch.norm(qu[..., 1:], dim=-1, keepdim=True),
        1.0,
    )

    ax[..., :3] = qu[..., 1:] * s
    ax[..., 3] = angle

    # fix identity quaternions to be about z axis
    mask_identity = angle == 0.0
    ax[mask_identity, 0] = 0.0
    ax[mask_identity, 1] = 0.0
    ax[mask_identity, 2] = 1.0
    ax[mask_identity, 3] = 0.0

    return ax


@torch.jit.script
def qu2ro(qu: Tensor) -> Tensor:
    """
    Converts quaternion representation to Rodrigues-Frank vector representation.

    Args:
        qu: shape (..., 4) quaternions in the format (w, x, y, z).

    Returns:
        Tensor: shape (..., 4) Rodrigues-Frank (x, y, z, tan(angle/2))
    """
    ro = torch.empty_like(qu)

    # Handle general case
    ro[..., :3] = qu[..., 1:] * torch.rsqrt(
        torch.sum(qu[..., 1:] ** 2, dim=-1, keepdim=True)
    )
    ro[..., 3] = torch.tan(torch.acos(torch.clamp(qu[..., 0], min=-1.0, max=1.0)))

    # w < 1e-8 for float32 / w < 1e-10 for float64 -> infinite tan
    eps = 1e-8 if qu.dtype == torch.float32 else 1e-10
    mask_zero = torch.abs(qu[..., 0]) < eps
    ro[mask_zero, 3] = float("inf")
    return ro


@torch.jit.script
def qu2bu(qu: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to Bunge angles (ZXZ Euler angles).

    Args:
        qu (Tensor): shape (..., 4) quaternions in the format (w, x, y, z).

    Returns:
        torch.Tensor: shape (..., 3) Bunge angles in radians.
    """

    bu = torch.empty(qu.shape[:-1] + (3,), dtype=qu.dtype, device=qu.device)

    q03 = qu[..., 0] ** 2 + qu[..., 3] ** 2
    q12 = qu[..., 1] ** 2 + qu[..., 2] ** 2
    chi = torch.sqrt((q03 * q12))

    mask_chi_zero = chi == 0
    mA = (mask_chi_zero) & (q12 == 0)
    mB = (mask_chi_zero) & (q03 == 0)
    mC = ~mask_chi_zero

    bu[mA, 0] = torch.atan2(-2 * qu[mA, 0] * qu[mA, 3], qu[mA, 0] ** 2 - qu[mA, 3] ** 2)
    bu[mA, 1] = 0
    bu[mA, 2] = 0

    bu[mB, 0] = torch.atan2(2 * qu[mB, 1] * qu[mB, 2], qu[mB, 1] ** 2 - qu[mB, 2] ** 2)
    bu[mB, 1] = torch.pi
    bu[mB, 2] = 0

    bu[mC, 0] = torch.atan2(
        (qu[mC, 1] * qu[mC, 3] - qu[mC, 0] * qu[mC, 2]) / chi[mC],
        (-qu[mC, 0] * qu[mC, 1] - qu[mC, 2] * qu[mC, 3]) / chi[mC],
    )
    bu[mC, 1] = torch.atan2(2 * chi[mC], q03[mC] - q12[mC])
    bu[mC, 2] = torch.atan2(
        (qu[mC, 0] * qu[mC, 2] + qu[mC, 1] * qu[mC, 3]) / chi[mC],
        (qu[mC, 2] * qu[mC, 3] - qu[mC, 0] * qu[mC, 1]) / chi[mC],
    )

    # add 2pi to negative angles for first and last angles
    bu[..., 0] = torch.where(bu[..., 0] < 0, bu[..., 0] + 2 * torch.pi, bu[..., 0])
    bu[..., 2] = torch.where(bu[..., 2] < 0, bu[..., 2] + 2 * torch.pi, bu[..., 2])

    return bu


@torch.jit.script
def bu2qu(bu: Tensor) -> Tensor:
    """
    Convert rotations given as Bunge angles (ZXZ Euler angles) to quaternions.

    Args:
        bu (Tensor): shape (..., 3) Bunge angles in radians.

    Returns:
        torch.Tensor: shape (..., 4) quaternions in the format (w, x, y, z).
    """
    qu = torch.empty(bu.shape[:-1] + (4,), dtype=bu.dtype, device=bu.device)

    sigma = 0.5 * (bu[..., 0] + bu[..., 2])
    delta = 0.5 * (bu[..., 0] - bu[..., 2])

    c = torch.cos(0.5 * bu[..., 1])
    s = torch.sin(0.5 * bu[..., 1])

    qu[..., 0] = c * torch.cos(sigma)
    qu[..., 1] = -s * torch.cos(delta)
    qu[..., 2] = -s * torch.sin(delta)
    qu[..., 3] = -c * torch.sin(sigma)

    # correct for negative real part of quaternion
    return qu * torch.where(qu[..., 0] < 0, -1, 1).unsqueeze(-1)


@torch.jit.script
def qu2cl(qu: Tensor) -> Tensor:
    """
    Convert rotations given as unit quaternions to Clifford Torus coordinates.
    The coordinates are in the format (X, Z_y, Y, X_z, Z, Y_x)

    Args:
        qu (Tensor): shape (..., 4) quaternions in the format (w, x, y, z).

    Returns:
        torch.Tensor: shape (..., 6) Clifford Torus coordinates.

    """

    cl = torch.empty(qu.shape[:-1] + (6,), dtype=qu.dtype, device=qu.device)

    cl[..., 0] = torch.atan(qu[..., 1] / qu[..., 0])
    cl[..., 1] = torch.atan(qu[..., 3] / qu[..., 2])
    cl[..., 2] = torch.atan(qu[..., 2] / qu[..., 0])
    cl[..., 3] = torch.atan(qu[..., 1] / qu[..., 3])
    cl[..., 4] = torch.atan(qu[..., 3] / qu[..., 0])
    cl[..., 5] = torch.atan(qu[..., 2] / qu[..., 1])

    return cl


@torch.jit.script
def qu2om(qu: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: Tensor of quaternions (real part first) of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    q_bar = qu[..., 0] ** 2 - torch.sum(qu[..., 1:] ** 2, dim=-1)
    matrix = torch.empty(qu.shape[:-1] + (3, 3), dtype=qu.dtype, device=qu.device)

    matrix[..., 0, 0] = q_bar + 2 * qu[..., 1] ** 2
    matrix[..., 0, 1] = 2 * (qu[..., 1] * qu[..., 2] - qu[..., 0] * qu[..., 3])
    matrix[..., 0, 2] = 2 * (qu[..., 1] * qu[..., 3] + qu[..., 0] * qu[..., 2])

    matrix[..., 1, 0] = 2 * (qu[..., 2] * qu[..., 1] + qu[..., 0] * qu[..., 3])
    matrix[..., 1, 1] = q_bar + 2 * qu[..., 2] ** 2
    matrix[..., 1, 2] = 2 * (qu[..., 2] * qu[..., 3] - qu[..., 0] * qu[..., 1])

    matrix[..., 2, 0] = 2 * (qu[..., 3] * qu[..., 1] - qu[..., 0] * qu[..., 2])
    matrix[..., 2, 1] = 2 * (qu[..., 3] * qu[..., 2] + qu[..., 0] * qu[..., 1])
    matrix[..., 2, 2] = q_bar + 2 * qu[..., 3] ** 2

    return matrix


@torch.jit.script
def zh2om(zh: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = zh[..., :3], zh[..., 3:]
    b1 = a1 / torch.norm(a1, p=2, dim=-1, keepdim=True)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, p=2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    b3 = b3 / torch.norm(b3, p=2, dim=-1, keepdim=True)
    return torch.stack((b1, b2, b3), dim=-2)


@torch.jit.script
def om2zh(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


@torch.jit.script
def zh2qu(zh: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to quaternion
    representation.

    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of quaternions of size (*, 4)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035

    """

    return om2qu(zh2om(zh))


@torch.jit.script
def qu2zh(quaternions: Tensor) -> Tensor:
    """
    Converts quaternion representation to 6D rotation representation by Zhou et al.

    Args:
        quaternions: batch of quaternions of size (*, 4)

    Returns:
        6D rotation representation, of size (*, 6)
    """

    return om2zh(qu2om(quaternions))


@torch.jit.script
def qu2cu(quaternions: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to cubochoric vectors.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return ho2cu(qu2ho(quaternions))


@torch.jit.script
def om2ho(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to homochoric vector representation.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Homochoric vector representation as tensor of shape (..., 3).
    """
    return ax2ho(om2ax(matrix))


@torch.jit.script
def ax2cu(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to cubochoric vector representation.

    Args:
        ax: Axis-angle representation as tensor of shape (..., 4).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return qu2cu(ax2qu(ax))


@torch.jit.script
def ro2qu(ro: Tensor) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to quaternions.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return ax2qu(ro2ax(ro))


@torch.jit.script
def ro2om(ro: Tensor) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to rotation matrices.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return ax2om(ro2ax(ro))


@torch.jit.script
def ro2cu(ro: Tensor) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to cubochoric vectors.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return ax2cu(ro2ax(ro))


@torch.jit.script
def ro2ho(ro: Tensor) -> Tensor:
    """
    Converts Rodrigues-Frank vector representation to homochoric vectors.

    Args:
        ro: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        Homochoric vectors as tensor of shape (..., 3).
    """
    return ax2ho(ro2ax(ro))


@torch.jit.script
def cu2ax(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to axis-angle representation.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Axis-angle representation as tensor of shape (..., 4).
    """
    return ho2ax(cu2ho(cubochoric_vectors))


@torch.jit.script
def cu2qu(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to quaternions.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return ax2qu(ho2ax(cu2ho(cubochoric_vectors)))


@torch.jit.script
def cu2om(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to rotation matrices.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return ax2om(ho2ax(cu2ho(cubochoric_vectors)))


@torch.jit.script
def cu2ro(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to Rodrigues-Frank vector representation.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return ax2ro(ho2ax(cu2ho(cubochoric_vectors)))


@torch.jit.script
def ho2qu(homochoric_vectors: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to quaternions.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return ax2qu(ho2ax(homochoric_vectors))


@torch.jit.script
def ho2om(homochoric_vectors: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to rotation matrices.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return ax2om(ho2ax(homochoric_vectors))


@torch.jit.script
def ho2ro(homochoric_vectors: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to Rodrigues-Frank vector representation.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return ax2ro(ho2ax(homochoric_vectors))


@torch.jit.script
def om2ro(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to Rodrigues-Frank vector representation.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return qu2ro(om2qu(matrix))


@torch.jit.script
def om2cu(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to cubochoric vector representation.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Cubochoric vector representation as tensor of shape (..., 3).
    """
    return qu2cu(om2qu(matrix))


@torch.jit.script
def bu2om(bunge_angles: Tensor) -> Tensor:
    """
    Convert rotations given as Bunge angles (ZXZ Euler angles) to rotation matrices.

    Args:
        bunge_angles: Bunge angles in radians as tensor of shape (..., 3).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return qu2om(bu2qu(bunge_angles))


@torch.jit.script
def bu2ax(bunge_angles: Tensor) -> Tensor:
    """
    Convert rotations given as Bunge angles (ZXZ Euler angles) to axis-angle representation.

    Args:
        bunge_angles: Bunge angles in radians as tensor of shape (..., 3).

    Returns:
        Axis-angle representation as tensor of shape (..., 4).
    """
    return qu2ax(bu2qu(bunge_angles))


@torch.jit.script
def bu2ro(bunge_angles: Tensor) -> Tensor:
    """
    Convert rotations given as Bunge angles (ZXZ Euler angles) to Rodrigues-Frank vector representation.

    Args:
        bunge_angles: Bunge angles in radians as tensor of shape (..., 3).

    Returns:
        Rodrigues-Frank vector representation as tensor of shape (..., 4).
    """
    return qu2ro(bu2qu(bunge_angles))


@torch.jit.script
def bu2cu(bunge_angles: Tensor) -> Tensor:
    """
    Convert rotations given as Bunge angles (ZXZ Euler angles) to cubochoric vector representation.

    Args:
        bunge_angles: Bunge angles in radians as tensor of shape (..., 3).

    Returns:
        Cubochoric vector representation as tensor of shape (..., 3).
    """
    return qu2cu(bu2qu(bunge_angles))


@torch.jit.script
def bu2ho(bunge_angles: Tensor) -> Tensor:
    """
    Convert rotations given as Bunge angles (ZXZ Euler angles) to homochoric vector representation.

    Args:
        bunge_angles: Bunge angles in radians as tensor of shape (..., 3).

    Returns:
        Homochoric vector representation as tensor of shape (..., 3).
    """
    return qu2ho(bu2qu(bunge_angles))


@torch.jit.script
def om2bu(matrix: Tensor) -> Tensor:
    """
    Convert rotations given as rotation matrices to Bunge angles (ZXZ Euler angles).

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Bunge angles in radians as tensor of shape (..., 3).
    """
    return qu2bu(om2qu(matrix))


@torch.jit.script
def ax2bu(axis_angle: Tensor) -> Tensor:
    """
    Convert rotations given as axis-angle representation to Bunge angles (ZXZ Euler angles).

    Args:
        axis_angle: Axis-angle representation as tensor of shape (..., 4).

    Returns:
        Bunge angles in radians as tensor of shape (..., 3).
    """
    return qu2bu(ax2qu(axis_angle))


@torch.jit.script
def ro2bu(rodrigues_frank: Tensor) -> Tensor:
    """
    Convert rotations given as Rodrigues-Frank vector representation to Bunge angles (ZXZ Euler angles).

    Args:
        rodrigues_frank: Rodrigues-Frank vector representation as tensor of shape (..., 4).

    Returns:
        Bunge angles in radians as tensor of shape (..., 3).
    """
    return qu2bu(ro2qu(rodrigues_frank))


@torch.jit.script
def cu2bu(cubochoric_vectors: Tensor) -> Tensor:
    """
    Convert rotations given as cubochoric vectors to Bunge angles (ZXZ Euler angles).

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Bunge angles in radians as tensor of shape (..., 3).
    """
    return qu2bu(cu2qu(cubochoric_vectors))


@torch.jit.script
def ho2bu(homochoric_vectors: Tensor) -> Tensor:
    """
    Convert rotations given as homochoric vectors to Bunge angles (ZXZ Euler angles).

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        Bunge angles in radians as tensor of shape (..., 3).
    """
    return qu2bu(ho2qu(homochoric_vectors))


"""

The Rosca-Lambert projection is a bijection between the unit sphere and the
square, scaled here to be between (-1, 1) X (-1, 1) for compatibility with the
PyTorch function "grid_sample".

For more information, see: "Roşca, D., 2010. New uniform grids on the sphere.
Astronomy & Astrophysics, 520, p.A63."

"""


@torch.jit.script
def theta_phi_to_xyz(theta: Tensor, phi: Tensor) -> Tensor:
    """
    Convert spherical coordinates to cartesian coordinates.

    Args:
        theta (Tensor): shape (..., ) of polar declination angles
        phi (Tensor): shape (..., ) of azimuthal angles

    Returns:
        Tensor: torch tensor of shape (..., 3) containing the cartesian
        coordinates
    """
    return torch.stack(
        (
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ),
        dim=1,
    )


@torch.jit.script
def xyz_to_theta_phi(xyz: Tensor) -> Tensor:
    """
    Convert cartesian coordinates to latitude and longitude.

    Args:
        xyz (Tensor): torch tensor of shape (..., 3) of cartesian coordinates

    Returns:
        Tensor: torch tensor of shape (..., 2) of declination from z-axis and
        azimuthal angle

    """
    return torch.stack(
        (
            torch.atan2(torch.norm(xyz[:, :2], dim=1), xyz[:, 2]),
            torch.atan2(xyz[:, 1], xyz[:, 0]),
        ),
        dim=1,
    )


@torch.jit.script
def rosca_lambert(pts: Tensor) -> Tensor:
    """
    Map unit sphere to (-1, 1) X (-1, 1) square via square Rosca-Lambert projection.

    Args:
        pts: torch tensor of shape (..., 3) containing the points
    Returns:
        torch tensor of shape (..., 2) containing the projected points
    """
    # x-axis and y-axis on the plane are labeled a and b
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]

    # floating point normalization to sphere can yield above 1.0:
    # --------------------------------------------,,,--
    # xyz = [1.7817265e-04, 2.8403841e-05, 1.0000001e0]
    # --------------------------------------------^^^--
    # xyz /= (x**2 + y**2 + z**2).sqrt() -> xyz (identical to input)
    # so we have to clamp to avoid sqrt of negative number
    factor = torch.sqrt(torch.clamp((1.0 - torch.abs(z)), min=0.0))

    cond = torch.abs(y) <= torch.abs(x)
    big = torch.where(cond, x, y)
    sml = torch.where(cond, y, x)
    simpler_term = torch.where(big < 0, -1, 1) * factor
    arctan_term = (
        torch.where(big < 0, -1, 1)
        * factor
        * torch.atan2(sml * torch.where(big < 0, -1, 1), torch.abs(big))
        * (4.0 / torch.pi)
    )
    # stack them together but flip the order if the condition is false
    out = torch.stack((simpler_term, arctan_term), dim=-1)
    out = torch.where(cond[..., None], out, out.flip(-1))
    return out


@torch.jit.script
def inv_rosca_lambert(pts: Tensor) -> Tensor:
    """
    Map (-1, 1) X (-1, 1) square to Northern hemisphere via inverse square
    lambert projection.

    Args:
        pts: torch tensor of shape (..., 2) containing the points

    Returns:
        torch tensor of shape (..., 3) containing the projected points

    This version is more efficient than the previous one, as it just plops
    everything into the first quadrant, then swaps the x and y coordinates
    if needed so that we always have x >= y. Then swaps back at the end and
    copy the sign of the original x and y coordinates.

    """
    pi = torch.pi
    # map to first quadrant and swap x and y if needed
    x_abs, y_abs = (
        torch.abs(pts[..., 0]) * (pi / 2) ** 0.5,
        torch.abs(pts[..., 1]) * (pi / 2) ** 0.5,
    )
    cond = x_abs >= y_abs
    x_new = torch.where(cond, x_abs, y_abs)
    y_new = torch.where(cond, y_abs, x_abs)

    # only one case now
    x_hs = (
        (2 * x_new / pi)
        * torch.sqrt(pi - x_new**2)
        * torch.cos(pi * y_new / (4 * x_new))
    )
    y_hs = (
        (2 * x_new / pi)
        * torch.sqrt(pi - x_new**2)
        * torch.sin(pi * y_new / (4 * x_new))
    )
    z_out = 1 - (2 * x_new**2 / pi)

    # swap back and copy sign
    x_out = torch.where(cond, x_hs, y_hs)
    y_out = torch.where(cond, y_hs, x_hs)
    x_out = x_out.copysign_(pts[..., 0])
    y_out = y_out.copysign_(pts[..., 1])

    return torch.stack((x_out, y_out, z_out), dim=-1)


@torch.jit.script
def rosca_lambert_side_by_side(pts: Tensor) -> Tensor:
    """
    Map unit sphere to (-1, 1) X (-1, 1) square via square Rosca-Lambert
    projection. Points with a positive z-coordinate are projected to the left
    side of the square, while points with a negative z-coordinate are projected
    to the right side of the square.

    Args:
        pts: torch tensor of shape (..., 3) containing the points

    Returns:
        torch tensor of shape (..., 2) containing the projected points

    """
    # x-axis and y-axis on the plane are labeled a and b
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]

    # floating point error can yield z above 1.0 example for float32 is:
    # xyz = [1.7817265e-04, 2.8403841e-05, 1.0000001e0]
    # so we have to clamp to avoid sqrt of negative number
    factor = torch.sqrt(torch.clamp(2.0 * (1.0 - torch.abs(z)), min=0.0))

    cond = torch.abs(y) <= torch.abs(x)
    big = torch.where(cond, x, y)
    sml = torch.where(cond, y, x)
    simpler_term = torch.where(big < 0, -1, 1) * factor * (2.0 / (8.0**0.5))
    arctan_term = (
        torch.where(big < 0, -1, 1)
        * factor
        * torch.atan2(sml * torch.where(big < 0, -1, 1), torch.abs(big))
        * (2.0 * (2.0**0.5) / torch.pi)
    )
    # stack them together but flip the order if the condition is false
    out = torch.stack((simpler_term, arctan_term), dim=-1)
    out = torch.where(cond[..., None], out, out.flip(-1))
    # halve the x index for all points then subtract 0.5 to move to [-1, 0]
    # then add 1 to the j coordinate if z is negative to move to [0, 1]
    # note that torch's grid_sample has flipped coordinates from ij indexing
    out[..., 0] = (out[..., 0] / 2.0) - 0.5 + torch.where(z < 0, 1.0, 0)
    return out


@torch.jit.script
def s2_fibonacci(
    n: int,
    device: torch.device,
    mode: str = "avg",
) -> Tensor:
    """
    Sample n points on the unit sphere using the Fibonacci spiral method.

    Args:
        n (int): the number of points to sample
        device (torch.device): the device to use
        mode (str): the mode to use for the Fibonacci lattice. "avg" will
            optimize for average spacing, while "max" will optimize for
            maximum spacing. Default is "avg".

    References:
    https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/


    """
    # initialize the golden ratio
    phi = (1 + 5**0.5) / 2
    # initialize the epsilon parameter
    if mode == "avg":
        epsilon = 0.36
    elif mode == "max":
        if n >= 600000:
            epsilon = 214.0
        elif n >= 400000:
            epsilon = 75.0
        elif n >= 11000:
            epsilon = 27.0
        elif n >= 890:
            epsilon = 10.0
        elif n >= 177:
            epsilon = 3.33
        elif n >= 24:
            epsilon = 1.33
        else:
            epsilon = 0.33
    else:
        raise ValueError('mode must be either "avg" or "max"')
    # generate the points (they must be doubles for large numbers of points)
    indices = torch.arange(n, dtype=torch.float64, device=device)
    theta = 2 * torch.pi * indices / phi
    phi = torch.acos(1 - 2 * (indices + epsilon) / (n - 1 + 2 * epsilon))
    points = theta_phi_to_xyz(theta, phi)
    return points


@torch.jit.script
def so3_fibonacci(
    n: int,
    device: torch.device,
) -> Tensor:
    """
    Super Fibonacci sampling of orientations.

    See the following paper for more information:

    Alexa, Marc. "Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling of SO
    (3)." In Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition, pp. 8291-8300. 2022.

    Args:
        n (int): the number of orientations to sample
        device (torch.device): the device to use

    Returns:
        Tensor: the 3D super Fibonacci sampling quaternions (n, 4)

    """

    PHI = 2.0**0.5
    # positive real solution to PSI^4 = PSI + 4
    PSI = 1.533751168755204288118041

    # don't use float32 for large numbers of points
    indices = torch.arange(n, device=device, dtype=torch.float64)
    s = indices + 0.5
    t = s / n
    d = 2 * torch.pi * s
    r = torch.sqrt(t)
    R = torch.sqrt(1 - t)
    alpha = d / PHI
    beta = d / PSI
    qu = torch.stack(
        [
            r * torch.sin(alpha),
            r * torch.cos(alpha),
            R * torch.sin(beta),
            R * torch.cos(beta),
        ],
        dim=1,
    )

    return qu


@torch.jit.script
def so3_cu_rand(n: int, device: torch.device) -> Tensor:
    """
    3D random sampling in cubochoric coordinates lifted to SO(3) as quaternions.

    Args:
        n (int): the number of orientations to sample device
        device (torch.device): the device to use

    Returns:
        Tensor: Quaternions of shape (n, 4) in form (w, x, y, z)

    """
    box_sampling = torch.rand(n, 3, device=device) * torch.pi ** (
        2.0 / 3.0
    ) - 0.5 * torch.pi ** (2.0 / 3.0)
    qu = cu2qu(box_sampling)
    qu = qu_std(qu / torch.norm(qu, dim=-1, keepdim=True))
    return qu


@torch.jit.script
def so3_uniform_quat(
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Shoemake's uniformly distributed elements of SO(3) as quaternions. This
    routine includes both quaternion hemispheres and will return quaternions
    with negative real part.

    Args:
        :n (int): the number of orientations to sample
        :device (torch.device): the device to use

    Returns:
        torch.Tensor: the 3D random sampling in cubochoric coordinates (n, 4)


    Notes:

    This function is based on the following work of Ken Shoemake:

    Shoemake, Ken. "Uniform random rotations." Graphics Gems III (IBM Version).
    Morgan Kaufmann, 1992. 124-132.

    """

    # h = ( sqrt(1-u) sin(2πv), sqrt(1-u) cos(2πv), sqrt(u) sin(2πw), sqrt(u) cos(2πw))
    u = torch.rand(n, device=device, dtype=dtype)
    v = torch.rand(n, device=device, dtype=dtype)
    w = torch.rand(n, device=device, dtype=dtype)
    h = torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * torch.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * torch.pi * v),
            torch.sqrt(u) * torch.sin(2 * torch.pi * w),
            torch.sqrt(u) * torch.cos(2 * torch.pi * w),
        ],
        dim=1,
    )
    return h


@torch.jit.script
def so3_cubochoric_grid(
    edge_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Generate a 3D grid sampling in cubochoric coordinates. Orientations
    are returned as unit quaternions with positive scalar part (w, x, y, z)

    Args:
        :edge_length (int): the number of points along each axis of the cube
        :device (torch.device): the device to use

    Returns:
        torch.Tensor: the 3D grid sampling in cubochoric coordinates (n, 3)

    """
    cu = torch.linspace(
        -0.5 * torch.pi ** (2.0 / 3.0),
        0.5 * torch.pi ** (2.0 / 3.0),
        edge_length + 1,  # add extra point at the opposite faces
        dtype=dtype,
        device=device,
    )
    # remove the last point
    cu = cu[:-1]

    cu = torch.stack(torch.meshgrid(cu, cu, cu, indexing="ij"), dim=-1).reshape(-1, 3)
    qu = cu2qu(cu)
    qu = qu_std(qu)
    return qu


"""
PCA via Direct Batched Covariance Matrix Updates
-----------------------------

This file contains a batched streamed implementation of PCA based on Welford's
online algorithm extended by Chan et al for covariance updates:

Welford, B. P. "Note on a method for calculating corrected sums of squares and
products." Technometrics 4.3 (1962): 419-420.

Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Updating formulae and a
pairwise algorithm for computing sample variances." COMPSTAT 1982 5th Symposium
held at Toulouse 1982: Part I: Proceedings in Computational Statistics.
Physica-Verlag HD, 1982.

This works fine for low-dimensional data.

"""


@torch.jit.script
def update_covmat(
    current_covmat: Tensor,
    current_obs: Tensor,
    current_mean: Tensor,
    data_new: Tensor,
    delta_dtype: torch.dtype,
) -> None:
    """
    Update the covariance matrix and mean using Welford's online algorithm.

    Args:
        current_covmat: current covariance matrix
        current_obs: current number of observations
        current_mean: current mean
        data_new: new data to be included in the covariance matrix

    Returns:
        None
    """
    # compute the batch mean
    N = data_new.shape[0]
    batch_mean = torch.mean(data_new, dim=0, keepdim=True)

    # update the global mean
    new_mean = (current_mean * current_obs + batch_mean * N) / (current_obs + N)

    # compute the deltas
    delta = data_new.to(delta_dtype) - (current_mean).to(delta_dtype)
    delta_prime = data_new.to(delta_dtype) - (new_mean).to(delta_dtype)

    # update the running covariance matrix
    current_covmat += torch.einsum("ij,ik->jk", delta, delta_prime).to(
        current_covmat.dtype
    )

    # update the number of observations and mean
    current_obs += N
    current_mean.copy_(new_mean)


class OnlineCovMatrix(Module):
    """
    Online covariance matrix calculator
    """

    def __init__(
        self,
        n_features: int,
        covmat_dtype: torch.dtype = torch.float32,
        delta_dtype: torch.dtype = torch.float32,
        correlation: bool = False,
    ):
        super(OnlineCovMatrix, self).__init__()
        self.n_features = n_features
        self.covmat_dtype = covmat_dtype
        self.delta_dtype = delta_dtype

        # Initialize
        self.register_buffer("mean", torch.zeros(1, n_features, dtype=covmat_dtype))
        self.register_buffer(
            "covmat_aggregate",
            torch.zeros((n_features, n_features), dtype=covmat_dtype),
        )
        self.register_buffer("obs", torch.tensor([0], dtype=torch.int64))
        self.correlation = correlation

    def forward(self, x: Tensor):
        """
        Update the covariance matrix with new data

        Args:
            x: torch tensor of shape (B, n_features) containing the new data

        Returns:
            None
        """
        # update the covariance matrix
        update_covmat(self.covmat_aggregate, self.obs, self.mean, x, self.delta_dtype)

    def get_covmat(self):
        """
        Get the covariance matrix

        Returns:
            torch tensor of shape (n_features, n_features) containing the covariance matrix
        """
        covmat = self.covmat_aggregate / (self.obs - 1).to(self.covmat_dtype)
        # calculate the correlation matrix
        if self.correlation:
            d_sqrt_inv = 1.0 / torch.sqrt(torch.diag(covmat))
            corr_mat = torch.einsum("ij,i,j->ij", covmat, d_sqrt_inv, d_sqrt_inv)
            return corr_mat
        else:
            return covmat

    def get_eigenvectors(self):
        """
        Get the eigenvectors of the covariance matrix

        Returns:
            torch tensor of shape (n_features, n_features) containing the eigenvectors
        """
        covmat = self.get_covmat()
        _, eigenvectors = torch.linalg.eigh(covmat)
        return eigenvectors


def progressbar(it, prefix="", prefix_min_length=15, size=60, out=sys.stdout):
    """
    Progress bar for iterators.

    Args:
        it (Iterable): The iterator to wrap.
        prefix (str): The prefix to print before the progress bar.
        prefix_min_lenght (int): The minimum length of the prefix.
        size (int): The size of the progress bar in characters.
        out (file): Write destination.

    """
    prefix = "{:<{}}".format(prefix, prefix_min_length)
    count = len(it)
    start = time.time()

    for i, item in enumerate(it):
        yield item
        j = i + 1
        x = int(size * j / count)
        seconds_per_iteration = (time.time() - start) / j
        remaining = (count - j) * seconds_per_iteration

        # remaining
        r_mins, r_sec = divmod(remaining, 60)
        r_time_str = (
            f"{int(r_mins):02}:{r_sec:04.1f}" if r_mins > 0 else f"{r_sec:04.1f}"
        )

        # elapsed
        e_mins, e_sec = divmod(time.time() - start, 60)
        e_time_str = (
            f"{int(e_mins):02}:{e_sec:04.1f}" if e_mins > 0 else f"{e_sec:04.1f}"
        )

        # current iterations per second
        ips = 1.0 / (seconds_per_iteration + 1e-8)
        speed_str = f"{ips:04.1f} itr/s" if ips > 1.0 else f"{1.0/ips:04.1f} s/itr"

        print(
            f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j / count * 100:04.1f}% {j}/{count} {e_time_str}<{r_time_str}, {speed_str}",
            end="\r",
            file=out,
            flush=True,
        )
    print("\n")


"""
Quaternion operators for the Laue groups were taken from the following paper:

Larsen, Peter Mahler, and Søren Schmidt. "Improved orientation sampling for
indexing diffraction patterns of polycrystalline materials." Journal of Applied
Crystallography 50, no. 6 (2017): 1571-1582.

"""


@torch.jit.script
def get_laue_mult(laue_group: int) -> int:
    """
    Multiplicity of a given Laue group (including inversion):

    1) Laue C1       Triclinic: 1-, 1
    2) Laue C2      Monoclinic: 2/m, m, 2
    3) Laue D2    Orthorhombic: mmm, mm2, 222
    4) Laue C4  Tetragonal low: 4/m, 4-, 4
    5) Laue D4 Tetragonal high: 4/mmm, 4-2m, 4mm, 422
    6) Laue C3    Trigonal low: 3-, 3
    7) Laue D3   Trigonal high: 3-m, 3m, 32
    8) Laue C6   Hexagonal low: 6/m, 6-, 6
    9) Laue D6  Hexagonal high: 6/mmm, 6-m2, 6mm, 622
    10) Laue T       Cubic low: m3-, 23
    11) Laue O      Cubic high: m3-m, 4-3m, 432

    Args:
        laue_group: integer between 1 and 11 inclusive

    Returns:
        integer containing the multiplicity of the Laue group

    """
    LAUE_MULTS = [
        2,  #   1 - Triclinic
        4,  #   2 - Monoclinic
        8,  #   3 - Orthorhombic
        8,  #   4 - Tetragonal low
        16,  #  5 - Tetragonal high
        6,  #   6 - Trigonal low
        12,  #  7 - Trigonal high
        12,  #  8 - Hexagonal low
        24,  #  9 - Hexagonal high
        24,  # 10 - Cubic low
        48,  # 11 - Cubic high
    ]
    return LAUE_MULTS[laue_group - 1]


@torch.jit.script
def laue_elements(laue_id: int) -> Tensor:
    """
    Generators for Laue group specified by the laue_id parameter. The first
    element is always the identity.

    1) Laue C1       Triclinic: 1-, 1
    2) Laue C2      Monoclinic: 2/m, m, 2
    3) Laue D2    Orthorhombic: mmm, mm2, 222
    4) Laue C4  Tetragonal low: 4/m, 4-, 4
    5) Laue D4 Tetragonal high: 4/mmm, 4-2m, 4mm, 422
    6) Laue C3    Trigonal low: 3-, 3
    7) Laue D3   Trigonal high: 3-m, 3m, 32
    8) Laue C6   Hexagonal low: 6/m, 6-, 6
    9) Laue D6  Hexagonal high: 6/mmm, 6-m2, 6mm, 622
    10) Laue T       Cubic low: m3-, 23
    11) Laue O      Cubic high: m3-m, 4-3m, 432

    Args:
        laue_id: integer between inclusive [1, 11]

    Returns:
        torch tensor of shape (cardinality, 4) containing the elements of the

    Notes:

    https://en.wikipedia.org/wiki/Space_group

    """

    # sqrt(2) / 2 and sqrt(3) / 2
    R2 = 1.0 / (2.0**0.5)
    R3 = (3.0**0.5) / 2.0

    LAUE_O = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
            [0.0, R2, R2, 0.0],
            [0.0, -R2, R2, 0.0],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [R2, R2, 0.0, 0.0],
            [R2, -R2, 0.0, 0.0],
            [R2, 0.0, R2, 0.0],
            [R2, 0.0, -R2, 0.0],
            [0.0, R2, 0.0, R2],
            [0.0, -R2, 0.0, R2],
            [0.0, 0.0, R2, R2],
            [0.0, 0.0, -R2, R2],
        ],
        dtype=torch.float64,
    )
    LAUE_T = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float64,
    )

    LAUE_D6 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 0.0, 0.0, 1.0],
            [R3, 0.0, 0.0, 0.5],
            [R3, 0.0, 0.0, -0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -0.5, R3, 0.0],
            [0.0, 0.5, R3, 0.0],
            [0.0, R3, 0.5, 0.0],
            [0.0, -R3, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C6 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 0.0, 0.0, 1.0],
            [R3, 0.0, 0.0, 0.5],
            [R3, 0.0, 0.0, -0.5],
        ],
        dtype=torch.float64,
    )

    LAUE_D3 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -0.5, R3, 0.0],
            [0.0, 0.5, R3, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C3 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
        ],
        dtype=torch.float64,
    )

    LAUE_D4 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
            [0.0, R2, R2, 0.0],
            [0.0, -R2, R2, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C4 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
        ],
        dtype=torch.float64,
    )

    LAUE_D2 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C2 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C1 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_GROUPS = [
        LAUE_C1,  #  1 - Triclinic
        LAUE_C2,  #  2 - Monoclinic
        LAUE_D2,  #  3 - Orthorhombic
        LAUE_C4,  #  4 - Tetragonal low
        LAUE_D4,  #  5 - Tetragonal high
        LAUE_C3,  #  6 - Trigonal low
        LAUE_D3,  #  7 - Trigonal high
        LAUE_C6,  #  8 - Hexagonal low
        LAUE_D6,  #  9 - Hexagonal high
        LAUE_T,  #  10 - Cubic low
        LAUE_O,  #  11 - Cubic high
    ]

    return LAUE_GROUPS[laue_id - 1]


"""

Functions for orientation fundamental zones under the 11 Laue point groups.

The orientation fundamental zone is a unique subset of the orientation space
that is used to represent all possible orientations of a crystal possessing a
given symmetry. Covered briefly in the beginning of the following paper:

Krakow, Robert, Robbie J. Bennett, Duncan N. Johnstone, Zoja Vukmanovic,
Wilberth Solano-Alvarez, Steven J. Lainé, Joshua F. Einsle, Paul A. Midgley,
Catherine MF Rae, and Ralf Hielscher. "On three-dimensional misorientation
spaces." Proceedings of the Royal Society A: Mathematical, Physical and
Engineering Sciences 473, no. 2206 (2017): 20170274.

"""


@torch.jit.script
def ori_to_fz_laue(quats: Tensor, laue_id: int) -> Tensor:
    """
    This function moves the given quaternions to the fundamental zone of the
    given Laue group. This computes the orientation fundamental zone, not the
    misorientation fundamental zone.

    Args:
        quats: quaternions to move to fundamental zone of shape (..., 4)
        laue_id: laue group of quaternions to move to fundamental zone

    Returns:
        orientations in fundamental zone of shape (..., 4)

    Notes:

    1) Laue C1       Triclinic: 1-, 1
    2) Laue C2      Monoclinic: 2/m, m, 2
    3) Laue D2    Orthorhombic: mmm, mm2, 222
    4) Laue C4  Tetragonal low: 4/m, 4-, 4
    5) Laue D4 Tetragonal high: 4/mmm, 4-2m, 4mm, 422
    6) Laue C3    Trigonal low: 3-, 3
    7) Laue D3   Trigonal high: 3-m, 3m, 32
    8) Laue C6   Hexagonal low: 6/m, 6-, 6
    9) Laue D6  Hexagonal high: 6/mmm, 6-m2, 6mm, 622
    10) Laue T       Cubic low: m3-, 23
    11) Laue O      Cubic high: m3-m, 4-3m, 432

    """
    # get the important shapes
    data_shape = quats.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))
    card = get_laue_mult(laue_id) // 2
    laue_group = laue_elements(laue_id).to(quats.dtype).to(quats.device)

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equivalent_quaternions_real = qu_prod_pos_real(
        quats.reshape(N, 1, 4), laue_group.reshape(card, 4)
    )

    # find the quaternion with the largest w value
    row_maximum_indices = torch.argmax(equivalent_quaternions_real, dim=-1)

    # gather the equivalent quaternions with the largest w value for each equivalent quaternion set
    output = qu_prod(quats.reshape(N, 4), laue_group[row_maximum_indices])

    return output.reshape(data_shape)


@torch.jit.script
def ori_equiv_laue(quats: Tensor, laue_id: int) -> Tensor:
    """
    Find the equivalent orientations under the given Laue group.

    Args:
        quats: quaternions to move to fundamental zone of shape (..., 4)
        laue_id: laue group of quaternions to move to fundamental zone

    Returns:
        Slices of equivalent quaternions of shape (..., |laue_group|, 4)

    """
    # get the important shapes
    data_shape = quats.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))
    laue_group = laue_elements(laue_id).to(quats.dtype).to(quats.device)

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equivalent_quaternions = qu_prod(quats.reshape(N, 1, 4), laue_group.reshape(-1, 4))

    return equivalent_quaternions.reshape(data_shape[:-1] + (len(laue_group), 4))


@torch.jit.script
def ori_in_fz_laue_brute(quats: Tensor, laue_id: int) -> Tensor:
    """
    Determine if the given unit quaternions with positive real part are in the
    orientation fundamental zone of the given Laue group.

    Args:
        quats: quaternions to move to fundamental zone of shape (..., 4)
        laue_id: laue group of quaternions to move to fundamental zone

    Returns:
        mask of quaternions in fundamental zone of shape (...,)

    Raises:
        ValueError: if the laue_id is not supported

    """
    # get the important shapes
    data_shape = quats.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))
    card = get_laue_mult(laue_id) // 2
    laue_group = laue_elements(laue_id).to(quats.dtype).to(quats.device)

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equiv_quats_real_part = qu_prod_pos_real(
        quats.reshape(N, 1, 4), laue_group.reshape(card, 4)
    ).abs()

    # find the quaternion with the largest w value
    row_maximum_indices = torch.argmax(equiv_quats_real_part, dim=-1)

    # first element is always the identity for the enumerations of the Laue operators
    # so if its index is 0, then a given orientation was already in the fundamental zone
    return (row_maximum_indices == 0).reshape(data_shape[:-1])


@torch.jit.script
def ori_in_fz_laue(quats: Tensor, laue_id: int) -> Tensor:
    """
    Determine if the given unit quaternions with positive real part are in the
    orientation fundamental zone of the given Laue group.

    Args:
        quats: quaternions to move to fundamental zone of shape (..., 4)
        laue_id: laue group of quaternions to move to fundamental zone

    Returns:
        mask of quaternions in fundamental zone of shape (...,)

    Raises:
        ValueError: if the laue_id is not supported

    """
    # all of the bound equality checks have to be inclusive to
    # match the behavior of the brute force method
    if laue_id == 11:
        # O: cubic high
        # max(abs(x,y,z)) < R2M1*abs(w) and sum(abs(x,y,z)) < abs(w)
        xyz_abs = torch.abs(quats[..., 1:])
        return (
            torch.max(xyz_abs, dim=-1).values <= (quats[..., 0] * (2**0.5 - 1))
        ) & (torch.sum(xyz_abs, dim=-1) <= quats[..., 0])
    elif laue_id == 10:
        # T: cubic low
        # sum(abs(x,y,z)) < abs(w)
        return torch.sum(torch.abs(quats[..., 1:]), dim=-1) <= quats[..., 0]
    elif laue_id == 9:
        # D6: hexagonal high
        # m, n = max(abs(x,y)), min(abs(x,y))
        # if m > TAN75 * n then rot = m else rot = R3O2 * m + 0.5 * n
        # if abs(z) < TAN15 * abs(w) and rot < abs(w) then in FZ
        x_abs, y_abs = torch.abs(quats[..., 1]), torch.abs(quats[..., 2])
        cond = x_abs > y_abs
        m = torch.where(cond, x_abs, y_abs)
        n = torch.where(cond, y_abs, x_abs)
        rot = torch.where(m > (2 + 3**0.5) * n, m, (3**0.5 / 2) * m + 0.5 * n)
        return (torch.abs(quats[..., 3]) <= (2 - 3**0.5) * quats[..., 0]) & (
            rot <= quats[..., 0]
        )
    elif laue_id == 8:
        # C6: hexagonal low
        # if abs(z) < TAN15 * abs(w)
        return torch.abs(quats[..., 3]) <= ((2 - 3**0.5) * quats[..., 0])
    elif laue_id == 7:
        # D3: trigonal high
        # if abs(x) > abs(y) * R3:
        #   rot = abs(x)
        # else:
        # rot = R3O2 * abs(y) + 0.5 * abs(x)
        # FZ: if abs(z) < TAN30 * abs(w) and rot < abs(w)
        rot = torch.where(
            torch.abs(quats[..., 1]) >= torch.abs(quats[..., 2]) * (3**0.5),
            torch.abs(quats[..., 1]),
            (3**0.5 / 2) * torch.abs(quats[..., 2]) + 0.5 * torch.abs(quats[..., 1]),
        )
        return (torch.abs(quats[..., 3]) <= ((1.0 / 3**0.5) * quats[..., 0])) & (
            rot <= quats[..., 0]
        )
    elif laue_id == 6:
        # C3: trigonal low
        # FZ: abs(z) < TAN30 * abs(w)
        return torch.abs(quats[..., 3]) <= (1.0 / 3**0.5) * quats[..., 0]
    elif laue_id == 5:
        # D4: tetragonal high
        # m, n = max(abs(x,y)), min(abs(x,y))
        # if m > TAN67_5 * n then rot = m else rot = R2O2 * m + R2O2 * n
        # FZ: abs(z) < TAN22_5 * abs(w) and rot < abs(w)
        x_abs, y_abs = torch.abs(quats[..., 1]), torch.abs(quats[..., 2])
        cond = x_abs > y_abs
        m = torch.where(cond, x_abs, y_abs)
        n = torch.where(cond, y_abs, x_abs)
        rot = torch.where(m > (2**0.5 + 1) * n, m, (1 / 2**0.5) * m + (1 / 2**0.5) * n)
        return (torch.abs(quats[..., 3]) <= ((2**0.5 - 1) * quats[..., 0])) & (
            rot <= quats[..., 0]
        )
    elif laue_id == 4:
        # C4: tetragonal low
        # FZ: abs(z) < TAN22_5 * abs(w)
        return torch.abs(quats[..., 3]) <= (2**0.5 - 1) * quats[..., 0]
    elif laue_id == 3:
        # D2: orthorhombic
        # FZ: max(abs(x,y,z)) < abs(w)
        return torch.max(torch.abs(quats[..., 1:]), dim=-1).values <= quats[..., 0]
    elif laue_id == 2:
        # C2: monoclinic
        # FZ: abs(z) < abs(w)
        return torch.abs(quats[..., 3]) <= quats[..., 0]
    elif laue_id == 1:
        # C1: triclinic
        return torch.full(quats.shape[:-1], True, dtype=torch.bool, device=quats.device)
    else:
        raise ValueError(f"Laue group {laue_id} is not supported")


@torch.jit.script
def cube_to_rfz(
    cc: Tensor,
    laue_group: int,
) -> Tensor:
    """
    Convert cubed coordinates to RFZ coordinates for
    a given Laue group.

    Args:
        :cc (Tensor): Cubed RFZ coordinates.
        :laue_group (int): The Laue group.

    Returns:
        :rfz_coords (Tensor): The RFZ coordinates.

    """
    cc_abs = torch.abs(cc)

    if laue_group == 11:  # cubic high symmetry
        # find indices to do the sorting by absolute values
        sorted, indices = torch.sort(cc_abs, dim=-1, descending=False)
        # we are pointing towards the 111 normal plane
        # find the scaling factor to hit the z = 2**0.5 - 1 face
        scales = sorted[..., 2:3] < (sorted[..., 0:1] + sorted[..., 1:2]) / 2**0.5
        factors = (
            torch.sum(cc_abs, dim=-1, keepdim=True) * (2**0.5 - 1) / sorted[..., 2:3]
        )
        # replace nans with 1 because we were at the origin
        factors[factors.isnan()] = 1
        # scale the sorted values
        rfz_coords = torch.where(scales, sorted * factors, sorted)
        # scatter the sorted values back to the original order
        rfz_coords = torch.scatter(rfz_coords, -1, indices, rfz_coords)
        # copy the sign of the original values
        rfz_coords.copysign_(cc)
        return rfz_coords
    else:
        raise NotImplementedError("Only cubic is supported for now.")


@torch.jit.script
def rfz_to_cube(
    rfz: Tensor,
    laue_group: int,
) -> Tensor:
    """
    Convert RFZ coordinates to cubed coordinates for
    a given Laue group.

    Args:
        :rfz_coords (Tensor): The RFZ coordinates.
        :laue_group (int): The Laue group.

    Returns:
        :cc (Tensor): The cubed coordinates.

    """
    rfz_abs = torch.abs(rfz)

    if laue_group == 11:  # cubic high symmetry
        # same as above but with reciprocal scaling
        sorted, indices = torch.sort(rfz_abs, dim=-1, descending=False)
        scales = sorted[..., 2:3] < (sorted[..., 0:1] + sorted[..., 1:2]) / 2**0.5
        # same factor but we divide by it
        factors = (
            torch.sum(rfz_abs, dim=-1, keepdim=True) * (2**0.5 - 1) / sorted[..., 2:3]
        )
        factors[factors.isnan()] = 1
        cc = torch.where(scales, sorted / factors, sorted)
        cc = torch.scatter(cc, -1, indices, cc)
        cc.copysign_(rfz)
        return cc

    else:
        raise NotImplementedError("Only cubic is supported for now.")


@torch.jit.script
def ori_angle_laue(quats1: Tensor, quats2: Tensor, laue_id: int) -> Tensor:
    """

    Return the misalignment angle in radians between the given quaternions. This
    is not the disorientation angle, which is the angle between the two quaternions
    with both pre and post multiplication by the respective Laue groups.

    Args:
        quats1: quaternions of shape (..., 4)
        quats2: quaternions of shape (..., 4)

    Returns:
        orientation angle in radians of shape (...)

    """

    # multiply without symmetry
    misori_quats = qu_prod(quats1, qu_conj(quats2))

    # move the orientation quaternions to the fundamental zone
    ori_quats_fz = ori_to_fz_laue(misori_quats, laue_id)

    # find the disorientation angle
    return qu_angle(qu_norm_std(ori_quats_fz))


@torch.jit.script
def disorientation(quats1: Tensor, quats2: Tensor, laue_id_1: int, laue_id_2: int):
    """

    Return the disorientation quaternion between the given quaternions.

    Args:
        quats1: quaternions of shape (..., 4)
        quats2: quaternions of shape (..., 4)
        laue_id_1: laue group ID of quats1
        laue_id_2: laue group ID of quats2

    Returns:
        disorientation quaternion of shape (..., 4)

    """

    # get the important shapes
    data_shape = quats1.shape

    # check that the shapes are the same
    if data_shape != quats2.shape:
        raise ValueError(
            f"quats1 and quats2 must have the same data shape, but got {data_shape} and {quats2.shape}"
        )

    # multiply by inverse of second (without symmetry)
    misori_quats = qu_prod(quats1, qu_conj(quats2))

    # find the number of quaternions (generic input shapes are supported)
    N = torch.prod(torch.tensor(data_shape[:-1]))

    # retrieve the laue group elements for the first quaternions
    laue_group_1 = laue_elements(laue_id_1).to(quats1.dtype).to(quats1.device)

    # if the laue groups are the same, then the second laue group is the same as the first
    if laue_id_1 == laue_id_2:
        laue_group_2 = laue_group_1
    else:
        laue_group_2 = laue_elements(laue_id_2).to(quats2.dtype).to(quats2.device)

    # pre / post mult by Laue operators of the second and first symmetry groups respectively
    # broadcasting is done so that the output is of shape (N, |laue_group_2|, |laue_group_1|, 4)
    equivalent_quaternions = qu_prod(
        laue_group_2.reshape(1, -1, 1, 4),
        qu_prod(misori_quats.view(N, 1, 1, 4), laue_group_1.reshape(1, 1, -1, 4)),
    )

    # flatten along the laue group dimensions
    equivalent_quaternions = equivalent_quaternions.reshape(N, -1, 4)

    # find the quaternion with the largest real part value (smallest angle)
    row_maximum_indices = torch.argmax(
        equivalent_quaternions[..., 0].abs(),
        dim=-1,
    )

    # TODO - Multiple equivalent quaternions can have the same angle. This function
    # should choose the one with an axis that is in the fundamental sector of the sphere
    # under the symmetry given by the intersection of the two Laue groups.

    # gather the equivalent quaternions with the largest w value for each equivalent quaternion set
    output = equivalent_quaternions[torch.arange(N), row_maximum_indices]

    return output.reshape(data_shape)


# @torch.jit.script # broadcast_shapes is not supported in torch.jit.script
def disori_angle_laue(quats1: Tensor, quats2: Tensor, laue_id_1: int, laue_id_2: int):
    """

    Return the disorientation angle in radians between the given quaternions.

    Args:
        quats1: quaternions of shape (..., 4)
        quats2: quaternions of shape (..., 4)
        laue_id_1: laue group ID of quats1
        laue_id_2: laue group ID of quats2

    Returns:
        disorientation angle in radians of shape (...,)

    """

    # get the important shapes
    data_shape = torch.broadcast_shapes(quats1.shape[:-1], quats2.shape[:-1])

    # multiply by inverse of second (without symmetry)
    misori_quats = qu_prod(quats1, qu_conj(quats2))

    # find the number of quaternions (generic input shapes are supported)
    N = torch.prod(torch.tensor(misori_quats.shape[:-1]))

    # retrieve the laue group elements for the first quaternions
    laue_group_1 = laue_elements(laue_id_1).to(quats1.dtype).to(quats1.device)

    # if the laue groups are the same, then the second laue group is the same as the first
    if laue_id_1 == laue_id_2:
        laue_group_2 = laue_group_1
    else:
        laue_group_2 = laue_elements(laue_id_2).to(quats2.dtype).to(quats2.device)

    # pre / post mult by Laue operators of the second and first symmetry groups respectively
    # broadcasting is done so that the output is of shape (N, |laue_group_2|, |laue_group_1|, 4)
    equivalent_quat_pos_real = qu_triple_prod_pos_real(
        laue_group_2.reshape(1, -1, 1, 4),
        misori_quats.view(N, 1, 1, 4),
        laue_group_1.reshape(1, 1, -1, 4),
    )

    # flatten along the laue group dimensions
    equivalent_quat_pos_real = equivalent_quat_pos_real.reshape(N, -1)

    # find the largest real part magnitude and return the angle
    cosine_half_angle = torch.max(equivalent_quat_pos_real, dim=-1).values

    # get angle
    angle = 2.0 * torch.acos(cosine_half_angle.clamp_(-1.0, 1.0))

    # reshape to the broadcasted data shape
    return angle.reshape(data_shape)


@torch.jit.script
def sample_ori_fz_laue(
    laue_id: int,
    target_n_samples: int,
    device: torch.device,
    permute: bool = True,
) -> Tensor:
    """

    A function to sample the fundamental zone of SO(3) for a given Laue group.
    This function uses the cubochoric grid sampling method, although other methods
    could be used. Rejection sampling is used so the number of samples will almost
    certainly be different than the target number of samples.

    Args:
        laue_id: integer between 1 and 11 inclusive
        target_n_samples: number of samples to use on the fundamental sector of SO(3)
        device: torch device to use
        permute: whether or not to randomly permute the samples

    Returns:
        torch tensor of shape (n_samples, 4) containing the sampled orientations

    """
    # get the multiplicity of the laue group
    laue_mult = get_laue_mult(laue_id)

    # multiply by half the Laue multiplicity (inversion is not included in the operators)
    required_oversampling = target_n_samples * 0.5 * laue_mult

    # take the cube root to get the edge length
    edge_length = int(required_oversampling ** (1.0 / 3.0))
    so3_samples = so3_cubochoric_grid(edge_length, device=device)

    # reject the points that are not in the fundamental zone
    so3_samples_fz = so3_samples[ori_in_fz_laue(so3_samples, laue_id)]

    # randomly permute the samples
    if permute:
        so3_samples_fz = so3_samples_fz[torch.randperm(so3_samples_fz.shape[0])]

    return so3_samples_fz


@torch.jit.script
def sample_ori_fz_laue_angle(
    laue_id: int,
    angular_resolution_deg: float,
    device: torch.device,
    permute: bool = True,
) -> Tensor:
    """

    A function to sample the fundamental zone of SO(3) for a given Laue group.
    This function uses the cubochoric grid sampling method, although other methods
    could be used. A target number of samples is used, as rejection sampling
    is used here, so the number of samples will almost certainly be different.

    Args:
        laue_id: integer between 1 and 11 inclusive
        target_mean_disorientation: target mean disorientation in radians
        device: torch device to use
        permute: whether or not to randomly permute the samples

    Returns:
        torch tensor of shape (n_samples, 4) containing the sampled orientations

    """
    # use empirical fit to get the number of samples
    N = int(round((131.97049) / (angular_resolution_deg - 0.03732)))
    edge_length = int(2 * N + 1)

    so3_samples = so3_cubochoric_grid(edge_length, device=device)

    # reject the points that are not in the fundamental zone
    so3_samples_fz = so3_samples[ori_in_fz_laue(so3_samples, laue_id)]

    # randomly permute the samples
    if permute:
        so3_samples_fz = so3_samples_fz[torch.randperm(so3_samples_fz.shape[0])]

    return so3_samples_fz


@torch.jit.script
def so3_cu_grid_laue(edge_length: int, laue_id: int, device: torch.device):
    """
    Generate a 3D grid sampling in cubochoric coordinates. Orientations
    are returned as unit quaternions with positive scalar part (w, x, y, z)

    Args:
        edge_length (int): the number of points along each axis of the cube
        laue_id (int): the Laue group ID for the crystal system
        device (torch.device): the device to use

    Returns:
        torch.Tensor: the 3D grid sampling in cubochoric coordinates (n, 3)

    """
    cu = torch.linspace(
        -0.5 * torch.pi ** (2 / 3),
        0.5 * torch.pi ** (2 / 3),
        edge_length + 1,  # add extra point at the opposite faces
        device=device,
    )
    # remove the added last point to make the spacing correct
    cu = cu[:-1]

    if laue_id == 1:  # Triclinic
        cu_x, cu_y, cu_z = cu, cu, cu
    elif laue_id == 2:  # Monoclinic C2
        # trim cu_z to points with |cu_z| <= 0.79
        cu_x, cu_y, cu_z = cu, cu, cu[cu.abs() <= 0.79]
    elif laue_id == 3:  # Orthorhombic D2
        # trim all points with |cu| <= 0.79
        cu = cu[cu.abs() <= 0.79]
        cu_x, cu_y, cu_z = cu, cu, cu
    elif laue_id == 4:  # Tetragonal Low C4
        # trim cu_z to points with |cu_z| <= 0.49
        cu_x, cu_y, cu_z = cu, cu, cu[cu.abs() <= 0.49]
    elif laue_id == 5:  # Tetragonal High D4
        # x and y to 0.66 and z to 0.49
        cu_x = cu[cu.abs() <= 0.66]
        cu_y = cu_x
        cu_z = cu[cu.abs() <= 0.49]
    elif laue_id == 6:  # Trigonal Low C3
        # z to 0.61
        cu_x, cu_y, cu_z = cu, cu, cu[cu.abs() <= 0.61]
    elif laue_id == 7:  # Trigonal High D3
        # x and y to 0.7 and z to 0.61
        cu_x = cu[cu.abs() <= 0.7]
        cu_y = cu_x
        cu_z = cu[cu.abs() <= 0.61]
    elif laue_id == 8:  # Hexagonal Low C6
        # z to 0.35
        cu_x, cu_y, cu_z = cu, cu, cu[cu.abs() <= 0.35]
    elif laue_id == 9:  # Hexagonal High D6
        # x and y to 0.64 and z to 0.35
        cu_x = cu[cu.abs() <= 0.64]
        cu_y = cu_x
        cu_z = cu[cu.abs() <= 0.35]
    elif laue_id == 10:  # Cubic Low T
        # xyz to 0.61
        cu_x = cu[cu.abs() <= 0.61]
        cu_y = cu_x
        cu_z = cu_x
    elif laue_id == 11:  # Cubic High O
        # xyz to 0.44
        cu_x = cu[cu.abs() <= 0.44]
        cu_y = cu_x
        cu_z = cu_x
    else:
        raise ValueError(f"Laue gprou {laue_id} is not an integer in [1, 11]")

    cu = torch.stack(torch.meshgrid(cu_x, cu_y, cu_z, indexing="ij"), dim=-1).reshape(
        -1, 3
    )

    qu = cu2qu(cu)
    qu = qu_std(qu)

    # filter out the quaternions that are not in the fundamental zone
    qu = qu[ori_in_fz_laue(qu, laue_id)]

    return qu


@torch.jit.script
def get_radial_mask(
    shape: tuple[int, int],
    radius: Optional[float] = None,
    center: tuple[float, float] = (0.5, 0.5),
) -> torch.Tensor:
    """
    Get a radial mask.

    Args:
        shape (tuple[int, int]): The shape of the mask.
        radius (float): The radius of the mask.
        center (tuple[float, float]): The center of the mask in fractional coordinates.

    Returns:
        torch.Tensor: The radial mask.

    """
    # get the shape of the mask
    h, w = shape

    # if the radius is not provided, set it to half the minimum dimension
    if radius is None:
        radius = min(h, w) / 2.0

    # create the grid
    ii, jj = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )

    mask = (
        (ii - center[0] * (h - 1)) ** 2 + (jj - center[1] * (w - 1)) ** 2
    ) < radius**2
    return mask.to(torch.bool)


@torch.jit.script
def skew(omega: Tensor) -> Tensor:
    """
    Compute the skew-symmetric matrix of a vector.

    Args:
        omega: torch tensor of shape (..., 3) containing the scaled axis of rotation

    Returns:
        torch tensor of shape (..., 3, 3) containing the skew-symmetric matrix

    """

    data_shape = omega.shape[:-1]
    data_n = int(torch.prod(torch.tensor(data_shape)))
    out = torch.zeros((data_n, 3, 3), dtype=omega.dtype, device=omega.device)
    out[..., 0, 1] = -omega[..., 2].view(-1)
    out[..., 0, 2] = omega[..., 1].view(-1)
    out[..., 1, 2] = -omega[..., 0].view(-1)
    out[..., 1, 0] = omega[..., 2].view(-1)
    out[..., 2, 0] = -omega[..., 1].view(-1)
    out[..., 2, 1] = omega[..., 0].view(-1)
    return out.view(data_shape + (3, 3))


@torch.jit.script
def w_exp_vmat(omega: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute both the matrix exponential of a skew-symmetric matrix
    and the V matrix. Angles below 0.01 radians use Taylor expansion.

    Args:
        omega: torch tensor of shape (..., 3) containing the omega

    Returns:
        torch tensor shape (..., 3, 3) of skew matrix exponential
        torch tensor shape (..., 3, 3) of v matrices

    """
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).unsqueeze(-1)
    skew_mat = skew(omega)
    skew_sq = torch.matmul(skew_mat, skew_mat)

    # Taylor expansion for small angles of each factor
    stable = (theta > 0.001)[..., 0, 0]

    # This prefactor is only used for the calculation of exp(skew)
    # sin(theta) / theta
    # expression: 1 - theta^2 / 6 + theta^4 / 120 - theta^6 / 5040 ...
    prefactor1 = 1 - theta[~stable] ** 2 / 6

    # This prefactor is shared between calculations of exp(skew) and v
    # (1 - cos(theta)) / theta^2
    # expression: 1/2 - theta^2 / 24 + theta^4 / 720 - theta^6 / 40320 ...
    prefactor2 = 1 / 2 - theta[~stable] ** 2 / 24

    # This prefactor is only used for the calculation of v
    # (theta - sin(theta)) / theta^3
    # expression: 1/6 - theta^2 / 120 + theta^4 / 5040 - theta^6 / 362880 ...
    prefactor3 = 1 / 6 - theta[~stable] ** 2 / 120

    skew_exp = torch.empty_like(skew_mat)
    skew_exp[stable] = (
        torch.eye(3, dtype=skew_mat.dtype, device=skew_mat.device)
        + (torch.sin(theta[stable]) / theta[stable]) * skew_mat[stable]
        + (1 - torch.cos(theta[stable])) / theta[stable] ** 2 * skew_sq[stable]
    )
    skew_exp[~stable] = (
        torch.eye(3, dtype=skew_mat.dtype, device=skew_mat.device)
        + prefactor1 * skew_mat[~stable]
        + prefactor2 * skew_sq[~stable]
    )
    # skew_exp = torch.matrix_exp(skew_mat)

    v = torch.empty_like(skew_mat)
    v[stable] = (
        torch.eye(3, dtype=skew_mat.dtype, device=skew_mat.device)
        + (1 - torch.cos(theta[stable])) / theta[stable] ** 2 * skew_mat[stable]
        + ((theta[stable] - torch.sin(theta[stable])) / theta[stable] ** 3)
        * skew_sq[stable]
    )
    v[~stable] = (
        torch.eye(3, dtype=skew_mat.dtype, device=skew_mat.device)
        + prefactor2 * skew_mat[~stable]
        + prefactor3 * skew_sq[~stable]
    )

    return skew_exp, v


@torch.jit.script
def se3_exp_map_om(vecs: Tensor) -> Tensor:
    """
    Compute the SE3 matrix from omega and tvec.

    Args:
        vec: torch tensor of shape (..., 6) containing the omega and tvec

    Returns:
        torch tensor of shape (..., 4, 4) containing the SE3 matrix

    """
    data_shape = vecs.shape[:-1]
    omegas = vecs[..., :3]
    tvecs = vecs[..., 3:]
    rexp, v = w_exp_vmat(omegas)
    se3 = torch.zeros(data_shape + (4, 4), dtype=vecs.dtype, device=vecs.device)
    se3[..., :3, :3] = rexp
    se3[..., :3, 3] = torch.matmul(v, tvecs[..., None]).view(data_shape + (3,))
    se3[..., 3, 3] = 1.0
    return se3


@torch.jit.script
def bruker_geometry_to_SE3(
    pattern_centers: Tensor,
    primary_tilt_deg: Tensor,
    secondary_tilt_deg: Tensor,
    detector_shape: Tuple[int, int],
) -> Tuple[Tensor, Tensor]:
    """
    Convert pattern centers in Bruker coordinates to SE3 transformation matrix.

    Args:
        :pattern_centers (Tensor): The pattern centers.
        :primary_tilt_deg (Tensor): The primary tilt in degrees.
        :secondary_tilt_deg (Tensor): The secondary tilt in degrees.
        :detector_shape (Tuple[int, int]): The detector shape.

    Returns:
        Rotation matrices (..., 3, 3) and translation vectors (..., 3).

    """

    pcx, pcy, pcz = torch.unbind(pattern_centers, dim=-1)
    rows, cols = detector_shape
    rows, cols = float(rows), float(cols)

    # convert to radians
    dt_m_sy = torch.deg2rad(primary_tilt_deg)
    sx = torch.deg2rad(secondary_tilt_deg)

    rotation_matrix = torch.stack(
        [
            -torch.sin(dt_m_sy),
            -torch.sin(sx) * torch.cos(dt_m_sy),
            torch.cos(sx) * torch.cos(dt_m_sy),
            torch.zeros_like(sx),
            torch.cos(sx),
            torch.sin(sx),
            -torch.cos(dt_m_sy),
            torch.sin(sx) * torch.sin(dt_m_sy),
            -torch.sin(dt_m_sy) * torch.cos(sx),
        ],
        dim=-1,
    ).view(-1, 3, 3)

    tx = (
        pcx * cols * torch.sin(sx) * torch.cos(dt_m_sy)
        + pcy * rows * torch.sin(dt_m_sy)
        + pcz * rows * torch.cos(sx) * torch.cos(dt_m_sy)
        - torch.sin(sx) * torch.cos(dt_m_sy) / 2
        - torch.sin(dt_m_sy) / 2
    )
    ty = -cols * pcx * torch.cos(sx) + pcz * rows * torch.sin(sx) + torch.cos(sx) / 2
    tz = (
        -cols * pcx * torch.sin(sx) * torch.sin(dt_m_sy)
        + pcy * rows * torch.cos(dt_m_sy)
        - pcz * rows * torch.sin(dt_m_sy) * torch.cos(sx)
        + torch.sin(sx) * torch.sin(dt_m_sy) / 2
        - torch.cos(dt_m_sy) / 2
    )
    translation_vector = torch.stack([tx, ty, tz], dim=-1)

    return rotation_matrix, translation_vector


@torch.jit.script
def bruker_geometry_from_SE3(
    rotation_matrix: Tensor,
    translation_vector: Tensor,
    detector_shape: Tuple[int, int],
):
    """
    Convert SE3 transformation back to Bruker geometry (invalid if z-axis rotation was optimized).

    Args:
        rotation_matrix (torch.Tensor): The rotation matrix (..., 3, 3).
        translation_vector (torch.Tensor): The translation vector (..., 3).
        detector_shape (Tuple[int, int]): Pattern shape in pixels, (H, W) with 'ij' indexing.

    Returns:
        torch.Tensor: Pattern center parameters (pcx, pcy, pcz).
    """
    tx, ty, tz = torch.unbind(translation_vector, dim=-1)
    rows, cols = detector_shape
    rows, cols = float(rows), float(cols)

    cos_sx = rotation_matrix[..., 1, 1]
    cos_dt_minus_sy = rotation_matrix[..., 0, 2] / cos_sx
    dt_minus_sy = torch.acos(cos_dt_minus_sy.clamp_(min=-1, max=1))
    sx = torch.acos(cos_sx.clamp_(min=-1, max=1))

    pcx = (
        tx * torch.sin(sx) * torch.cos(dt_minus_sy)
        - ty * torch.cos(sx)
        - tz * torch.sin(dt_minus_sy) * torch.sin(sx)
        + 0.5
    ) / cols
    pcy = (tx * torch.sin(dt_minus_sy) + tz * torch.cos(dt_minus_sy) + 0.5) / rows
    pcz = (
        tx * torch.cos(dt_minus_sy) * torch.cos(sx)
        + ty * torch.sin(sx)
        - tz * torch.sin(dt_minus_sy) * torch.cos(sx)
    ) / rows

    pattern_centers = torch.stack([pcx, pcy, pcz], dim=-1)

    return pattern_centers, torch.rad2deg(dt_minus_sy), torch.rad2deg(sx)


class EBSDGeometry(Module):
    """
    WARNING: THE JACOBIAN OF THE SE3 EXPONENTIAL MAPPING MUST BE EXPLICITLY IMPLEMENTED!
    BACKPROP IS ILL-DEFINED. THIS IS A KNOWN ISSUE AND CURRENTLY NOT ADDRESSED.

    Args:
        :detector_shape:
            Pattern shape in pixels, H x W with 'ij' indexing. Number of rows of
            pixels then number of columns of pixels.
        :tilts_degrees:
            Tilt of the sample about the x-axis in degrees (default 0). Tilt of
            the sample about the y-axis in degrees (default 70). Declination of
            the detector below the horizontal in degrees (default 0).
        :proj_center:
            The initial guess for the pattern center. This pattern center is in
            the Bruker convention so it is implicitly specifying the pixel size
            in microns (default (0.5, 0.5, 0.5)).
        :se3_vector:
            The initial guess for the SE3 transformation specified as a Lie
            algebra vector. The vector is in the form (rx, ry, rz, tx, ty, tz).
        :with_se3:
            Whether to fit an SE3 matrix on top of the pattern center
            parameterization.
        :opt_rots:
            Which rotations to optimize. Tuple of booleans (yz rot, xz rot, xy
            rot). Default is (False, False, False) - fitting a pattern center
            only.
        :opt_shifts:
            Which shifts to optimize. Tuple of booleans (x shift, y shift, z
            shift).

    WARNING: THE JACOBIAN OF THE SE3 EXPONENTIAL MAPPING MUST BE EXPLICITLY IMPLEMENTED!
    BACKPROP IS ILL-DEFINED. THIS IS A KNOWN ISSUE AND CURRENTLY NOT ADDRESSED.

    """

    def __init__(
        self,
        detector_shape: Tuple[int, int],
        tilts_degrees: Optional[Tuple[float, float, float]] = (0.0, 70.0, 0.0),
        proj_center: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        se3_vector: Optional[Tensor] = None,
        opt_rots: Tuple[bool, bool, bool] = (False, False, False),
        opt_shifts: Tuple[bool, bool, bool] = (False, False, False),
    ):

        super(EBSDGeometry, self).__init__()

        self.detector_shape = detector_shape

        # convert the tilts to radians
        tilts_deg = tuple(torch.tensor(tilt) for tilt in tilts_degrees)
        self.register_buffer("sample_x_tilt_deg", tilts_deg[0])
        self.register_buffer("sample_y_tilt_deg", tilts_deg[1])
        self.register_buffer("detector_tilt_deg", tilts_deg[2])

        # convert the projection center to a un-optimized SE3 matrix
        rot, translation = bruker_geometry_to_SE3(
            torch.tensor(proj_center).view(1, 3),
            self.detector_tilt_deg - self.sample_y_tilt_deg,
            self.sample_x_tilt_deg,
            self.detector_shape,
        )
        pc_SE3 = torch.zeros(1, 4, 4)
        pc_SE3[..., :3, :3] = rot
        pc_SE3[..., :3, 3] = translation
        pc_SE3[..., 3, 3] = 1.0
        self.register_buffer("pc_SE3_matrix", pc_SE3)

        # convert the SE3 matrix to a parameter
        if se3_vector is None:
            se3_vector = torch.zeros(1, 6)
            self.se3_vector = torch.nn.Parameter(se3_vector)
        else:
            self.se3_vector = torch.nn.Parameter(se3_vector.view(1, 6))

        # make a mask from the opt_rots and opt_shifts
        rotation_mask = torch.tensor(opt_rots).bool()
        translation_mask = torch.tensor(opt_shifts).bool()
        self.register_buffer("mask", torch.cat([rotation_mask, translation_mask]))
        # if the mask has no True values, set opt_se3 to False
        self.opt_se3 = torch.any(self.mask).item()

    def set_optimizable_se3_params(
        self,
        opt_rots: Optional[Tuple[bool, bool, bool]] = None,
        opt_shifts: Optional[Tuple[bool, bool, bool]] = None,
    ):
        """
        Set which parameters are optimizable.

        Args:
            opt_rots:
                Which rotations to optimize. Tuple of booleans (yz rot, xz rot, xy rot).
            opt_shifts:
                Which shifts to optimize. Tuple of booleans (x shift, y shift, z shift).

        """
        # set the masks
        if opt_rots is not None:
            self.mask[:3] = torch.tensor(opt_rots).bool()
        if opt_shifts is not None:
            self.mask[3:] = torch.tensor(opt_shifts).bool()

        # update the opt_se3 flag
        self.opt_se3 = torch.any(self.mask).item()

    def get_detector2sample(self) -> Tuple[Tensor, Tensor]:
        """
        Get the SE3 transformation matrix from detector to sample.

        Returns:
            Rotation matrix and translation vector. Shapes (3, 3) and (3,)

        """
        # check if using a tunable SE3 matrix
        if self.opt_se3:
            # get the SE3 matrix
            tune_se3 = se3_exp_map_om(self.se3_vector * self.mask).squeeze(0)
        else:
            with torch.no_grad():
                tune_se3 = se3_exp_map_om(self.se3_vector * self.mask).squeeze(0)

        # find the combined transform with the tunable one first
        detector_2_sample = self.pc_SE3_matrix @ tune_se3

        # split the SE3 matrix into rotation and translation
        rotation_matrix = detector_2_sample[..., :3, :3]
        translation_vector = detector_2_sample[..., :3, 3]

        return rotation_matrix, translation_vector

    def project2sample(
        self,
        pixel_coordinates: Tensor,
        scan_coordinates: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Project detector coordinates to the sample reference frame.

        Args:
            pixel_coordinates:
                The pixel coordinates in the detector plane. Shape (..., 2).
                Where the z-coordinate is implicitly 0.
            scan_coordinates:
                The offsets of the scan in microns. Shape (..., 2) with (i, j) indexing.

        pixel_coordinates and scan_coordinates are broadcastable:

        If we have 100x100 detector with 10000 separate pixel coordinates with
        placeholder dimensions: (100, 100, 1, 1, 2) and want to find the
        projected coordinates for over each choice of my 100 scan positions: (1,
        1, 10, 10, 2).

        The result would be (100, 100, 10, 10, 2) with the 10000 pixel
        coordinates projected for each of the 100 scan positions.

        Returns:
            The coordinates in the sample frame. Shape (..., 3). If
            scan_coordinates are provided then the coordinates are manually
            shifted by the scan offsets.

        Notes:
            If the scan position was large and in the bottom right away from the
            origin then the relative coordinates would tend negative - "behind"
            the origin - in both X and Y (Kikuchipy convention).

        See Also:
            https://kikuchipy.org/en/stable/tutorials/reference_frames.html

        """

        # get the transformation matrix
        rotation_matrix, translation_vector = self.get_detector2sample()

        # convert to 3D coordinates
        pixel_coordinates = torch.cat(
            [pixel_coordinates, torch.zeros_like(pixel_coordinates[..., 0:1])], dim=-1
        )

        # apply the transformation
        sample_coordinates = (
            ((rotation_matrix @ pixel_coordinates[..., None]))
            + translation_vector[..., None]
        ).squeeze(-1)

        # if scan offsets are provided, apply them
        if scan_coordinates is not None:
            scan_coordinates_3D = torch.concatenate(
                [
                    scan_coordinates,
                    torch.zeros_like(scan_coordinates[..., 0:1]),
                ],
                dim=-1,
            )
            sample_coordinates = sample_coordinates - scan_coordinates_3D

        return sample_coordinates

    def backproject2detector(
        self,
        ray_directions: Tensor,
        scan_coordinates: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Backproject ray directions to the detector plane. By default the origin
        of the sample reference frame is assumed to be the origin for all rays.
        Sample coordinates are given in microns but the results are the same up
        to a scale factor due to the projective nature of the transformation.

        Providing the scan coordinates will define the ray origins a small way
        away from the sample reference frame origin. This is useful for
        backprojecting to the detector plane with a single transformation matrix
        given a firm belief about the relative scan point positions.

        Args:
            ray_directions: sample frame coordinates. Shape (..., 3).
                Technically given in microns but projection is identical up to a scale factor.
            scan_coords: offsets of the scan in microns. Shape (..., 2) with (i, j) indexing.

        Returns:
            Coordinates in the detector plane. Shape (..., 2) in pixels.

        """

        # get the transformation matrix
        rotation_matrix, translation_vector = self.get_detector2sample()

        if scan_coordinates is not None:
            # add the scan coordinates to the ray directions to get the sample coordinates
            ray_origins_sample = torch.concatenate(
                [
                    scan_coordinates,
                    torch.zeros_like(scan_coordinates[..., 0:1]),
                ],
                dim=-1,
            )
            # apply the inverse transformation
            ray_origins_detector = (
                -translation_vector + ray_origins_sample
            ) @ rotation_matrix
        else:
            # if no scan points given, assume all rays originate from the origin
            ray_origins_detector = -translation_vector @ rotation_matrix

        # situate the rays as outgoing from the actual origins
        ray_tips_sample = ray_directions

        if scan_coordinates is not None:
            scan_coordinates_3D = torch.concatenate(
                [
                    scan_coordinates,
                    torch.zeros_like(scan_coordinates[..., 0:1]),
                ],
                dim=-1,
            )
            ray_tips_sample = ray_tips_sample + scan_coordinates_3D

        # transform the tip coordinates to the detector frame
        ray_tips_detector = (ray_tips_sample - translation_vector) @ rotation_matrix

        # find the rays from the origins out towards the tips all in the detector frame
        rays_detector = ray_tips_detector - ray_origins_detector

        # find the intersection with the detector plane
        t = -ray_origins_detector[..., 2] / rays_detector[..., 2]
        pixel_coordinates = (
            ray_origins_detector[..., :2] + t[..., None] * rays_detector[..., :2]
        )

        return pixel_coordinates

    def get_coords_sample_frame(
        self,
        binning: Tuple[int, int],
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> Tensor:
        """
        Get the coordinates of each of the detector pixels in the sample reference frame.

        Args:
            binning:
                The binning of the detector (factor along detector H, factor along detector W).
            dtype:
                The data type of the coordinates. Default is torch.float32.

        Returns:
            Detector pixel coordinates in the sample reference frame. Shape (h, w, 3).

        Notes:
            Norming the coordinates will yield rays from the sample reference
            frame origin to each pixel in the detector plane. These are returned
            unnormed in microns so that each scan location can be used as a ray
            origin. This is handled by the ExperimentPatterns class which modifies
            rays by individual orientation, F-matrix, and scan position.

        """

        # check the binning evenly divides the detector shape
        if self.detector_shape[0] % binning[0] != 0:
            raise ValueError(
                f"A height binning of {binning[0]} does not evenly divide the detector height of {self.detector_shape[0]}."
            )
        if self.detector_shape[1] % binning[1] != 0:
            raise ValueError(
                f"A width binning of {binning[1]} does not evenly divide the detector width of {self.detector_shape[1]}."
            )

        # get binned shape
        binned_shape = (
            self.detector_shape[0] // binning[0],
            self.detector_shape[1] // binning[1],
        )

        binning_fp = (float(binning[0]), float(binning[1]))

        # create the pixel coordinates
        # these are the i indices for 4x4 detector:
        # 0, 0, 0, 0
        # 1, 1, 1, 1
        # 2, 2, 2, 2
        # 3, 3, 3, 3
        # For binning, we don't want every other pixel. We need fractional pixel coordinates.
        # 0,  0,  0,  0
        #   x       x
        # 1,  1,  1,  1
        #
        # 2,  2,  2,  2
        #   x       x
        # 3,  3,  3,  3
        # ... x marks the spots where we want the coordinates.
        # create the pixel coordinates
        pixel_coordinates = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    0.5 * (binning_fp[0] - 1),
                    self.detector_shape[0] - (0.5 * (binning_fp[0] - 1)),
                    binned_shape[0],
                    dtype=dtype,
                    device=self.pc_SE3_matrix.device,  # is always on model device
                ),
                torch.linspace(
                    0.5 * (binning_fp[1] - 1),
                    self.detector_shape[1] - (0.5 * (binning_fp[1] - 1)),
                    binned_shape[1],
                    dtype=dtype,
                    device=self.pc_SE3_matrix.device,  # is always on model device
                ),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)

        return self.project2sample(pixel_coordinates)


@torch.jit.script
def clahe_grayscale(
    x: torch.Tensor,
    clip_limit: float = 40.0,
    n_bins: int = 64,
    grid_shape: Tuple[int, int] = (4, 4),
) -> Tensor:
    """
    CLAHE on batches of 2D tensor.

    Args:
        x (Tensor): Input grayscale images of shape (B, 1, H, W).
        clip_limit (float): The clip limit for the histogram equalization.
        n_bins (int): The number of bins to use for the histogram equalization.
        grid_shape (Tuple[int, int]): The shape of the grid to divide the image into.

    Returns:
        Tensor: Output images of shape (B, 1, H, W).


    """
    # get shapes
    h_img, w_img = x.shape[-2:]
    signal_shape = x.shape[:-2]
    B = int(torch.prod(torch.tensor(signal_shape)).item())

    h_grid, w_grid = grid_shape
    n_tiles = h_grid * w_grid
    h_tile = math.ceil(h_img / h_grid)
    w_tile = math.ceil(w_img / w_grid)
    voxels_per_tile = h_tile * w_tile
    # pad the input to be divisible by the tile counts in each dimension
    pad_w = w_grid - (w_img % w_grid) if w_img % w_grid != 0 else 0
    pad_h = h_grid - (h_img % h_grid) if h_img % h_grid != 0 else 0
    paddings = (0, pad_w, 0, pad_h)
    # torch.nn.functional.pad uses last dimension to first in pairs
    x_padded = torch.nn.functional.pad(
        x,
        paddings,
        mode="reflect",
    )
    # unfold the input into tiles of shape (B, voxels_per_tile, -1)
    tiles = torch.nn.functional.unfold(
        x_padded, kernel_size=(h_tile, w_tile), stride=(h_tile, w_tile)
    )
    tiles = tiles.view((B, voxels_per_tile, n_tiles))
    # permute from (B, voxels_per_tile, n_tiles) to (B, n_tiles, voxels_per_tile)
    tiles = tiles.swapdims(-1, -2)  # tiles are ordered row-major
    # here we pre-allocate the pdf tensor to avoid having all residuals in memory at once
    pdf = torch.zeros((B, n_tiles, n_bins), device=x.device, dtype=torch.float32)
    # use scatter to do an inplace histogram calculation per tile
    # large memory usage because scatter only supports int64 indices
    x_discrete = (tiles * (n_bins - 1)).to(torch.int64)
    pdf.scatter_(
        dim=-1,
        index=x_discrete,
        value=1,
        reduce="add",
    )
    pdf = pdf / pdf.sum(dim=-1, keepdim=True)
    # pdf is handled in pixel counts for OpenCV equivalence
    histos = (pdf * voxels_per_tile).view(-1, n_bins)
    if clip_limit > 0:
        # calc limit
        limit = max(clip_limit * voxels_per_tile // n_bins, 1)
        histos.clamp_(max=limit)
        # calculate the clipped pdf of shape (B, n_tiles, n_bins)
        clipped = voxels_per_tile - histos.sum(dim=-1)
        # calculate the excess of shape (B, n_tiles, n_bins)
        residual = torch.remainder(clipped, n_bins)
        redist = (clipped - residual).div(n_bins)
        histos += redist[..., None]
        # trick to avoid using a loop to assign the residual
        v_range = torch.arange(n_bins, device=histos.device)
        mat_range = v_range.repeat(histos.shape[0], 1)
        histos += mat_range < residual[None].transpose(0, 1)
    # cdf (B, n_tiles, n_bins)
    cdfs = torch.cumsum(histos, dim=-1) / voxels_per_tile
    cdfs = cdfs.clamp_(min=0.0, max=1.0)
    cdfs = cdfs.view(
        B,
        h_grid,
        w_grid,
        n_bins,
    )
    coords = torch.meshgrid(
        [
            torch.linspace(-1.0, 1.0, w_img, device=x.device),
            torch.linspace(-1.0, 1.0, h_img, device=x.device),
        ],
        indexing="xy",
    )
    coords = torch.stack(coords, dim=-1)

    # we use grid_sample with border padding to handle the extrapolation
    # we have to use trilinear as tri-cubic is not available
    coords_into_cdfs = torch.cat(
        [
            x[..., None].view(B, 1, h_img, w_img, 1) * 2.0 - 1.0,
            coords[None, None].repeat(B, 1, 1, 1, 1),
        ],
        dim=-1,
    )
    x = torch.nn.functional.grid_sample(
        cdfs[:, None, :, :, :],  # (B, 1, n_bins, GH, GW)
        coords_into_cdfs,  # (B, 1, H, W, 3)
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return x[:, 0].reshape(
        signal_shape + (h_img, w_img)
    )  # (B, C, D, H, W) of (B, 1, 1, H, W) -> (B, 1, H, W) -> in shape


class ExperimentPatterns(Module):
    """

    Class for processing and sampling EBSD patterns from a scan.

    Args:
        :patterns (Tensor): Pattern data tensor shaped (SCAN_H, SCAN_W, H, W),
            (N_PATS, H, W), or (H, W).

    """

    def __init__(
        self,
        patterns: Tensor,
    ):
        super(ExperimentPatterns, self).__init__()

        if len(patterns.shape) < 2:
            raise ValueError("'patterns' requires >= 2 dimensions (B, H, W) or (H, W)")
        if len(patterns.shape) < 3:
            patterns = patterns.unsqueeze(0)
        if len(patterns.shape) > 4:
            raise ValueError(
                "3D Volume not yet supported... "
                + f"'patterns' requires dimensions: "
                + "(SCAN_H, SCAN_W, H, W), (N_PATS, H, W), or (H, W)"
            )

        self.pattern_shape = patterns.shape[-2:]
        self.spatial_shape = patterns.shape[:-2] if len(patterns.shape) > 2 else (1, 1)
        self.patterns = patterns.view(-1, *patterns.shape[-2:])

        # set number of patterns and pixels
        self.n_patterns = self.patterns.shape[0]
        self.n_pixels = self.pattern_shape[0] * self.pattern_shape[1]

        self.phases = None
        self.orientations = None
        self.inv_f_matrix = None

    def set_spatial_coords(
        self,
        spatial_coords: Tensor,
        indices: Optional[Tensor] = None,
    ):
        """
        Set the spatial coordinates for the ExperimentPatterns object.

        Args:
            :spatial_coords (Tensor): Spatial coordinates tensor shaped (N_PATS, N_Spatial_Dims).

        """
        if indices is None:
            if spatial_coords.shape[0] != self.n_patterns:
                raise ValueError(
                    f"Spatial coordinates must have the same number of patterns as the ExperimentPatterns object. "
                    + f"Got {spatial_coords.shape[0]} spatial coordinates and {self.n_patterns} patterns."
                )
            self.spatial_coords = spatial_coords
        else:
            if spatial_coords.shape[0] != indices.shape[0]:
                raise ValueError(
                    f"Spatial coordinates must have the same number of patterns as the indices. "
                    + f"Got {spatial_coords.shape[0]} spatial coordinates and {indices.shape[0]} indices."
                )
            self.spatial_coords[indices] = spatial_coords

    def get_spatial_coords(
        self,
        indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Retrieve the spatial coordinates for the ExperimentPatterns object.

        Args:
            :indices (Tensor): Indices of the patterns to retrieve.

        Returns:
            Tensor: Retrieved spatial coordinates.

        """
        if self.spatial_coords is None:
            raise ValueError("Spatial coordinates must be set before retrieving them.")
        else:
            if indices is None:
                return self.spatial_coords
            return self.spatial_coords[indices]

    def set_orientations(
        self,
        orientations: Tensor,
        indices: Optional[Tensor] = None,
    ):
        """
        Set the orientations for the ExperimentPatterns object.

        Args:
            :orientations (Tensor): Orientations tensor shaped (N_PATS, 4).

        """
        # check the shape
        if len(orientations.shape) != 2:
            raise ValueError(
                f"Orientations must be quaternions (N_ORI, 4). Got {orientations.shape}."
            )
        if indices is None:
            if orientations.shape[0] != self.n_patterns:
                raise ValueError(
                    f"Orientations must have the same number of patterns as the ExperimentPatterns object. "
                    + f"Got {orientations.shape[0]} orientations and {self.n_patterns} patterns."
                )
            if orientations.shape[1] != 4:
                raise ValueError(
                    f"Orientations must be quaternions (w, x, y, z). Got {orientations.shape[1]}."
                )
            self.orientations = orientations
        else:
            # check dim of indices
            if len(indices.shape) != 1:
                raise ValueError(f"Indices must be (N_ORI,). Got {indices.shape}.")
            if orientations.shape[0] != indices.shape[0]:
                raise ValueError(
                    f"Orientations must have the same number of patterns as the indices. "
                    + f"Got {orientations.shape[0]} orientations and {indices.shape[0]} indices."
                )
            if orientations.shape[1] != 4:
                raise ValueError(
                    f"Orientations must be quaternions (w, x, y, z). Got {orientations.shape[1]} components."
                )
            self.orientations[indices] = orientations

    def get_orientations(
        self,
        indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Retrieve the orientations for the ExperimentPatterns object.

        Returns:
            Tensor: Rotations tensor shaped (N_PATS, 4).

        """
        if indices is None:
            ori = self.orientations
        else:
            ori = self.orientations[indices]
        return ori

    def set_inv_f_matrix(
        self,
        f_matrix: Optional[Tensor] = None,
    ):
        """
        Set the deformation gradient for the ExperimentPatterns object.

        Args:
            :deformation_gradient (Tensor): Deformation gradient tensor shaped
                (N_PATS, 3, 3) or (N_PATS, 9) or (1, 3, 3) or (1, 9). If None,
                the deformation gradient is set to the identity for each pattern.

        """
        if f_matrix is None:
            f_matrix = torch.zeros(
                self.n_patterns,
                9,
                device=self.patterns.device,
            )
            f_matrix[:, 0] = 1.0
            f_matrix[:, 4] = 1.0
            f_matrix[:, 8] = 1.0
            f_matrix = f_matrix.reshape(self.n_patterns, 3, 3)
        else:
            if f_matrix.shape[0] != self.n_patterns:
                raise ValueError(
                    f"Deformation gradients must have the same number of patterns as the ExperimentPatterns object. "
                    + f"Got {f_matrix.shape[0]} gradients and {self.n_patterns} patterns."
                )
            if (f_matrix.shape[-2] != 3 or f_matrix.shape[-1] != 3) and f_matrix.shape[
                -1
            ] != 9:
                raise ValueError(
                    f"Deformation gradients must be 3x3 matrices. Got {f_matrix.shape[-2:]} components."
                )
        self.inv_f_matrix = f_matrix.view(self.n_patterns, 3, 3)

    def get_inv_f_matrix(
        self,
        indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Retrieve the deformation gradient tensor inverse for the ExperimentPatterns object.

        Returns:
            Tensor: Deformation gradient tensor shaped (N_PATS, 3, 3).

        """
        if self.inv_f_matrix is None:
            raise ValueError(
                "Deformation gradients must be set before retrieving them."
            )
        if indices is None:
            return self.inv_f_matrix
        return self.inv_f_matrix[indices]

    def subtract_overall_background(
        self,
    ):
        """
        Subtract the overall background from the patterns.

        """

        # subtract the mean of each pixel from all patterns
        self.patterns -= torch.mean(self.patterns, dim=0, keepdim=True)

    def contrast_enhance_clahe(
        self,
        clip_limit: float = 40.0,
        tile_grid_size: int = 4,
        n_bins: int = 64,
    ):
        """
        Contrast enhance the patterns using CLAHE.

        Args:
            :clip_limit (float): Clip limit for CLAHE.
            :tile_grid_size (int): Tile grid size for CLAHE.

        """

        self.patterns = clahe_grayscale(
            self.patterns[:, None],
            clip_limit=clip_limit,
            n_bins=n_bins,
            grid_shape=(tile_grid_size, tile_grid_size),
        ).squeeze(1)

    def normalize_per_pattern(
        self,
        norm_type: str = "minmax",
    ):
        """
        Normalize the patterns.

        Args:
            :method (str): Normalization method: "minmax", "zeromean", "standard"

        """

        self.patterns = self.patterns.view(self.n_patterns, -1)
        if norm_type == "minmax":
            pat_mins = torch.min(self.patterns, dim=-1).values
            pat_maxs = torch.max(self.patterns, dim=-1).values
            self.patterns -= pat_mins[..., None]
            self.patterns /= 1e-4 + pat_maxs[..., None] - pat_mins[..., None]
        elif norm_type == "zeromean":
            self.patterns -= torch.mean(self.patterns, dim=-1, keepdim=True)
        elif norm_type == "standard":
            self.patterns -= torch.mean(self.patterns, dim=-1, keepdim=True)
            self.patterns /= torch.std(self.patterns, dim=-1, keepdim=True)
        else:
            raise ValueError(
                f"Invalid normalization method. Got {norm_type} but expected 'minmax', 'zeromean', or 'standard'."
            )
        self.patterns = self.patterns.view(-1, *self.pattern_shape)

    def standard_clean(
        self,
    ):
        """
        Standard cleaning of the patterns.

        """
        self.subtract_overall_background()
        self.normalize_per_pattern(norm_type="minmax")
        self.contrast_enhance_clahe()
        self.normalize_per_pattern(norm_type="minmax")

    def get_patterns(
        self,
        indices: Tensor,
        binning: int,
    ):
        """
        Retrieve patterns from the ExperimentPatterns object.

        Args:
            :indices (Tensor): Indices of the patterns to retrieve.
            :binning (Union[float, int]): Binning factor can be non-integer for pseudo-binning.

        Returns:
            Tensor: Retrieved patterns.

        """

        # binning is always a factor of both H and W
        # so downscaling is easy
        pats = self.patterns[indices]

        # use avg pooling to bin the patterns if integer binning factor
        if binning > 1:
            # if isinstance(binning, int) or (binning % 1 == 0):
            pats = torch.nn.functional.avg_pool2d(pats, binning)
            # else:
            #     blurrer = BlurAndDownsample(scale_factor=binning).to(pats.device)
            #     pats = blurrer(pats.view(-1, 1, *self.pattern_shape)).squeeze(1)
        return pats

    def get_indices_per_phase(
        self,
    ) -> List[Tensor]:
        """
        Retrieve the indices of the patterns for each phase.

        Returns:
            List[Tensor]: List of indices for each phase.

        """
        if self.phases is None:
            raise ValueError(
                "Phases must be indexed before retrieving indices per phase."
            )
        return [
            torch.nonzero(self.phases == i, as_tuple=False).squeeze()
            for i in range(self.phases.max().item() + 1)
        ]

    def set_raw_indexing_results(
        self,
        quaternions: Tensor,
        metrics: Tensor,
        phase_id: int,
    ):
        """
        Set the dictionary matches for the ExperimentPatterns object for a single phase.

        Args:
            :dictionary_matches (Tensor): Dictionary matches tensor.

        """
        if not hasattr(self, "raw_indexing_results"):
            self.raw_indexing_results = {
                phase_id: (quaternions, metrics),
            }
        else:
            self.raw_indexing_results[phase_id] = (quaternions, metrics)

    def combine_indexing_results(
        self,
        higher_is_better: bool,
    ) -> None:
        """
        Collapse the raw indexing results along the phase and only take the
        top match for each pattern.
        """

        if not hasattr(self, "raw_indexing_results"):
            raise ValueError("Raw indexing results must be set before combining.")

        if len(self.raw_indexing_results) > 1:
            # loop over the phases, concatenate the metrics and only store the phase
            # and quaternion with the best metric
            if higher_is_better:
                # fill metric with -inf
                best_metrics = torch.full(
                    (self.n_patterns,),
                    float("-inf"),
                    device=self.patterns.device,
                )
            else:
                # fill metric with +inf
                best_metrics = torch.full(
                    (self.n_patterns,),
                    float("inf"),
                    device=self.patterns.device,
                )

            # make a phase_id tensor
            phase_ids = torch.full(
                self.n_patterns,
                fill_value=-1,
                dtype=torch.uint8,
                device=self.patterns.device,
            )

            # make a quaternion tensor
            orientations = torch.zeros(
                self.n_patterns,
                4,
                device=self.patterns.device,
            )

            for phase_id, (quaternions, metrics) in self.raw_indexing_results.items():
                if higher_is_better:
                    mask = metrics > best_metrics
                else:
                    mask = metrics < best_metrics
                best_metrics[mask] = metrics[mask]
                phase_ids[mask] = phase_id
                orientations[mask] = quaternions[mask]
            self.phases = phase_ids
            self.orientations = orientations
            self.best_metrics = best_metrics
        else:
            self.phases = torch.full(
                (self.n_patterns,),
                fill_value=list(self.raw_indexing_results.keys())[0],
                dtype=torch.uint8,
                device=self.patterns.device,
            )
            self.orientations = self.raw_indexing_results[0][0][:, 0]
            self.best_metrics = self.raw_indexing_results[0][1][:, 0]


"""

Lenstra elliptic-curve factorization method 

Originally from:

https://stackoverflow.com/questions/4643647/fast-prime-factorization-module

and

https://en.wikipedia.org/wiki/Lenstra_elliptic-curve_factorization


Useful for reshaping arrays into approximately square shapes for GPU processing.

"""


def FactorECM(N0):
    def factor_trial_division(x):
        factors = []
        while (x & 1) == 0:
            factors.append(2)
            x >>= 1
        for d in range(3, int(math.sqrt(x)) + 1, 2):
            while x % d == 0:
                factors.append(d)
                x //= d
        if x > 1:
            factors.append(x)
        return sorted(factors)

    def is_probably_prime_fermat(n, trials=32):
        for _ in range(trials):
            if pow(random.randint(2, n - 2), n - 1, n) != 1:
                return False
        return True

    def gen_primes_sieve_of_eratosthenes(end):
        composite = [False] * end
        for p in range(2, int(math.sqrt(end)) + 1):
            if composite[p]:
                continue
            for i in range(p * p, end, p):
                composite[i] = True
        return [p for p in range(2, end) if not composite[p]]

    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a

    def egcd(a, b):
        ro, r, so, s = a, b, 1, 0
        while r != 0:
            ro, (q, r) = r, divmod(ro, r)
            so, s = s, so - q * s
        return ro, so, (ro - so * a) // b

    def modular_inverse(a, n):
        g, s, _ = egcd(a, n)
        if g != 1:
            raise ValueError(a)
        return s % n

    def elliptic_curve_add(N, A, B, X0, Y0, X1, Y1):
        if X0 == X1 and Y0 == Y1:
            l = ((3 * X0**2 + A) * modular_inverse(2 * Y0, N)) % N
        else:
            l = ((Y1 - Y0) * modular_inverse(X1 - X0, N)) % N
        x = (l**2 - X0 - X1) % N
        y = (l * (X0 - x) - Y0) % N
        return x, y

    def elliptic_curve_mul(N, A, B, X, Y, k):
        k -= 1
        BX, BY = X, Y
        while k != 0:
            if k & 1:
                X, Y = elliptic_curve_add(N, A, B, X, Y, BX, BY)
            BX, BY = elliptic_curve_add(N, A, B, BX, BY, BX, BY)
            k >>= 1
        return X, Y

    def factor_ecm(N, bound=512, icurve=0):
        def next_factor_ecm(x):
            return factor_ecm(x, bound=bound + 512, icurve=icurve + 1)

        def prime_power(p, bound2=int(math.sqrt(bound) + 1)):
            mp = p
            while mp * p < bound2:
                mp *= p
            return mp

        if N < (1 << 16):
            return factor_trial_division(N)

        if is_probably_prime_fermat(N):
            return [N]

        while True:
            X, Y, A = [random.randrange(N) for _ in range(3)]
            B = (Y**2 - X**3 - A * X) % N
            if 4 * A**3 - 27 * B**2 != 0:
                break

        for p in gen_primes_sieve_of_eratosthenes(bound):
            k = prime_power(p)
            try:
                X, Y = elliptic_curve_mul(N, A, B, X, Y, k)
            except ValueError as ex:
                g = gcd(ex.args[0], N)
                if g != N:
                    return sorted(next_factor_ecm(g) + next_factor_ecm(N // g))
                else:
                    return next_factor_ecm(N)
        return next_factor_ecm(N)

    return factor_ecm(N0)


def nearly_square_factors(n):
    """
    For a given highly composite number n, find two factors that are roughly
    close to each other. This is useful for reshaping some arrays into square
    shapes for GPU processing.
    """
    factor_a = 1
    factor_b = 1
    factors = FactorECM(n)
    for factor in factors[::-1]:
        if factor_a > factor_b:
            factor_b *= factor
        else:
            factor_a *= factor
    return factor_a, factor_b


@torch.jit.script
def space_group_to_laue(space_group: int) -> int:
    if space_group < 1 or space_group > 230:
        raise ValueError("The space group must be between 1 and 230 inclusive.")
    if space_group > 206:
        laue_group = 11
    elif space_group > 193:
        laue_group = 10
    elif space_group > 176:
        laue_group = 9
    elif space_group > 167:
        laue_group = 8
    elif space_group > 155:
        laue_group = 7
    elif space_group > 142:
        laue_group = 6
    elif space_group > 88:
        laue_group = 5
    elif space_group > 74:
        laue_group = 4
    elif space_group > 15:
        laue_group = 3
    elif space_group > 2:
        laue_group = 2
    else:
        laue_group = 1
    return laue_group


@torch.jit.script
def point_group_to_laue(point_group: str) -> int:
    if point_group in ["m3-m", "4-3m", "432"]:
        laue_group = 11
    elif point_group in ["m3-", "23"]:
        laue_group = 10
    elif point_group in ["6/mmm", "6-m2", "6mm", "622"]:
        laue_group = 9
    elif point_group in ["6/m", "6-", "6"]:
        laue_group = 8
    elif point_group in ["3-m", "3m", "32"]:
        laue_group = 7
    elif point_group in ["3-", "3"]:
        laue_group = 6
    elif point_group in ["4/mmm", "4-2m", "4mm", "422"]:
        laue_group = 5
    elif point_group in ["4/m", "4-", "4"]:
        laue_group = 4
    elif point_group in ["mmm", "mm2", "222"]:
        laue_group = 3
    elif point_group in ["2/m", "m", "2"]:
        laue_group = 2
    elif point_group in ["1-", "1"]:
        laue_group = 1
    else:
        raise ValueError(
            f"The point group symbol is not recognized, as one of, "
            + "m3-m, 4-3m, 432, m3-, 23, 6/mmm, 6-m2, 6mm, 622, 6/m, "
            + "6-, 6, 3-m, 3m, 32, 3-, 3, 4/mmm, 4-2m, 4mm, 422, 4/m, "
            + "4-, 4, mmm, mm2, 222, 2/m, m, 2, 1-, 1"
        )
    return laue_group


class MasterPattern(Module):
    """
    Master pattern class for storing and interpolating master patterns.

    Args:
        master_pattern (Tensor): The master pattern as a 2D tensor. The Northern hemisphere
            is on top and the Southern hemisphere is on the bottom. They are concatenated
            along the first dimension. If a tuple is provided, the first tensor is the
            Northern hemisphere and the second tensor is the Southern hemisphere.
        laue_group (int, str, optional):
            The Laue group number or symbol in Schönflies notation.
        point_group (str, optional):
            The point group symbol in Hermann-Mauguin notation.
        space_group (int, optional):
            The space group number. Must be between 1 and 230 inclusive.

    """

    def __init__(
        self,
        master_pattern: Union[Tensor, Tuple[Tensor, Tensor]],
        laue_group: Optional[int] = None,
        point_group: Optional[str] = None,
        space_group: Optional[int] = None,
    ):
        super(MasterPattern, self).__init__()

        # check that at least one of laue_groups, point_groups, or space_groups is not None
        if laue_group is None and point_group is None and space_group is None:
            raise ValueError(
                "At least one of laue_groups, point_groups, or space_groups must be provided"
            )

        # check that only one of laue_groups, point_groups, or space_groups is not None
        if (
            (laue_group is not None and point_group is not None)
            or (laue_group is not None and space_group is not None)
            or (point_group is not None and space_group is not None)
        ):
            raise ValueError(
                "Only one of laue_groups, point_groups, or space_groups should be provided"
            )

        # if master_pattern is a tuple, concatenate the two tensors
        if isinstance(master_pattern, tuple):
            # check they are both 2D tensors with the same shape
            if len(master_pattern[0].shape) != 2 or len(master_pattern[1].shape) != 2:
                raise ValueError(
                    "Both tensors in 'master_pattern' must be 2D tensors when provided as a tuple"
                )
            master_pattern = torch.cat(master_pattern, dim=-1)

        # check that the master pattern is a 2D tensor
        if len(master_pattern.shape) != 2:
            raise ValueError(
                f"'master_pattern' must be a 2D tensor, got shape {master_pattern.shape}"
            )

        self.register_buffer("master_pattern", master_pattern)

        # set the laue group
        if laue_group is not None:
            self.laue_group = laue_group
        elif point_group is not None:
            self.laue_group = point_group_to_laue(point_group)
        else:
            self.laue_group = space_group_to_laue(space_group)

        self.master_pattern = master_pattern
        self.master_pattern_binned = None
        self.factor_dict = {}

    def bin(
        self,
        binning_factor: int,
    ) -> None:
        """
        Set the binning factor for the master pattern under a separate attribute
        called `master_pattern_binned`.

        Args:
            :binning (Union[float, int]): Binning factor can be non-integer for pseudo-binning.

        """
        # blurrer = BlurAndDownsample(scale_factor=binning_factor).to(
        #     self.master_pattern.device
        # )
        # self.master_pattern_binned = blurrer(self.master_pattern[None, None, ...])[0, 0]

        self.master_pattern_binned = torch.nn.functional.avg_pool2d(
            self.master_pattern[None, None, ...], binning_factor, stride=binning_factor
        )[0, 0]

    def normalize(
        self,
        norm_type: str,
    ) -> None:
        """
        Normalize the master pattern.

        Args:
            :norm_type (str): Normalization type: "minmax", "zeromean", "standard"

        """
        if norm_type == "minmax":
            pat_mins = torch.min(self.master_pattern)
            pat_maxs = torch.max(self.master_pattern)
            self.master_pattern -= pat_mins
            self.master_pattern /= 1e-4 + pat_maxs - pat_mins
        elif norm_type == "zeromean":
            self.master_pattern -= torch.mean(self.master_pattern)
        elif norm_type == "standard":
            self.master_pattern -= torch.mean(self.master_pattern)
            self.master_pattern /= torch.std(self.master_pattern)
        else:
            raise ValueError(
                f"Invalid normalization method. Got {norm_type} but expected 'minmax', 'zeromean', or 'standard'."
            )

    def interpolate(
        self,
        coords_3D: Tensor,
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = False,
        normalize_coords: bool = True,
        virtual_binning: int = 1,
    ) -> Tensor:
        """
        Interpolate the master pattern at the given angles.

        Args:
            :coords_3D (Tensor): (..., 3) Cartesian points to interpolate on the sphere.
            :mode (str): Interpolation mode. Default: "bilinear".
            :padding_mode (str): Padding mode. Default: "zeros".
            :align_corners (bool): Align corners. Default: True.
            :normalize_coords (bool): Normalize the coordinates. Default: True.
            :virtual_binning (int): Virtual binning factor on the passed coordinates. Default: 1.

        Returns:
            The interpolated master pattern pixel values.

        """
        # norm
        if normalize_coords:
            coords_3D = coords_3D / torch.norm(coords_3D, dim=-1, keepdim=True)

        # blur the master pattern if virtual binning is used
        if virtual_binning > 1:
            if self.master_pattern_binned is None:
                self.bin(virtual_binning)
            master_pattern_prep = self.master_pattern_binned
        else:
            master_pattern_prep = self.master_pattern

        # project to the square [-1, 1] x [-1, 1]
        projected_coords_2D = rosca_lambert_side_by_side(coords_3D)

        # we want to accept a generic number of dimensions (..., 2)
        # grid_sample will perform extremely poorly when given grid shaped:
        # (1, 1000000000, 1, 2) or (1, 1, 1000000000, 2) etc.
        # instead we find the pseudo height and width closest to that of a square
        # (..., 3) -> (1, H*, W*, 2) -> (...,) where H* x W* = n_coords
        #  3rd fastest integer factorization algorithm is good for small numbers
        n_elem = math.prod(projected_coords_2D.shape[:-1])
        if n_elem in self.factor_dict:
            H_pseudo, W_pseudo = self.factor_dict[n_elem]
        else:
            H_pseudo, W_pseudo = nearly_square_factors(n_elem)
            if len(self.factor_dict) > 100:
                self.factor_dict = {}
            self.factor_dict[n_elem] = H_pseudo, W_pseudo

        projected_coords_2D = projected_coords_2D.view(H_pseudo, W_pseudo, 2)

        output = torch.nn.functional.grid_sample(
            # master patterns are viewed as (1, 1, H_master, 2*W_master)
            master_pattern_prep[None, None, ...],
            # coordinates should usually be shaped:
            # (1, n_ori, H_pat*W_pat, 2) or (1, H_pat, W_pat, 2)
            projected_coords_2D[None, ...],
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        # reshape back to the original shape using coords_3D as a template
        return output.view(coords_3D.shape[:-1])

    def apply_clahe(
        self,
        clip_limit: float = 40.0,
        n_bins: int = 64,
        grid_shape: Tuple[int, int] = (8, 8),
    ):
        """
        Apply CLAHE to the master pattern.

        Args:
            :clip_limit (float): The clip limit for the histogram equalization.
            :n_bins (int): The number of bins to use for the histogram equalization.
            :grid_shape (Tuple[int, int]): The shape of the grid to divide the image into.

        """
        self.master_pattern = clahe_grayscale(
            self.master_pattern[None, None, ...],
            clip_limit=clip_limit,
            n_bins=n_bins,
            grid_shape=grid_shape,
        )[0, 0]


"""
The conventional brute force code is taken from KeOps benchmarks found at:
https://www.kernel-operations.io/keops/_auto_benchmarks/benchmark_KNN.html
Feydy, Jean, Alexis Glaunès, Benjamin Charlier, and Michael Bronstein. "Fast
geometric learning with symbolic matrices." Advances in Neural Information
Processing Systems 33 (2020): 14448-14462.

CPU quantization (doing arithmetic in int8) is also supported, notes here:

PyTorch provides a very nice interface with FBGEMM for quantized matrix
multiplication. This is used to accelerate the distance matrix calculation
tremendously. PyTorch plans to support GPU quantization in the future on NVIDIA
GPUs. This will provide a 10x speedup over the current GPU implementation. Apple
silicon *CPUs* are not yet supported by the ARM64 alternative of FBGEMM called
QNNPACK (I am not absolutely certain.), but a Metal Performance Shaders (MPS)
PyTorch backend provies Apple Silicon *GPU* support today.

I have explored many KNN search approximation methods such as HNSW, IVF-Flat,
etc (on the GPU too).

They are not worth running on the raw EBSD images because the time invested in
building the index is large. For cubic materials, a dictionary around 100,000
images is needed, and graph building will take many minutes if the patterns are
larger than 60x60 = 3600 dimensions. PCA is fast and leverages the compactness
of the dictionary in image space.

TURNS OUT THAT PCA + Approx NN (ANN) IS A REALLY REALLY GOOD IDEA.

"""


class LinearLayer(Module):
    """
    This is a wrapper around torch.nn.Linear defining a no bias linear layer
    model. The 8-bit quantized arithmetic libraries take neural networks as
    inputs. This allows us to quantize a simple matrix multiplication.

    """

    def __init__(self, in_dim: int, out_dim: int):
        super(LinearLayer, self).__init__()
        self.fc = Linear(in_dim, out_dim, bias=False)

    def forward(self, inp: Tensor) -> Tensor:
        return self.fc(inp)


def quant_model_data(data: Tensor) -> Module:
    layer = LinearLayer(data.shape[1], data.shape[0])
    layer.fc.weight.data = data
    layer.eval()
    return quantize_dynamic(
        model=layer, qconfig_spec={Linear}, dtype=torch.qint8, inplace=True
    )


def quant_model_query(query: Tensor) -> Module:
    layer = LinearLayer(query.shape[1], query.shape[0])
    layer.fc.weight.data = query
    layer.eval()
    return quantize_dynamic(
        model=layer, qconfig_spec={Linear}, dtype=torch.qint8, inplace=True
    )


def topk_quantized_data(query_batch: Tensor, quantized_data: Module, k: int) -> Tensor:
    res = torch.topk(quantized_data(query_batch), k, dim=1, largest=True, sorted=False)
    return res.values, res.indices


def topk_quantized_query(data_batch: Tensor, quantized_query: Module, k: int) -> Tensor:
    res = torch.topk(quantized_query(data_batch), k, dim=0, largest=True, sorted=False)
    return res.values.t(), res.indices.t()


@torch.jit.script
def knn_batch(
    data: Tensor, query: Tensor, topk: int, metric: str
) -> Tuple[Tensor, Tensor]:
    lgst = True if metric == "angular" else False
    if metric == "euclidean":
        # norms must be precomputed in FP32.
        data_norm = (data.float() ** 2).sum(dim=-1)
        q_norm = (query.float() ** 2).sum(dim=-1)
        dist = (
            q_norm.view(-1, 1)
            + data_norm.view(1, -1)
            - (2.0 * query @ data.t()).float()
        )
    elif metric == "manhattan":
        dist = (query[:, None, :] - data[None, :, :]).abs().sum(dim=2)
    elif metric == "angular":
        dist = query @ data.t()
    else:
        raise NotImplementedError(f"'{metric}' not supported.")
    topk_distances, topk_indices = torch.topk(
        dist, topk, dim=1, largest=lgst, sorted=False
    )
    return topk_distances, topk_indices


class ChunkedKNN:
    def __init__(
        self,
        data_size: int,
        query_size: int,
        topk: int,
        match_device: torch.device,
        distance_metric: str = "angular",
        match_dtype: torch.dtype = torch.float16,
        quantized_via_ao: bool = True,
    ):
        """
        Initialize the ChunkedKNN indexer to do batched k-nearest neighbors search.

        Args:
            data_size (int): The total number of data entries.
            query_size (int): The total number of query entries.
            topk (int): The number of nearest neighbors to return.
            match_device (torch.device): The device to use for matching.
            distance_metric (str): The distance metric to use ('angular', 'euclidean', or 'manhattan').
            match_dtype (torch.dtype): The data type to use for matching.
            quantized (bool): Whether to use quantized arithmetic.
        """
        self.data_size = data_size
        self.query_size = query_size
        self.topk = topk
        self.match_device = match_device
        self.distance_metric = distance_metric

        if (
            quantized_via_ao
            and match_dtype != torch.float32
            and match_device.type == "cpu"
        ):
            print(
                "CPU Quantization requires float32 data type. Forcing float32 match_dtype."
            )
            match_dtype = torch.float32

        self.match_dtype = match_dtype
        self.quantized = quantized_via_ao

        self.big_better = self.distance_metric == "angular"
        ind_dtype = torch.int64 if self.data_size < 2**31 else torch.int32
        self.knn_indices = torch.empty(
            (self.query_size, self.topk), device=self.match_device, dtype=ind_dtype
        )
        self.knn_distances = torch.full(
            (self.query_size, self.topk),
            -torch.inf if self.big_better else torch.inf,
            device=self.match_device,
            dtype=self.match_dtype,
        )

        self.prepared_data = None
        self.curr_end = 0  # 0 indexed

    def set_data_chunk(self, data_chunk: Tensor):
        """
        Set the current data chunk. Shape (N, D).

        Args:
            data_chunk (Tensor): The data chunk to set.
        """
        data_chunk = data_chunk.to(self.match_dtype).to(self.match_device)

        # it is easier to track the current end index than the start index
        self.curr_end += data_chunk.shape[0]
        self.curr_start = self.curr_end - data_chunk.shape[0]

        # Quantize the data if needed
        if self.quantized and self.match_device.type == "cpu":
            self.prepared_data = quant_model_data(data_chunk)
        else:
            self.prepared_data = data_chunk

    def query_all(self, query: Tensor):
        """
        Perform k-nearest neighbors search on all queries.

        Args:
            query (Tensor): The queries to search.
        """
        # throw an error if the data chunk is not set
        if self.prepared_data is None:
            raise ValueError("Data chunk is not set.")

        # send query to match device and dtype
        query = query.to(self.match_dtype).to(self.match_device)

        if self.quantized and self.match_device.type == "cpu":
            knn_dists_chunk, knn_inds_chunk = topk_quantized_data(
                query, self.prepared_data, self.topk
            )
        else:
            knn_dists_chunk, knn_inds_chunk = knn_batch(
                self.prepared_data, query, self.topk, self.distance_metric
            )

        # chunk indices -> global indices
        knn_inds_chunk += self.curr_start

        # Merge the old and new top-k indices and distances
        merged_knn_dists = torch.cat((self.knn_distances, knn_dists_chunk), dim=1)
        merged_knn_inds = torch.cat((self.knn_indices, knn_inds_chunk), dim=1)

        # get the overall topk
        topk_indices = torch.topk(
            merged_knn_dists, self.topk, dim=1, largest=self.big_better, sorted=False
        )[1]
        self.knn_indices = torch.gather(merged_knn_inds, 1, topk_indices)
        self.knn_distances = torch.gather(merged_knn_dists, 1, topk_indices)

    def query_chunk(self, query_chunk: Tensor, query_start: int):
        """
        Perform k-nearest neighbors search on a contiguous query chunk.

        Args:
            query_chunk (Tensor): The query chunk to search.
            query_start (int): The start index of the query chunk.

        """

        # throw an error if the data chunk is not set
        if self.prepared_data is None:
            raise ValueError("Data chunk is not set.")

        query_end = query_start + query_chunk.shape[0]
        query_chunk = query_chunk.to(self.match_dtype).to(self.match_device)

        if self.quantized and self.match_device.type == "cpu":
            knn_distances_chunk, knn_indices_chunk = topk_quantized_data(
                query_chunk, self.prepared_data, self.topk
            )
        else:
            knn_distances_chunk, knn_indices_chunk = knn_batch(
                self.prepared_data, query_chunk, self.topk, self.distance_metric
            )

        knn_indices_chunk += self.curr_start

        # Merge the old and new top-k indices and distances
        old_knn_indices = self.knn_indices[query_start:query_end]
        old_knn_distances = self.knn_distances[query_start:query_end]
        merged_knn_distances = torch.cat(
            (old_knn_distances, knn_distances_chunk), dim=1
        )
        merged_knn_indices = torch.cat((old_knn_indices, knn_indices_chunk), dim=1)
        topk_indices = torch.topk(
            merged_knn_distances,
            self.topk,
            dim=1,
            largest=self.big_better,
            sorted=False,
        )[1]
        self.knn_indices[query_start:query_end] = torch.gather(
            merged_knn_indices, 1, topk_indices
        )
        self.knn_distances[query_start:query_end] = torch.gather(
            merged_knn_distances, 1, topk_indices
        )

    def retrieve_topk(
        self,
    ) -> Tuple[Tensor, Tensor]:
        """
        Retrieve the top-k nearest neighbors indices and distances.

        Args:
            device (torch.device): The device to return the results on.

        Returns:
            Tensor: The indices of the nearest neighbors.
        """
        # sort the topk indices and distances
        topk_indices = torch.topk(
            self.knn_distances, self.topk, dim=1, largest=self.big_better, sorted=True
        )[1]
        knn_indices = torch.gather(self.knn_indices, 1, topk_indices)
        knn_distances = torch.gather(self.knn_distances, 1, topk_indices)

        return knn_indices, knn_distances


@torch.jit.script
def get_local_orientation_grid(
    semi_edge_in_degrees: float,
    kernel_radius_in_steps: int,
    axial_grid_dimension: int = 3,
) -> Tensor:
    """
    Get the local orientation grid using cubochoric coordinates. The grid will
    be a cube with 2*semi_edge_in_degrees side length and (2*kernel_radius +
    1)^3 grid points.

    The cubochoric box with the identity orientation centered at the origin
    extends from -0.5 * (pi)^(2/3) to 0.5 * (pi)^(2/3) along each axis. The
    points on the outside surface are 180 degree rotations.

    A convenient equal volume subgrid around the origin can be mapped to
    quaternions and used to explore the space of orientations around the initial
    guess via rotational composition.

    Args:
        :semi_edge_in_degrees (float): The semi-edge of the grid in degrees.
        :kernel_radius_in_steps (int): Divisions per side of the kernel.
        :axial_grid_dimension (int): Defaults to 3 (complete outer product). 2
        excluded cube corners. 1 only has the axial points.

    Returns:
        cu (Tensor): The local orientation grid in cubochoric coordinates.

    """

    # get the cubochoric grid edge length
    semi_edge_cubochoric = semi_edge_in_degrees * torch.pi ** (2 / 3) / 360.0

    # make a meshgrid of cubochoric coordinates
    cu = torch.linspace(
        -semi_edge_cubochoric,
        semi_edge_cubochoric,
        2 * kernel_radius_in_steps + 1,
    )

    # get cartesian product meshgrid
    cu = torch.stack(torch.meshgrid(cu, cu, cu, indexing="ij"), dim=-1).reshape(-1, 3)

    if axial_grid_dimension == 3:
        pass
    elif axial_grid_dimension == 2:
        # biaxial grid: points require at least 1 zero
        mask = torch.any(cu == 0, dim=-1)
        cu = cu[mask]
    elif axial_grid_dimension == 1:
        # uniaxial grid: points require at least 2 zeros
        mask = torch.sum(cu == 0, dim=-1) >= 2
        cu = cu[mask]
    else:
        raise ValueError("axial_grid_dimension must be 1, 2, or 3.")
    return cu


def orientation_grid_refinement(
    master_patterns: Union[MasterPattern, List[MasterPattern]],
    geometry: EBSDGeometry,
    experiment_patterns: ExperimentPatterns,
    grid_semi_edge_in_degrees: float,
    batch_size: int,
    virtual_binning: int = 1,
    n_iter: int = 3,
    axial_grid_dimension: int = 3,
    kernel_radius_in_steps: int = 1,
    shrink_factor: float = 0.5,
    average_pattern_center: bool = True,
    match_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Refine the orientation of the EBSD patterns.

    Args:
        master_patterns (Union[MasterPattern, List[MasterPattern]]): The master patterns.
        geometry (EBSDGeometry): The EBSD geometry.
        experiment_patterns (ExperimentPatterns): The experiment patterns.
        batch_size (int): The batch size.
        virtual_binning (int): The virtual binning.
        n_iter (int): The number of iterations.
        grid_semi_edge_in_degrees (float): The semi-edge of the grid in degrees.
        kernel_radius_in_steps (int): The kernel radius in steps.
        average_pattern_center (bool): Assume all patterns come from a point source.

    Returns:
        None. The experiment orientations are updated in place.

    """

    # check that the indexing results have been combined
    if not hasattr(experiment_patterns, "orientations"):
        raise ValueError("The experiment patterns must be indexed before refinement.")

    # create a dictionary object for each master pattern
    if isinstance(master_patterns, MasterPattern):
        master_patterns = [master_patterns]

    # get the coordinates of the detector pixels in the sample reference frame
    detector_coords = geometry.get_coords_sample_frame(
        binning=(virtual_binning, virtual_binning)
    ).view(-1, 3)

    if not average_pattern_center:
        # broadcast subtract the sample scan coordinates from the detector coords
        detector_coords = detector_coords[None, :, :] + torch.cat(
            [
                experiment_patterns.spatial_coords[:, None, :],
                torch.zeros_like(experiment_patterns.spatial_coords[:, None, [-1]]),
            ],
            dim=-1,
        )

    # normalize the detector coordinates to be unit vectors
    detector_coords = detector_coords / detector_coords.norm(dim=-1, keepdim=True)

    # get a list of indices into the experiment patterns for each phase
    phase_indices = experiment_patterns.get_indices_per_phase()

    all_dots = []

    for i, indices in enumerate(phase_indices):
        mp = master_patterns[i]
        pb = progressbar(
            list(torch.split(indices, batch_size)),
            prefix=f"REFINE MP {i+1:01d}/{len(master_patterns):01d} ",
        )

        # make a reference
        cu_grid = (
            get_local_orientation_grid(
                grid_semi_edge_in_degrees,
                kernel_radius_in_steps,
                axial_grid_dimension,
            )
            .view(1, -1, 3)
            .to(detector_coords.device)
        )

        dots = torch.zeros(
            (experiment_patterns.n_patterns, n_iter),
            device=detector_coords.device,
            dtype=match_dtype,
        )

        for indices_batch in pb:
            # get the local orientation grid
            # (N_EXP_PATS, N_GRID_POINTS, 3)
            cu_grid_current = cu_grid.clone().repeat(len(indices_batch), 1, 1)

            # get the experiment patterns for this batch
            # (N_EXP_PATS, H, W)
            exp_pats = experiment_patterns.get_patterns(
                indices_batch, binning=virtual_binning
            ).to(detector_coords.device)

            # reshape from (N_EXP_PATS, H, W) to (N_EXP_PATS, H*W)
            exp_pats = exp_pats.view(exp_pats.shape[0], -1)

            # subtract the mean from the patterns
            exp_pats = exp_pats - torch.mean(exp_pats, dim=-1, keepdim=True)

            # get the current orientations
            # shape (N_EXP_PATS, 4)
            qu_current = experiment_patterns.get_orientations(indices_batch)

            for j in range(n_iter):
                # convert the cu_grid to quaternions
                qu_grid = cu2qu(cu_grid_current)

                # augment with the grid via broadcasted quaternion multiplication
                # shape (N_EXP_PATS, N_GRID_POINTS, 4)
                qu_augmented = qu_prod(qu_grid, qu_current[:, None, :])

                # make sure the quaternions are in the fundamental zone
                qu_augmented = ori_to_fz_laue(qu_augmented, mp.laue_group)

                # apply the qu_augmented to the detector coordinates
                # shape (N_EXP_PATS, N_SIM_PATS, N_DETECTOR_PIXELS, 3)
                if average_pattern_center:
                    detector_coords_rotated = qu_apply(
                        qu_augmented[:, :, None, :], detector_coords[None, None, :, :]
                    )
                else:
                    detector_coords_rotated = qu_apply(
                        qu_augmented[:, :, None, :],
                        detector_coords[indices_batch][:, None, :, :],
                    )

                # interpolate the master pattern
                # shape (N_EXP_PATS, N_SIM_PATS, N_DETECTOR_PIXELS)
                sim_pats = mp.interpolate(
                    detector_coords_rotated,
                    mode="bilinear",
                    align_corners=True,
                    normalize_coords=False,  # already normalized above
                    virtual_binning=virtual_binning,  # coarser grid requires blur of MP
                ).squeeze()

                # zero mean the sim_pats
                sim_pats = sim_pats - torch.mean(sim_pats, dim=-1, keepdim=True)

                # use einsum to do the dot product
                # (N_EXP_PATS, N_DETECTOR_PIXELS) & (N_EXP_PATS, N_SIM_PATS, N_DETECTOR_PIXELS)
                dot_products = torch.einsum(
                    "ij,ikj->ik",
                    exp_pats.to(match_dtype),
                    sim_pats.to(match_dtype),
                )

                # get the best dot product
                # shape (N_EXP_PATS,)
                best_dots, best_indices = torch.max(dot_products, dim=-1)

                # update the dots
                dots[indices_batch, j] = best_dots

                # update center of the grid
                qu_current = qu_augmented[
                    torch.arange(qu_current.shape[0]), best_indices
                ]

                # update the grid size is the center was the best
                cu_grid_current[
                    (best_indices == ((cu_grid_current.shape[1] - 1) / 2))
                ] *= shrink_factor

            # update the experiment patterns
            experiment_patterns.set_orientations(qu_current, indices_batch)

        all_dots.append(dots)

    return all_dots


def dictionary_index_orientations(
    master_patterns,
    geometry,
    experiment_patterns,
    signal_mask=None,
    subtract_exp_pat_mean=True,
    experiment_chunk_size=256,
    dictionary_resolution_degrees=0.5,
    dictionary_chunk_size=256,
    virtual_binning=1,
    top_k_matches=1,
    distance_metric="angular",
    match_dtype=torch.float32,
    match_device=None,
    quantization=None,
):
    """
    Enhanced dictionary indexing with GPU quantization support.

    Args:
        master_patterns: The master patterns
        geometry: The EBSD geometry object
        experiment_patterns: The experiment patterns
        signal_mask: The signal mask
        subtract_exp_pat_mean: Whether to zero mean each experimental pattern
        experiment_chunk_size: Size of experiment chunks
        dictionary_resolution_degrees: Resolution of the dictionary in degrees
        dictionary_chunk_size: Size of dictionary chunks
        virtual_binning: Virtual binning factor
        top_k_matches: Number of top matches to retrieve
        distance_metric: The distance metric ("angular", "euclidean", "manhattan")
        match_dtype: Data type for matching (torch.float32, torch.float16)
        quantization: Quantization strategy ("int8", "fp16", "fp8", None)
        average_pattern_center: Whether to consider all patterns from same origin

    Returns:
        None - results are stored in experiment_patterns object
    """
    # Ensure master_patterns is a list
    if isinstance(master_patterns, MasterPattern):
        master_patterns = [master_patterns]

    # Process each master pattern
    for i, mp in enumerate(master_patterns):
        # Generate orientation dictionary
        ori_tensor = sample_ori_fz_laue_angle(
            laue_id=mp.laue_group,
            angular_resolution_deg=dictionary_resolution_degrees,
            device=mp.master_pattern.device,
            permute=True,
        )

        if match_device is None:
            match_device = mp.master_pattern.device

        # Create enhanced KNN object
        knn = ChunkedKNN(
            data_size=len(ori_tensor),
            query_size=experiment_patterns.n_patterns,
            topk=top_k_matches,
            match_device=match_device,
            distance_metric=distance_metric,
            match_dtype=match_dtype,
            quantized_via_ao=quantization,
        )

        # Define synchronization function
        if experiment_patterns.patterns.device.type == "cuda":
            sync = torch.cuda.synchronize
        elif experiment_patterns.patterns.device.type == "mps":
            sync = torch.mps.synchronize
        elif (
            experiment_patterns.patterns.device.type == "xpu"
            or experiment_patterns.patterns.device.type == "xla"
        ):
            sync = torch.xpu.synchronize
        else:
            sync = lambda: None

        # Get detector coordinates
        detector_coords = geometry.get_coords_sample_frame(
            binning=(virtual_binning, virtual_binning)
        )

        # Normalize detector coordinates
        detector_coords = detector_coords / detector_coords.norm(dim=-1, keepdim=True)

        # Progress bar for dictionary chunks
        pb = progressbar(
            list(torch.split(ori_tensor, dictionary_chunk_size)),
            prefix=f"INDX MP {i+1:01d}/{len(master_patterns):01d} ",
        )

        # Process each orientation batch
        for ori_batch in pb:
            # Rotate detector pixel positions using orientations
            batch_rotated_coords = qu_apply(
                ori_batch[:, None, :],
                detector_coords[None, ...],
            )

            # Interpolate master pattern at rotated coordinates
            simulated_patterns = mp.interpolate(
                batch_rotated_coords,
                mode="bilinear",
                align_corners=False,
                normalize_coords=False,
                virtual_binning=virtual_binning,
            ).squeeze()

            # Reshape simulated patterns
            simulated_patterns = simulated_patterns.view(len(ori_batch), -1)

            # Apply signal mask if provided
            if signal_mask is not None:
                simulated_patterns = simulated_patterns[:, signal_mask.flatten()]

            # Zero-mean the simulated patterns
            simulated_patterns = simulated_patterns - torch.mean(
                simulated_patterns, dim=-1, keepdim=True
            )

            # Set the data chunk for KNN
            knn.set_data_chunk(simulated_patterns)

            # Process all experimental patterns at once if chunk size is large enough
            if experiment_chunk_size >= experiment_patterns.n_patterns:
                # print(f"Num patterns: {experiment_patterns.n_patterns}")
                # print(f"Experiment chunk size: {experiment_chunk_size}")
                exp_pats = (
                    experiment_patterns.get_patterns(
                        torch.arange(experiment_patterns.n_patterns),
                        binning=virtual_binning,
                    )
                    .view(experiment_patterns.n_patterns, -1)
                    .to(match_device)
                )

                # Apply signal mask if provided
                if signal_mask is not None:
                    exp_pats = exp_pats[:, signal_mask.flatten().to(match_device)]

                # Zero-mean the experimental patterns
                if subtract_exp_pat_mean:
                    exp_pats = exp_pats - torch.mean(exp_pats, dim=-1, keepdim=True)

                # Query KNN with all patterns
                knn.query_all(exp_pats)
                sync()
            else:
                # Process experimental patterns in chunks
                for exp_pat_batch_indices in list(
                    torch.split(
                        torch.arange(experiment_patterns.n_patterns),
                        experiment_chunk_size,
                    )
                ):
                    # Get chunk of experimental patterns
                    query_chunk = (
                        experiment_patterns.get_patterns(
                            exp_pat_batch_indices,
                            binning=virtual_binning,
                        )
                        .view(len(exp_pat_batch_indices), -1)
                        .to(match_device)
                    )

                    # Apply signal mask if provided
                    if signal_mask is not None:
                        query_chunk = query_chunk[
                            :, signal_mask.flatten().to(match_device)
                        ]

                    # Zero-mean the chunk
                    if subtract_exp_pat_mean:
                        query_chunk = query_chunk - torch.mean(
                            query_chunk, dim=-1, keepdim=True
                        )

                    # Query KNN with chunk
                    knn.query_chunk(
                        query_chunk,
                        query_start=exp_pat_batch_indices[0],
                    )
                    sync()

        # Get the matches and distances
        matches, metric_values = knn.retrieve_topk()

        # Store results in experiment patterns object
        experiment_patterns.set_raw_indexing_results(
            ori_tensor[matches],
            metric_values,
            phase_id=i,
        )

    # Combine indexing results
    experiment_patterns.combine_indexing_results(
        higher_is_better=(distance_metric == "angular")
    )


def compute_pca_components_covmat(
    master_patterns: Union[MasterPattern, List[MasterPattern]],
    geometry: EBSDGeometry,
    n_pca_components: int,
    signal_mask: Optional[Tensor] = None,
    dictionary_resolution_learn_deg: float = 0.5,
    dictionary_chunk_size: int = 256,
    virtual_binning: int = 1,
) -> Tensor:
    # create a dictionary object for each master pattern
    if isinstance(master_patterns, MasterPattern):
        master_patterns = [master_patterns]

    for i, mp in enumerate(master_patterns):
        # get an orientation dictionary
        ori_tensor = sample_ori_fz_laue_angle(
            laue_id=mp.laue_group,
            angular_resolution_deg=dictionary_resolution_learn_deg,
            device=mp.master_pattern.device,
        )

        # get the dimensionality of each pattern
        if signal_mask is not None:
            n_pixels = signal_mask.sum().item()

        # make an object to do the PCA
        pcacovmat = OnlineCovMatrix(
            n_features=n_pixels,
        ).to(mp.master_pattern.device)

        # get a helper function to sync devices
        if mp.master_pattern.device.type == "cuda":
            sync = torch.cuda.synchronize
        elif mp.master_pattern.device.type == "mps":
            sync = torch.mps.synchronize
        elif (
            mp.master_pattern.device.type == "xpu"
            or mp.master_pattern.device.type == "xla"
        ):
            sync = torch.xpu.synchronize
        else:
            sync = lambda: None

        pb = progressbar(
            list(torch.split(ori_tensor, dictionary_chunk_size)),
            prefix=f" PCA MP {i+1:01d}/{len(master_patterns):01d} ",
        )

        detector_coords = geometry.get_coords_sample_frame(
            binning=(virtual_binning, virtual_binning)
        )[None, ...]

        if not signal_mask is None:
            detector_coords = detector_coords[:, signal_mask.flatten()]

        # iterate over the dictionary in chunks
        for ori_batch in pb:
            # use orientations to rotate the rays to detector pixel positions
            # (n_ori, 4) -> (n_ori, 1, 4) and (H*W, 3) -> (1, H*W, 3) for broadcasting
            batch_rotated_coords = qu_apply(
                ori_batch[:, None, :],
                detector_coords,
            )

            # interpolate the master pattern at the rotated coordinates
            simulated_patterns = mp.interpolate(
                batch_rotated_coords,
                mode="bilinear",
                align_corners=False,
                normalize_coords=True,  # not already normalized above
                virtual_binning=virtual_binning,  # coarser grid requires blur of MP
            ).squeeze()

            # flatten simulated patterns to (n_ori, H*W)
            simulated_patterns = simulated_patterns.view(len(ori_batch), -1)

            # must remove mean from each simulated pattern
            simulated_patterns = simulated_patterns - torch.mean(
                simulated_patterns,
                dim=-1,
                keepdim=True,
            )

            # update the PCA object
            pcacovmat(simulated_patterns)

            sync()

        # get the eigenvectors and eigenvalues
        eigenvectors = pcacovmat.get_eigenvectors()

        # trim the eigenvectors to the number of components
        # they are returned in ascending order of eigenvalue
        pca_matrix = eigenvectors[:, -n_pca_components:]

        return pca_matrix


def pca_dictionary_index_orientations(
    master_patterns: Union[MasterPattern, List[MasterPattern]],
    geometry: EBSDGeometry,
    experiment_patterns: ExperimentPatterns,
    pca_matrix: Tensor,
    signal_mask: Optional[Tensor] = None,
    subtract_exp_pat_mean: bool = True,
    experiment_chunk_size: int = 256,
    dictionary_resolution_index_deg: float = 0.5,
    dictionary_chunk_size: int = 256,
    virtual_binning: int = 1,
    top_k_matches: int = 1,
    distance_metric: str = "angular",
    match_dtype: torch.dtype = torch.float16,
    match_device: Optional[torch.device] = None,
    quantized_via_ao: bool = True,
) -> None:
    # create a dictionary object for each master pattern
    if isinstance(master_patterns, MasterPattern):
        master_patterns = [master_patterns]

    if match_device is None:
        match_device = master_patterns[0].device

    # we could save some compute by only computing the cosine vectors once
    # per unique Laue group ID and reuse those coordinates for all master patterns
    # with the same Laue group ID. But more than a few phases in a sample is rare.
    for i, mp in enumerate(master_patterns):
        ori_tensor = sample_ori_fz_laue_angle(
            laue_id=mp.laue_group,
            angular_resolution_deg=dictionary_resolution_index_deg,
            device=mp.master_pattern.device,
        )
        # make an object to do the pattern comparisons
        knn = ChunkedKNN(
            data_size=len(ori_tensor),
            query_size=experiment_patterns.n_patterns,
            topk=top_k_matches,
            match_device=match_device,
            distance_metric=distance_metric,
            match_dtype=match_dtype,
            quantized_via_ao=quantized_via_ao,
        )

        # get a helper function to sync devices
        if mp.master_pattern.device.type == "cuda":
            sync = torch.cuda.synchronize
        elif mp.master_pattern.device.type == "mps":
            sync = torch.mps.synchronize
        elif (
            mp.master_pattern.device.type == "xpu"
            or mp.master_pattern.device.type == "xla"
        ):
            sync = torch.xpu.synchronize
        else:
            sync = lambda: None

        detector_coords = geometry.get_coords_sample_frame(
            binning=(virtual_binning, virtual_binning)
        )[None, ...]

        if not signal_mask is None:
            detector_coords = detector_coords[:, signal_mask.flatten()]

        # project the dictionary onto the PCA components
        pb = progressbar(
            list(torch.split(ori_tensor, dictionary_chunk_size)),
            prefix=f"INDX MP {i+1:01d}/{len(master_patterns):01d} ",
        )

        pca_matrix = pca_matrix.to(match_device)

        # iterate over the dictionary in chunks
        for ori_batch in pb:
            # use orientations to rotate the rays to detector pixel positions
            # (n_ori, 4) -> (n_ori, 1, 4) and (H*W, 3) -> (1, H*W, 3) for broadcasting
            batch_rotated_coords = qu_apply(
                ori_batch[:, None, :],
                detector_coords,
            )

            # interpolate the master pattern at the rotated coordinates
            simulated_patterns = mp.interpolate(
                batch_rotated_coords,
                mode="bilinear",
                align_corners=False,
                normalize_coords=True,  # not already normalized above
                virtual_binning=virtual_binning,  # coarser grid requires blur of MP
            ).squeeze()

            # flatten simulated patterns to (n_ori, H*W)
            simulated_patterns = simulated_patterns.view(len(ori_batch), -1)

            # must remove mean from each simulated pattern
            simulated_patterns = simulated_patterns - torch.mean(
                simulated_patterns,
                dim=-1,
                keepdim=True,
            )

            # project the simulated patterns onto the PCA components
            simulated_patterns_pca = torch.matmul(
                simulated_patterns.to(match_device), pca_matrix
            )

            # set the data for the KNN object
            knn.set_data_chunk(simulated_patterns_pca)

            # # retrieve and project the experiment patterns
            if experiment_chunk_size > experiment_patterns.n_patterns:
                exp_pats = experiment_patterns.patterns.view(
                    experiment_patterns.n_patterns, -1
                ).to(match_device)
                # query all the experiment patterns
                # first apply the signal mask if provided
                if signal_mask is not None:
                    exp_pats = exp_pats[:, signal_mask.flatten().to(match_device)]

                # subtract the mean from each pattern and query the KNN object
                if subtract_exp_pat_mean:
                    exp_pats = exp_pats - torch.mean(exp_pats, dim=-1, keepdim=True)

                # PCA projection
                exp_pats = torch.matmul(exp_pats, pca_matrix)

                # query the KNN object
                knn.query_all(exp_pats)

                # # synchronize the device for the progress bar
                # sync()
            else:
                # loop over the experiment patterns in chunks and feed them to the KNN object
                for exp_pat_batch_indices in list(
                    torch.split(
                        torch.arange(experiment_patterns.n_patterns),
                        experiment_chunk_size,
                    )
                ):
                    # get a chunk of experiment patterns
                    query_chunk = (
                        experiment_patterns.get_patterns(
                            exp_pat_batch_indices,
                            binning=virtual_binning,
                        )
                        .view(len(exp_pat_batch_indices), -1)
                        .to(match_device)
                    )

                    # apply the signal mask if provided
                    if signal_mask is not None:
                        query_chunk = query_chunk[
                            :, signal_mask.flatten().to(match_device)
                        ]

                    # subtract the mean
                    query_chunk = query_chunk - torch.mean(
                        query_chunk, dim=-1, keepdim=True
                    )

                    # project the query chunk onto the PCA components
                    query_chunk = torch.matmul(query_chunk, pca_matrix)

                    # query the KNN object
                    knn.query_chunk(
                        query_chunk.view(len(query_chunk), -1),
                        query_start=exp_pat_batch_indices[0],
                    )

                    # # synchronize the device for the progress bar
                    # sync()

        # get the matches and distances
        matches, metric_values = knn.retrieve_topk()

        # set the matches and distances in the experiment patterns object
        experiment_patterns.set_raw_indexing_results(
            ori_tensor[matches],
            metric_values,
            phase_id=i,
        )

    # combine the indexing results
    experiment_patterns.combine_indexing_results(
        higher_is_better=(distance_metric == "angular")
    )


"""
:Author: Zachary T. Varley
:Year: 2025
:License: MIT

Oja with exponentially decaying learning rate (batch count known before hand).

This module implements the batched version of Oja's update rule for PCA. Each
batch is used to update the estimates of the top-k eigenvectors of the
covariance matrix. Then a QR decomposition is performed to re-orthogonalize the
estimates. QR is O(mn^2) compute via Schwarz-Rutishauser for m x n matrices.

References:

Allen-Zhu, Zeyuan, and Yuanzhi Li. “First Efficient Convergence for Streaming
K-PCA: A Global, Gap-Free, and Near-Optimal Rate.” 2017 IEEE 58th Annual
Symposium on Foundations of Computer Science (FOCS), IEEE, 2017, pp. 487–92.
DOI.org (Crossref), https://doi.org/10.1109/FOCS.2017.51.

Tang, Cheng. “Exponentially Convergent Stochastic K-PCA without Variance
Reduction.” Advances in Neural Information Processing Systems, vol. 32, 2019.

"""


class OjaPCAExp(Module):
    def __init__(
        self,
        n_features: int,
        n_components: int,
        total_steps: int,
        initial_eta: float = 0.5,
        final_eta: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        use_oja_plus: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.initial_eta = initial_eta
        self.final_eta = final_eta
        self.total_steps = total_steps
        self.use_oja_plus = use_oja_plus

        # Calculate decay rate
        self.alpha = -torch.log(torch.tensor(final_eta / initial_eta)) / total_steps

        # Initialize parameters
        self.register_buffer("Q", torch.randn(n_features, n_components, dtype=dtype))
        self.register_buffer("step", torch.zeros(1, dtype=torch.int64))

        # For Oja++
        if self.use_oja_plus:
            self.register_buffer(
                "initialized_cols", torch.zeros(n_components, dtype=torch.bool)
            )
            self.register_buffer("next_col_to_init", torch.tensor(0, dtype=torch.int64))

    def get_current_lr(self) -> float:
        return self.initial_eta * torch.exp(-self.alpha * self.step.float()).item()

    def forward(self, x: Tensor) -> float:
        # Get current learning rate
        current_eta = self.get_current_lr()

        # Forward pass and reconstruction
        projection = x @ self.Q
        reconstruction = projection @ self.Q.T
        current_error = torch.mean((x - reconstruction) ** 2).item()

        # Update then Orthonormalize Q_t using QR decomposition
        self.Q.copy_(torch.linalg.qr(self.Q + current_eta * (x.T @ (projection)))[0])

        # Update step counter
        self.step.add_(1)

        # For Oja++, gradually initialize columns
        if self.use_oja_plus and self.next_col_to_init < self.n_components:
            if self.step % (self.n_components // 2) == 0:
                self.Q[:, self.next_col_to_init] = torch.randn(
                    self.n_features, dtype=self.Q.dtype
                )
                self.initialized_cols[self.next_col_to_init] = True
                self.next_col_to_init.add_(1)

        return current_error

    def get_components(self) -> Tensor:
        return self.Q.T

    def transform(self, x: Tensor) -> Tensor:
        return x @ self.Q

    def inverse_transform(self, x: Tensor) -> Tensor:
        return x @ self.Q.T


"""
:Author: Zachary T. Varley
:Year: 2025
:License: MIT

Oja batched update with a fixed learning rate.

This module implements the batched version of Oja's update rule for PCA. Each
batch is used to update the estimates of the top-k eigenvectors of the
covariance matrix. Then a QR decomposition is performed to re-orthogonalize the
estimates. QR is O(mn^2) compute via Schwarz-Rutishauser for m x n matrices.

References:

Allen-Zhu, Zeyuan, and Yuanzhi Li. “First Efficient Convergence for Streaming
K-PCA: A Global, Gap-Free, and Near-Optimal Rate.” 2017 IEEE 58th Annual
Symposium on Foundations of Computer Science (FOCS), IEEE, 2017, pp. 487–92.
DOI.org (Crossref), https://doi.org/10.1109/FOCS.2017.51.

Tang, Cheng. “Exponentially Convergent Stochastic K-PCA without Variance
Reduction.” Advances in Neural Information Processing Systems, vol. 32, 2019.

"""


class OjaPCA(Module):
    def __init__(
        self,
        n_features: int,
        n_components: int,
        eta: float = 0.005,
        dtype: torch.dtype = torch.float32,
        use_oja_plus: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.eta = eta
        self.use_oja_plus = use_oja_plus

        # Initialize parameters
        self.register_buffer("Q", torch.randn(n_features, n_components, dtype=dtype))
        self.register_buffer("step", torch.zeros(1, dtype=torch.int64))

        # For Oja++, we initialize columns gradually
        if self.use_oja_plus:
            self.register_buffer(
                "initialized_cols", torch.zeros(n_components, dtype=torch.bool)
            )
            self.register_buffer("next_col_to_init", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: Tensor) -> None:
        """Update PCA with new batch of data using Oja's algorithm"""
        # Update then Orthonormalize Q_t using QR decomposition
        self.Q.copy_(torch.linalg.qr(self.Q + self.eta * (x.T @ (x @ self.Q)))[0])

        # Update step
        self.step.add_(1)

        # For Oja++, gradually initialize columns
        if self.use_oja_plus and self.next_col_to_init < self.n_components:
            if self.step % (self.n_components // 2) == 0:
                self.Q[:, self.next_col_to_init] = torch.randn(
                    self.n_features, dtype=self.Q.dtype
                )
                self.initialized_cols[self.next_col_to_init] = True
                self.next_col_to_init.add_(1)

    def get_components(self) -> Tensor:
        return self.Q.T

    def transform(self, x: Tensor) -> Tensor:
        return x @ self.Q

    def inverse_transform(self, x: Tensor) -> Tensor:
        return x @ self.Q.T
