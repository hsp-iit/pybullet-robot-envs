import numpy as np
import math as m

def goal_distance(a: object, b: object) -> object:
    if not a.shape == b.shape:
        raise AssertionError("distance(): shape of points mismatch")
    return np.linalg.norm(a - b, axis=-1)


def axis_angle_to_quaternion(vec_aa):
    qx = vec_aa[0] * m.sin(vec_aa[3] / 2)
    qy = vec_aa[1] * m.sin(vec_aa[3] / 2)
    qz = vec_aa[2] * m.sin(vec_aa[3] / 2)
    qw = m.cos(vec_aa[3] / 2)
    quat = [qx, qy, qz, qw]
    return quat

def quaternion_to_axis_angle(quat):
    angle = 2 * m.acos(quat[3])
    x = quat[0] / m.sqrt(1 - quat[3] * quat[3])
    y = quat[1] / m.sqrt(1 - quat[3] * quat[3])
    z = quat[2] / m.sqrt(1 - quat[3] * quat[3])
    vec_aa = [x, y, z, angle]
    return vec_aa

def floor_vec(vec):
    r_vec = [0]*len(vec)
    for i, v in enumerate(vec):
        r_vec[i] = np.sign(v) * m.floor(m.fabs(v) * 100) / 100
    return r_vec

def sph_coord(x, y, z):
    ro = m.sqrt(x*x + y*y + z*z)
    theta = m.acos(z/ro)
    phi = m.atan2(y,x)
    return [ro, theta, phi]