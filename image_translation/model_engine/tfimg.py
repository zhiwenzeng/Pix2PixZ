import tensorflow as tf

# RGB TO LAB
def rgb2lab(srgb):
    srgb_pixels = tf.reshape(srgb, [-1, 3])
    linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
    exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    rgb_to_xyz = tf.constant([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334], # R
        [0.357580, 0.715160, 0.119193], # G
        [0.180423, 0.072169, 0.950227], # B
    ])
    xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

    # normalize for D65 white point
    xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

    epsilon = 6/29
    linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
    exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

    # convert to lab
    fxfyfz_to_lab = tf.constant([
        #  l       a       b
        [  0.0,  500.0,    0.0], # fx
        [116.0, -500.0,  200.0], # fy
        [  0.0,    0.0, -200.0], # fz
    ])

    lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

    lab = tf.reshape(lab_pixels, tf.shape(srgb))
    return lab

# LAB TO RGB
def lab2rgb(lab):
    lab_pixels = tf.reshape(lab, [-1, 3])
    # convert to fxfyfz
    lab_to_fxfyfz = tf.constant([
        #   fx      fy        fz
        [1/116.0, 1/116.0,  1/116.0], # l
        [1/500.0,     0.0,      0.0], # a
        [    0.0,     0.0, -1/200.0], # b
    ])
    fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

    # convert to xyz
    epsilon = 6/29
    linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
    exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
    xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

    xyz_to_rgb = tf.constant([
        #     r           g          b
        [ 3.2404542, -0.9692660,  0.0556434], # x
        [-1.5371385,  1.8760108, -0.2040259], # y
        [-0.4985314,  0.0415560,  1.0572252], # z
    ])
    rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
    linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
    exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

    src = tf.reshape(srgb_pixels, tf.shape(lab))
    return src

def preprocess_lab(lab):
    L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
    return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]

def deprocess_lab(L_chan, a_chan, b_chan):
    return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=2)