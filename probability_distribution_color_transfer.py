from pathlib import Path
import numpy as np
import cv2

### THIS IS CODE FROM OPEN SOURCE ###
### https://github.com/pengbo-learn/python-color-transfer/tree/f96017e11a352aea1c75241f9137a511211f6eac ###
### NOT MY ORIGINAL WORK ###

class ColorTransfer:
    """ Methods for color transfer of images. """

    def __init__(self, eps=1e-6, m=6, c=3):
        """Hyper parameters.
        Attributes:
            c: dim of rotation matrix, 3 for oridnary img.
            m: num of random orthogonal rotation matrices.
            eps: prevents from zero dividing.
        """
        
        rotation_matrices = [
            [
                [1.000000, 0.000000, 0.000000],
                [0.000000, 1.000000, 0.000000],
                [0.000000, 0.000000, 1.000000],
            ],
            [
                [0.333333, 0.666667, 0.666667],
                [0.666667, 0.333333, -0.666667],
                [-0.666667, 0.666667, -0.333333],
            ],
            [
                [0.577350, 0.211297, 0.788682],
                [-0.577350, 0.788668, 0.211352],
                [0.577350, 0.577370, -0.577330],
            ],
            [
                [0.577350, 0.408273, 0.707092],
                [-0.577350, -0.408224, 0.707121],
                [0.577350, -0.816497, 0.000029],
            ],
            [
                [0.332572, 0.910758, 0.244778],
                [-0.910887, 0.242977, 0.333536],
                [-0.244295, 0.333890, -0.910405],
            ],
            [
                [0.243799, 0.910726, 0.333376],
                [0.910699, -0.333174, 0.244177],
                [-0.333450, -0.244075, 0.910625],
            ],
            # [[-0.109199, 0.810241, 0.575834], [0.645399, 0.498377, -0.578862], [0.756000, -0.308432, 0.577351]],
            # [[0.759262, 0.649435, -0.041906], [0.143443, -0.104197, 0.984158], [0.634780, -0.753245, -0.172269]],
            # [[0.862298, 0.503331, -0.055679], [-0.490221, 0.802113, -0.341026], [-0.126988, 0.321361, 0.938404]],
            # [[0.982488, 0.149181, 0.111631], [0.186103, -0.756525, -0.626926], [-0.009074, 0.636722, -0.771040]],
            # [[0.687077, -0.577557, -0.440855], [0.592440, 0.796586, -0.120272], [-0.420643, 0.178544, -0.889484]],
            # [[0.463791, 0.822404, 0.329470], [0.030607, -0.386537, 0.921766], [-0.885416, 0.417422, 0.204444]],
        ]
        
        self.eps = eps
        self.rotation_matrices = [np.array(x) for x in rotation_matrices]

    def pdf_transfer(self, img_arr_in=None, img_arr_ref=None, regrain=False):
        """Apply probability density function transfer.
        img_o = t(img_i) so that f_{t(img_i)}(r, g, b) = f_{img_r}(r, g, b),
        where f_{img}(r, g, b) is the probability density function of img's rgb values.
        Args:
            img_arr_in: bgr numpy array of input image.
            img_arr_ref: bgr numpy array of reference image.
        Returns:
            img_arr_out: transfered bgr numpy array of input image.
        """

        # reshape (h, w, c) to (c, h*w)
        [h, w, c] = img_arr_in.shape
        reshape_arr_in = img_arr_in.reshape(-1, c).transpose() / 255.0
        reshape_arr_ref = img_arr_ref.reshape(-1, c).transpose() / 255.0
        # pdf transfer
        reshape_arr_out = self.pdf_transfer_nd(arr_in=reshape_arr_in,
                                                arr_ref=reshape_arr_ref)
        # reshape (c, h*w) to (h, w, c)
        reshape_arr_out[reshape_arr_out < 0] = 0
        reshape_arr_out[reshape_arr_out > 1] = 1
        reshape_arr_out = (255.0 * reshape_arr_out).astype("uint8")
        img_arr_out = reshape_arr_out.transpose().reshape(h, w, c)
        if regrain:
            img_arr_out = self.RG.regrain(img_arr_in=img_arr_in,
                                            img_arr_col=img_arr_out)
        return img_arr_out

    def pdf_transfer_nd(self, arr_in=None, arr_ref=None, step_size=1):
        """Apply n-dim probability density function transfer.
        Args:
            arr_in: shape=(n, x).
            arr_ref: shape=(n, x).
            step_size: arr = arr + step_size * delta_arr.
        Returns:
            arr_out: shape=(n, x).
        """
        # n times of 1d-pdf-transfer
        arr_out = np.array(arr_in)
        for rotation_matrix in self.rotation_matrices:
            rot_arr_in = np.matmul(rotation_matrix, arr_out)
            rot_arr_ref = np.matmul(rotation_matrix, arr_ref)
            rot_arr_out = np.zeros(rot_arr_in.shape)
            for i in range(rot_arr_out.shape[0]):
                rot_arr_out[i] = self._pdf_transfer_1d(rot_arr_in[i],
                                                        rot_arr_ref[i])
            # func = lambda x, n : self._pdf_transfer_1d(x[:n], x[n:])
            # rot_arr = np.concatenate((rot_arr_in, rot_arr_ref), axis=1)
            # rot_arr_out = np.apply_along_axis(func, 1, rot_arr, rot_arr_in.shape[1])
            rot_delta_arr = rot_arr_out - rot_arr_in
            delta_arr = np.matmul(
                rotation_matrix.transpose(), rot_delta_arr
            )  # np.linalg.solve(rotation_matrix, rot_delta_arr)
            arr_out = step_size * delta_arr + arr_out
        return arr_out

    def _pdf_transfer_1d(self, arr_in=None, arr_ref=None, n=300):
        """Apply 1-dim probability density function transfer.
        Args:
            arr_in: 1d numpy input array.
            arr_ref: 1d numpy reference array.
            n: discretization num of distribution of image's pixels.
        Returns:
            arr_out: transfered input array.
        """

        arr = np.concatenate((arr_in, arr_ref))
        # discretization as histogram
        min_v = arr.min() - self.eps
        max_v = arr.max() + self.eps
        xs = np.array(
            [min_v + (max_v - min_v) * i / n for i in range(n + 1)])
        hist_in, _ = np.histogram(arr_in, xs)
        hist_ref, _ = np.histogram(arr_ref, xs)
        xs = xs[:-1]
        # compute probability distribution
        cum_in = np.cumsum(hist_in)
        cum_ref = np.cumsum(hist_ref)
        d_in = cum_in / cum_in[-1]
        d_ref = cum_ref / cum_ref[-1]
        # transfer
        t_d_in = np.interp(d_in, d_ref, xs)
        t_d_in[d_in <= d_ref[0]] = min_v
        t_d_in[d_in >= d_ref[-1]] = max_v
        arr_out = np.interp(arr_in, xs, t_d_in)
        return arr_out



input_image = 'images/house.jpeg'
ref_image = 'images/hats.png'

img_arr_in = cv2.imread(input_image)
img_arr_ref = cv2.imread(ref_image)

PT = ColorTransfer()

# Pdf transfer
img_arr_pdf_reg = PT.pdf_transfer(img_arr_in=img_arr_in,
                                  img_arr_ref=img_arr_ref,
                                  regrain=False)


# Show result
cv2.imshow("input", img_arr_in)
cv2.imshow("reference", img_arr_ref)
cv2.imshow("pdf_reg", img_arr_pdf_reg)
cv2.waitKey(10000)
cv2.destroyAllWindows()