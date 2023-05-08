import numpy as np
import imageio
from numpy.linalg import cholesky, eigh

class LinearColourTransfer():
    def __init__(self, source, target, method=['cholesky', 'pca'], norm_eps=1e-6):
        self.source_img = imageio.imread(source, pilmode='RGB').astype(np.float32) / 255.0
        self.target_img = imageio.imread(target, pilmode='RGB').astype(np.float32) / 255.0
        self.method = method
        self.norm_eps = norm_eps

    def transfer(self):
        target_mean = np.mean(self.target_img, axis=(0, 1))
        source_mean = np.mean(self.source_img, axis=(0, 1))

        target_norm = self.target_img - target_mean
        source_norm = self.source_img - source_mean

        # reshape to 3 x N
        target = target_norm.transpose(2, 0, 1).reshape(3, -1)
        source = source_norm.transpose(2, 0, 1).reshape(3, -1)

        # compute covariance matrix
        target_cov = np.cov(target)
        source_cov = np.cov(source)

        # compute the transformation matrix
        if self.method == 'cholesky':
            transform = self.cholesky(target_cov, source_cov)
        elif self.method == 'pca':
            transform = self.pca(target_cov, source_cov)
        else:
            raise NotImplementedError

        # compute the transformed image
        transformed = transform @ source_norm

        # reshape back to H x W x 3
        transformed = transformed.reshape(self.target_img.shape).transpose(1, 2, 0)

        # add the mean back
        transformed += target_mean

        # clip the values
        transformed = np.clip(transformed, 0, 1)

        return transformed

    def cholesky(self, target_cov, source_cov):
        target_cholesky = cholesky(target_cov + self.norm_eps * np.eye(3))
        source_cholesky = cholesky(source_cov + self.norm_eps * np.eye(3))

        # compute the transformation matrix
        transform = target_cholesky @ np.linalg.inv(source_cholesky)
        return transform

    def pca(target_cov, source_cov):
        target_eigval, target_eigvec = eigh(target_cov)
        source_eigval, source_eigvec = eigh(source_cov)

        sqrt_target = np.sqrt(np.diag(target_eigval))
        target_transform = target_eigvec.dot(sqrt_target).dot(target_eigvec.T)
        
        sqrt_source = np.sqrt(np.diag(source_eigval))
        source_transform = source_eigvec.dot(sqrt_source).dot(source_eigvec.T)

        # compute the transformation matrix
        transform = target_transform @ np.linalg.inv(source_transform)
        return transform