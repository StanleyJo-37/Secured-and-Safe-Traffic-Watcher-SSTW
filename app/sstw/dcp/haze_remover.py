import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

class HazeRemover(object):
  def __init__(self, omega=0.95, t0=0.001, radius=7, r=60, eps=1e-4):
    self.omega = omega
    self.t0 = t0
    self.radius = radius
    self.r = r
    self.eps = eps

  def get_dark_channel(self, img: np.ndarray) -> np.ndarray:
    r, g, b = cv2.split(img)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.radius, self.radius))
    dc = cv2.erode(dc, kernel)
    return dc

  def estimate_atmospheric_light(self, img: np.ndarray, dc: np.ndarray):
    [h, w] = img.shape[:2]
    img_size = h * w

    dc_vec = dc.reshape(img_size)
    img_vec = img.reshape(img_size, 3)

    num_px = int(max(math.floor(img_size * self.t0), 1))
    indices = dc_vec.argsort()
    indices = indices[-num_px::]

    atmospheric_light = np.mean(img_vec[indices, :], axis=0)
    return atmospheric_light

  def get_guided_filter(self, im, p, r, eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

  def estimate_transmission(self, dark_channel, A):
    normalized_dc = dark_channel / np.max(A)
    transmission = 1 - self.omega * normalized_dc
    return np.clip(transmission, self.t0, 1)

  def refine_transmission(self, img: np.ndarray, transmission):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray) / 255.
    transmission = self.get_guided_filter(gray, transmission, self.r, self.eps)
    return transmission

  def recover_scene_radiance(self, img: np.ndarray, transmission: np.ndarray, A: np.ndarray):
    res = np.empty(img.shape,img.dtype);
    t = cv2.max(self.t0, transmission)

    for ind in range(0,3):
        res[:,:,ind] = (img[:,:,ind] - A[ind]) / t + A[ind]

    return res

  def remove_haze(self, img: np.ndarray):
    dark_channel = self.get_dark_channel(img)
    atmospheric_light = self.estimate_atmospheric_light(img, dark_channel)
    transmission = self.refine_transmission(img, self.estimate_transmission(dark_channel, atmospheric_light))

    return self.recover_scene_radiance(img, transmission, atmospheric_light)