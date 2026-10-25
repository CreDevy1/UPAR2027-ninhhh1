"""TCA in linear sRGB, applied before surveillance degradation."""
import numpy as np

M = np.array([[3.2406, -1.5372, -.4986], [-.9689, 1.8758, .0415], [.0557, -.2040, 1.0570]])


def planck_xy(t):
    if t <= 4000:
        x = -.2661239e9/t**3 - .2343589e6/t**2 + .8776956e3/t + .179910
    else:
        x = -3.0258469e9/t**3 + 2.1070379e6/t**2 + .2226347e3/t + .240390
    if t <= 2222:
        y = -1.1063814*x**3 - 1.34811020*x**2 + 2.18555832*x - .20219683
    elif t <= 4000:
        y = -.9549476*x**3 - 1.37418593*x**2 + 2.09137015*x - .16748867
    else:
        y = 3.0817580*x**3 - 5.87338670*x**2 + 3.75112997*x - .37001483
    return x, y


def white_rgb(t):
    x, y = planck_xy(t)
    rgb = M @ np.array([x/y, 1., (1-x-y)/y])
    return rgb / rgb[1]


REF = white_rgb(6500.)


def apply_tca(image, temperature, beta, exposure):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('TCA requires uint8 RGB')
    gain = white_rgb(temperature) / REF
    gain = (gain / gain[1]) ** beta
    x = image.astype(np.float32)/255.
    linear = np.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4)
    linear = np.clip(linear * gain[None, None, :] * exposure, 0, 1)
    out = np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
    return (out*255+.5).astype(np.uint8)


def tca(image, rng=np.random):
    temperature = float(np.exp(rng.uniform(np.log(4000), np.log(10000))))
    beta = float(rng.uniform(.25, 1.))
    exposure = float(np.exp(rng.uniform(np.log(.75), np.log(1.33))))
    return apply_tca(image, temperature, beta, exposure)
