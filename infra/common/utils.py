from domain.interface import point
from domain.utils import trim_base64_header
from PIL import Image, ImageOps
import numpy as np
import base64
import cv2 

def data_uri_to_cv2_img(logger, data:str):
    data, _ = trim_base64_header(logger, data)
    if data is None:
        return data
    nparr = np.fromstring(base64.b64decode(data), np.uint8)
    #TODO: use from buffer instead
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) #TODO: switch to TURBOJPEG 
    return img

def pad_to_square(image, mean_color = (123, 116, 103)):
    """
    Pads the given image to make it square using the specified mean color.
    
    Args:
        image (PIL.Image.Image): The image to pad.
        mean_color (tuple): The RGB mean color to use for padding.
        
    Returns:
        PIL.Image.Image: The padded square image.
    """
    width, height = image.size
    if width == height:
        return image  # Already square
    
    # Determine the size of the new square
    max_side = max(width, height)
    
    # Calculate padding
    delta_w = max_side - width
    delta_h = max_side - height
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2)
    )
    
    # Apply padding
    padded_image = ImageOps.expand(image, padding, fill=mean_color)
    return padded_image

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    '''
        debug use
        draw the yaw, pitch, roll axis on the given image
        returns the drawn image
    '''
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    (x1, y1), (x2, y2), (x3, y3) = get_axis(yaw, pitch, roll, tdx, tdy, size)

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img

def get_axis(yaw, pitch, roll, tdx, tdy, size=100):
    pitch = -(pitch * np.pi / 180)
    yaw = -(yaw * np.pi / 180)
    roll = -(roll * np.pi / 180)

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy
    return (x1, y1), (x2, y2), (x3, y3)

def get_axis_points(yaw, pitch, roll, tdx, tdy, size=100, point_cnt=5):
    (x1, y1), (x2, y2), (x3, y3) = get_axis(yaw, pitch, roll, tdx, tdy, size)
    x_axis = [point(tdx, tdy)]
    y_axis = []
    z_axis = []
    for i in range(1, point_cnt + 1):
        step = i/point_cnt
        x_axis.append(point(tdx + (x1 - tdx) * step, tdy + (y1 - tdy) * step))
        y_axis.append(point(tdx + (x2 - tdx) * step, tdy + (y2 - tdy) * step))
        z_axis.append(point(tdx + (x3 - tdx) * step, tdy + (y3 - tdy) * step))
    return [x_axis, y_axis, z_axis]

def mean_image_subtraction(images, means=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    image normalization *RGB*
    :param images: bs * w * h * channel 
    :param means:
    :return:
    '''
    num_channels = images.shape[2]
    if len(means) != num_channels:
      print(len(means))
      print(num_channels)
      raise ValueError('len(means) must match the number of channels')
    for i in range(num_channels):
        images[:,:,i] -= means[i]
        images[:,:,i] /= std[i]
    return images 