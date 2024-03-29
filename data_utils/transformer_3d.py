import numpy as np
from skimage.transform import warp
from transforms3d.euler import euler2mat
from transforms3d.affines import compose



class RandomTranslationRotationZoom3D(object):
    '''
    Data augmentation method.
    Including random translation, rotation and zoom, which keep the shape of input.
    Args:
    - mode: string, consisting of 't','r' or 'z'. Optional methods and 'trz'is default.
            't'-> translation,
            'r'-> rotation,
            'z'-> zoom.
    '''
    def __init__(self, mode='trz',num_class=2):
        self.mode = mode
        self.num_class = num_class

    def __call__(self, sample):
        # image: numpy array
        # label: numpy array
        image = sample['image']
        label = sample['label']
        # get transform coordinate
        if len(image.shape) == 3:
            img_size = image.shape
        else: 
            img_size = image.shape[1:]

        coords0, coords1, coords2 = np.mgrid[:img_size[0], :img_size[1], :
                                             img_size[2]]
        coords = np.array([
            coords0 - img_size[0] / 2, coords1 - img_size[1] / 2,
            coords2 - img_size[2] / 2
        ])
        tform_coords = np.append(coords.reshape(3, -1),
                                 np.ones((1, np.prod(img_size))),
                                 axis=0)
        # transform configuration
        # translation
        if 't' in self.mode:
            translation = [
                0, np.random.uniform(-5, 5),
                np.random.uniform(-5, 5)
            ]
        else:
            translation = [0, 0, 0]

        # rotation
        if 'r' in self.mode:
            rotation = euler2mat(
                np.random.uniform(-5, 5) / 180.0 * np.pi, 0, 0, 'sxyz')
        else:
            rotation = euler2mat(0, 0, 0, 'sxyz')

        # zoom
        if 'z' in self.mode:
            zoom = [
                1, np.random.uniform(0.9, 1.1),
                np.random.uniform(0.9, 1.1)
            ]
        else:
            zoom = [1, 1, 1]

        # compose
        warp_mat = compose(translation, rotation, zoom)

        # transform
        w = np.dot(warp_mat, tform_coords)
        w[0] = w[0] + img_size[0] / 2
        w[1] = w[1] + img_size[1] / 2
        w[2] = w[2] + img_size[2] / 2
        warp_coords = w[0:3].reshape(3, img_size[0], img_size[1], img_size[2])

        if len(image.shape) == 3:
            image = warp(image, warp_coords)
        else:
            image_1st_c = warp(image[0], warp_coords)
            image_2nd_c = warp(image[1], warp_coords)
            image = np.stack([image_1st_c, image_2nd_c])

        new_label = np.zeros(label.shape, dtype=np.float32)
        for z in range(1,self.num_class):
            temp = warp((label == z).astype(np.float32),warp_coords)
            new_label[temp >= 0.5] = z
        label = new_label   
        
        sample['image'] = image
        sample['label'] = label
        return sample


class RandomFlip3D(object):
    '''
    Data augmentation method.
    Flipping the image, including horizontal and vertical flipping.
    Args:
    - mode: string, consisting of 'h' and 'v'. Optional methods and 'hv' is default.
            'h'-> horizontal flipping,
            'v'-> vertical flipping,
            'hv'-> random flipping.

    '''
    def __init__(self, mode='hv'):
        self.mode = mode

    def __call__(self, sample):
        # image: numpy array, (D,H,W)
        # label: integer, 0,1,..
        image = sample['image']
        label = sample['label']
        if len(image.shape) == 3:
            if 'h' in self.mode and 'v' in self.mode:
                random_factor = np.random.uniform(0, 1)
                if random_factor < 0.3:
                    image = image[:, ::-1, ...]
                    label = label[:, ::-1, ...]
                elif random_factor < 0.6:
                    image = image[..., ::-1]
                    label = label[..., ::-1]

            elif 'h' in self.mode:
                if np.random.uniform(0, 1) > 0.5:
                    image = image[:, ::-1, ...]
                    label = label[:, ::-1, ...]

            elif 'v' in self.mode:
                if np.random.uniform(0, 1) > 0.5:
                    image = image[..., ::-1]
                    label = label[..., ::-1]
        else:
            if 'h' in self.mode and 'v' in self.mode:
                random_factor = np.random.uniform(0, 1)
                if random_factor < 0.3:
                    image_1st_c =image[0]
                    image_1st_c = image_1st_c[:, ::-1, ...]
                    image_2nd_c = image[1]
                    image_2nd_c = image_2nd_c[:, ::-1, ...]
                    image = np.stack([image_1st_c, image_2nd_c])
                    label = label[:, ::-1, ...]
                elif random_factor < 0.6:
                    image_1st_c =image[0]
                    image_1st_c = image_1st_c[..., ::-1]
                    image_2nd_c = image[1]
                    image_2nd_c = image_2nd_c[..., ::-1]
                    image = np.stack([image_1st_c, image_2nd_c])
                    label = label[..., ::-1]

            elif 'h' in self.mode:
                if np.random.uniform(0, 1) > 0.5:
                    image_1st_c =image[0]
                    image_1st_c = image_1st_c[:, ::-1, ...]
                    image_2nd_c = image[1]
                    image_2nd_c = image_2nd_c[:, ::-1, ...]
                    image = np.stack([image_1st_c, image_2nd_c])
                    label = label[:, ::-1, ...]

            elif 'v' in self.mode:
                if np.random.uniform(0, 1) > 0.5:
                    image_1st_c =image[0]
                    image_1st_c = image_1st_c[..., ::-1]
                    image_2nd_c = image[1]
                    image_2nd_c = image_2nd_c[..., ::-1]
                    image = np.stack([image_1st_c, image_2nd_c])
                    label = label[..., ::-1]
        # avoid the discontinuity of array memory
        image = image.copy()
        label = label.copy()
        sample['image'] = image
        sample['label'] = label
        return sample
