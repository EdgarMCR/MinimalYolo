import pickle
from pathlib import Path

from PIL import Image

import numpy as np


def draw_box(img: np.ndarray, box_width_min_max: tuple, box_color: tuple = (255, 255, 255)):
    """ Random location and size square. """
    height, width = img.shape[:2]
    box_width = np.random.randint(*box_width_min_max)
    start_x, start_y = np.random.randint(0, height - box_width, 2)
    img[start_y:start_y + box_width, start_x:start_x + box_width] = box_color
    return img, start_x, start_y, box_width


def draw_circle(img: np.ndarray, circle_radius_min_max: tuple, circle_color: tuple = (255, 255, 255)):
    """ Random location and size circle. """
    height, width = img.shape[:2]
    circle_radius = np.random.randint(*circle_radius_min_max)
    center_x, center_y = np.random.randint(circle_radius, height - circle_radius, 2)

    for ii in range(center_x - circle_radius, center_x + circle_radius + 1):
        for jj in range(center_y - circle_radius, center_y + circle_radius + 1):
            distance = np.sqrt((ii - center_x) ** 2 + (jj - center_y) ** 2)
            if distance < circle_radius:
                img[jj, ii] = circle_color

    return img, center_x, center_y, circle_radius


def add_objects(img: np.ndarray, max_number: int, size_range: tuple):
    # bounding box:  xmin, ymin, xmax, ymax.
    classes, bbox = [], []
    for _ in range(np.random.randint(1, max_number + 1)):
        colour = np.random.randint(0, 254, 3)
        if np.random.rand() > 0.5:
            img, x, y, w = draw_box(img, size_range, colour)
            classes.append(0)
            bbox.append([x, y, x + w, y + w])

        else:
            img, x, y, w = draw_circle(img, size_range, colour)
            classes.append(1)
            bbox.append([x - w, y - w, x + w, y + w])
    return img, classes, bbox


def generate_synthetic_data(n: int, height: int, width: int, max_number: int = 3, size_range: tuple = (10, 40)):
    """
    classes: 0=square, 1=circle
    bounding box:  xmin, ymin, xmax, ymax.
    """
    for _ in range(n):
        background_colour = np.random.randint(0, 255, 3)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = background_colour
        img, classes, bbox = add_objects(img, max_number, size_range)
        yield img, classes, bbox


def generate_synthetic_data_and_save(out_folder: Path, n: int, height: int, width: int, max_number: int = 3,
                                     size_range: tuple = (10, 40)):
    """ Save the synthetic data to disk (as this simulates usual datasets). """
    results = []
    for idx, (img, classes, bbox) in enumerate(generate_synthetic_data(n, height, width, max_number, size_range)):
        img_path = out_folder / f"synthetic_data_{idx}.jpg"
        im = Image.fromarray(img)
        im.save(img_path)

        results.append((img_path, classes, bbox))

    with open(out_folder / 'synthetic_data.pickle', 'wb') as f:
        pickle.dump(results, f)
    return results


if __name__ == '__main__':
    out_folder = Path(r'/mnt/data/kirintec/synth_data')
    generate_synthetic_data_and_save(out_folder, 10000, 224, 224)
