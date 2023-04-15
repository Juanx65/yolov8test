import random

import numpy as np

random.seed(0)

# detection model classes
CLASSES = ('pellet')

# colors for per classes
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}

# colors for segment masks
MASK_COLORS = np.array([(255, 56, 56), (255, 157, 151), (255, 112, 31),
                        (255, 178, 29), (207, 210, 49), (72, 249, 10),
                        (146, 204, 23), (61, 219, 134), (26, 147, 52),
                        (0, 212, 187), (44, 153, 168), (0, 194, 255),
                        (52, 69, 147), (100, 115, 255), (0, 24, 236),
                        (132, 56, 255), (82, 0, 133), (203, 56, 255),
                        (255, 149, 200), (255, 55, 199)],
                       dtype=np.float32) / 255.

# alpha for segment masks
ALPHA = 0.5
