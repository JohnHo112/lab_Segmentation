import numpy as np
import Tool



# filter setting
sobel_x1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/4
sobel_x2 = np.array([[-1, -1, 0, 1, 1], [-2, -2, 0, 2, 2], [-1, -1, 0, 1, 1]])/8
sobel_x3 = np.array([[-1, -1, -1, 0, 1, 1, 1], [-2, -2, -2, 0, 2, 2, 2], [-1, -1, -1, 0, 1, 1, 1]])/12
laplace2 = np.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 24, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])/24
laplace1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/8
exp_x1 = Tool.filter(0.8, 10)
exp_x2 = Tool.filter(0.6, 10)
exp_x3 = Tool.filter(0.4, 10)
exp_x4 = Tool.filter(0.2, 10)

# weight setting
L1 = {
    "l": 0.8,
    "l1": 0.25,
    "l2": 0.25,
    "l3": 0.25,
    "lt": 0.25
}
L2 = {
    "l": 0.8,
    "l1": 0.1,
    "l2": 0.3,
    "l3": 0.3,
    "lt": 0.3
}
L3 = {
    "l": 0.8,
    "l1": 0.1,
    "l2": 0.5,
    "l3": 0.3,
    "lt": 0.1
}
L4 = {
    "l": 0.8,
    "l1": 0.3,
    "l2": 0.5,
    "l3": 0.2,
    "lt": 0
}


Filters = {
    1: {"sobelx": sobel_x3, "sobely": np.transpose(sobel_x3), "laplace": laplace2},
    2: {"sobelx": sobel_x2, "sobely": np.transpose(sobel_x2), "laplace": laplace2},
    3: {"sobelx": sobel_x1, "sobely": np.transpose(sobel_x1), "laplace": laplace1},
    4: {"sobelx": sobel_x1, "sobely": np.transpose(sobel_x1), "laplace": laplace1},
    }
Thresholds = {
    1: {"delta": 17, "distance": 25},
    2: {"delta": 50, "distance": 50},
    3: {"delta": 100, "distance": 80},
    4: {"delta": 300, "distance": 60},
    }
L = {1: L1, 2: L2, 3: L3, 4: L4}



