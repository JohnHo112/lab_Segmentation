import numpy as np

# filter setting
sobel_x1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/4
sobel_x2 = np.array([[-1, -1, 0, 1, 1], [-2, -2, 0, 2, 2], [-1, -1, 0, 1, 1]])/8
sobel_x3 = np.array([[-1, -1, -1, 0, 1, 1, 1], [-2, -2, -2, 0, 2, 2, 2], [-1, -1, -1, 0, 1, 1, 1]])/12
laplace2 = np.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 24, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])/24
laplace1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/8

# weight setting
L1_l = {
    "ld": 0, 
    "l1": 1,
    "l2": 0,
    "l3": 0,
    "lt": 0
}
L2_l = {
    "ld": 0.1, 
    "l1": 0.05,
    "l2": 0.03,
    "l3": 0,
    "lt": 0
}
L3_l = {
    "ld": 0.03,
    "l1": 0.02,
    "l2": 0.02,
    "l3": 10,
    "lt": 10
}

Filters = {
    1: {"sobelx": sobel_x3, "sobely": np.transpose(sobel_x3), "laplace": laplace2},
    2: {"sobelx": sobel_x2, "sobely": np.transpose(sobel_x2), "laplace": laplace2},
    3: {"sobelx": sobel_x2, "sobely": np.transpose(sobel_x2), "laplace": laplace2},
    }
Thresholds = {
    1: {"delta": 0, "distance": 50, "area": 1500, "tx": 15, "ty": 15},
    2: {"delta": 0, "distance": 80, "area": 3000, "tx": 15, "ty": 15},
    3: {"delta": 0, "distance": 100, "area": 3000, "tx": 15, "ty": 15},
    }

L_multi = {
    1: L1_l, 
    2: L2_l, 
    3: L3_l,
}
