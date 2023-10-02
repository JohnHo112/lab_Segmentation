import numpy as np
import Tool



# filter setting
sobel_x1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/4
sobel_x11 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])/3
sobel_x2 = np.array([[-1, -1, 0, 1, 1], [-2, -2, 0, 2, 2], [-1, -1, 0, 1, 1]])/8
sobel_x3 = np.array([[-1, -1, -1, 0, 1, 1, 1], [-2, -2, -2, 0, 2, 2, 2], [-1, -1, -1, 0, 1, 1, 1]])/12
laplace2 = np.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 24, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])/24
laplace1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/8
laplace21 = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, -24, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])/24
laplace11 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])/8
exp_x1 = Tool.filter(0.8, 10)
exp_x2 = Tool.filter(0.6, 10)
exp_x3 = Tool.filter(0.4, 10)
exp_x4 = Tool.filter(0.2, 10)

# weight setting
L1 = {
    #"l": 0.8,
    "l1": 0.2,
    "l2": 0.2,
    "l3": 0.1,
    "lt": 0.1
}
L2 = {
    #"l": 0.8,
    "l1": 1,
    "l2": 0.4,
    "l3": 0.2,
    "lt": 0.2
}
L3 = {
    #"l": 0.8,
    "l1": 0.5,
    "l2": 0.2,
    "l3": 0.2,
    "lt": 0.2
}
L4 = {
    #"l": 0.8,
    "l1": 0.2,
    "l2": 0.4,
    "l3": 0.3,
    "lt": 0.3
}
L5 = {
    #"l": 0.8,
    "l1": 0.1,
    "l2": 0.4,
    "l3": 0.2,
    "lt": 0.2
}
L6 = {
    #"l": 0.8,
    "l1": 0.1,
    "l2": 0.3,
    "l3": 0.1,
    "lt": 0.1
}
L7 = {
    #"l": 0.8,
    "l1": 0.25,
    "l2": 0.25,
    "l3": 0.25,
    "lt": 0.25
}


Filters = {
    1: {"sobelx": sobel_x3, "sobely": np.transpose(sobel_x3), "laplace": laplace2},
    2: {"sobelx": sobel_x2, "sobely": np.transpose(sobel_x2), "laplace": laplace2},
    3: {"sobelx": sobel_x2, "sobely": np.transpose(sobel_x2), "laplace": laplace2},
    4: {"sobelx": sobel_x1, "sobely": np.transpose(sobel_x1), "laplace": laplace1},
    5: {"sobelx": sobel_x1, "sobely": np.transpose(sobel_x1), "laplace": laplace1},
    6: {"sobelx": sobel_x1, "sobely": np.transpose(sobel_x1), "laplace": laplace1},
    #7: {"sobelx": sobel_x1, "sobely": np.transpose(sobel_x1), "laplace": laplace1},
    }
Thresholds = {
    1: {"delta": 9, "distance": 20, "area": 40, "tx": 8, "ty": 7.5},
    2: {"delta": 16, "distance": 20, "area": 200, "tx": 6.5, "ty": 5.5},
    3: {"delta": 30, "distance": 20, "area": 300, "tx": 6, "ty": 4.5},
    4: {"delta": 50, "distance": 25, "area": 800, "tx": 5, "ty": 4},
    5: {"delta": 75, "distance": 20, "area": 800, "tx": 5, "ty": 4},
    6: {"delta": 100, "distance": 20, "area": 1000, "tx": 5, "ty": 4},
    #7: {"delta": 200, "distance": 20, "area": 1000, "tx": 5, "ty": 4},
    }
L = {
    1: L1, 
    2: L2, 
    3: L3, 
    4: L4,
    5: L5,
    6: L6,
    #7: L7,
    }



