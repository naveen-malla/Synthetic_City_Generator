import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_coordinate_comparison(original_coords, generated_coords, initial_coords):
    # Convert tuple lists to numpy arrays
    original_coords = np.array(original_coords)
    generated_coords = np.array(generated_coords)
    initial_coords = np.array(initial_coords)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot original coordinates
    ax1.scatter(original_coords[:, 1], original_coords[:, 0], 
                c='#2ECC71', marker='o', s=100,
                label='Original Coordinates')
    ax1.scatter(initial_coords[:, 1], initial_coords[:, 0], 
                c='royalblue', marker='o', s=120, 
                label='Initial Coordinates')
    ax1.set_title('Original Coordinates', fontsize=14, pad=15)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.legend()
    
    # Plot generated coordinates
    ax2.scatter(generated_coords[:, 1], generated_coords[:, 0], 
                c='#E74C3C', marker='o', s=100, 
                label='Predicted Coordinates')
    ax2.scatter(initial_coords[:, 1], initial_coords[:, 0], 
                c='royalblue', marker='o', s=120, 
                label='Initial Coordinates')
    ax2.set_title('Generated Coordinates', fontsize=14, pad=15)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('X Coordinate', fontsize=12)
    ax2.set_ylabel('Y Coordinate', fontsize=12)
    ax2.legend()
    
    plt.suptitle('Coordinate Comparison', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()




# coordinates of test city 1


initial = [
(143, 107),
(96, 233),
(173, 49),
(75, 212)
]

original = [

(113, 255),
(8, 202),
(255, 108),
(39, 83),
(163, 123),
(106, 102),
(0, 0),
(28, 91),
(99, 195),
(29, 141),
(27, 189),
(103, 149),
(194, 201),
(187, 184),
(162, 94),
(206, 182)
]


generated = [
(193, 107),
(143, 164),
(201, 212),
(173, 144),
(173, 233),
(143, 233),
(96, 164),
(96, 212),
(75, 164),
(75, 212),
(49, 144),
(49, 212),
(49, 164)
]


# Plot the coordinates
plot_coordinate_comparison(original, generated, initial)

# cooridnats of test city 2

# initial = [
#     (222, 0),
#     (218, 59),
#     (188, 93),
#     (165, 222),
#     (253, 242),
#     (246, 230),
#     (228, 192),
#     (247, 73),
#     (249, 168)
# ]
# original = [
    
#     (229, 74),
#     (240, 177),
#     (255, 212),
#     (19, 6),
#     (187, 30),
#     (210, 148),
#     (185, 175),
#     (163, 184),
#     (98, 187),
#     (196, 220),
#     (114, 228),
#     (93, 222),
#     (94, 217),
#     (71, 209),
#     (51, 194),
#     (131, 114),
#     (136, 186),
#     (142, 225),
#     (53, 196),
#     (44, 255),
#     (201, 0),
#     (173, 116),
#     (202, 158),
#     (161, 141),
#     (173, 182),
#     (178, 222),
#     (188, 138),
#     (136, 69),
#     (160, 94),
#     (101, 132),
#     (144, 65),
#     (101, 122),
#     (102, 112),
#     (163, 121),
#     (205, 15),
#     (102, 103),
#     (129, 132),
#     (0, 255)
# ]

# generated = [

# (245, 204),
# (229, 165),
# (232, 152),
# (217, 136),
# (197, 112),
# (181, 87),
# (168, 65),
# (155, 48),
# (145, 32),
# (137, 20),
# (128, 11),
# (118, 3),
# (109, 21),
# (103, 40),
# (102, 58),
# (102, 77),
# (103, 97),
# (104, 117),
# (106, 137),
# (108, 156),
# (110, 174)
# ]


# # Plot the coordinates
# plot_coordinate_comparison(original, generated, initial)

