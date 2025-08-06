from image.pkg import Backgroundremover, load_img
from image.strain.pkg import Denosie_Position, Calculate_neighbor, Vector_centers, standard_vector, classify_vectors
from image.strain.pkg import AtomSelector, calculate_strain, plot_strain

# Load the original STEM image
image = load_img()

# Step 1: Remove the background from the image
# max_scale and min_scale define the scale range for background removal
remover = Backgroundremover(image, max_scale=2048, min_scale=32)
image = remover.remove()
remover.visualize()              # Visualize the image after background removal

# Step 2: Denoise the image and detect atomic positions
# periodicity: number of boxes per row/column (the bigger one) for denoising and segmentation
# stride: step size of sliding window
# batch_size: batch size for the denoising model
# model: model type for denoising ('unet' or 'unet3')
# split_method: method to split denoised regions ('watershed' or 'floodfill')
# min_distance: minimum distance between detected atoms
# filter_ratio: ratio threshold to filter weak detections (usually 0.5-1)
# I: intensity metric to consider ('i_peak' or 'i_mean')
denoised_image, filtered_image, cropped_image, filtered_cropped_image, atoms, atoms_crop = Denosie_Position(
            image,
            periodicity=3,
            stride=6/8,
            batch_size=8,
            model='unet',
            split_method='floodfill',
            min_distance=3,
            filter_ratio=0.8,
            I='i_peak',
            save=False
        )

# Step 3: Calculate neighboring atoms for each detected atom within radius r
# neighbor_dis: distances to neighbors
# neighbor_pos: neighbor positions
# cutoff: distance cutoff threshold
r = 200            # Neighbor search radius in pixels (or unit matching image scale)
neighbor_dis, neighbor_pos, cutoff = Calculate_neighbor(atoms.to_numpy(), r, denoised_image)
neighbor_dis_crop, neighbor_pos_crop, _ = Calculate_neighbor(atoms_crop.to_numpy(), r, cropped_image, if_crop=True)

# Step 4: Classify standard lattice vectors and label atoms in the cropped region
# vector_centers: centers of standard lattice vectors
# atoms_labels_cropped: labels assigned to atoms in cropped region based on vector classification
vector_centers, atoms_labels_cropped = Vector_centers(
        atoms_crop.to_numpy(), neighbor_dis_crop, neighbor_pos_crop, cutoff
        )

# Step 5: Extract lattice vectors for all atoms in the original image and classify them
# vectors_list: list of lattice vectors for each atom
# vector_labels_list: labels for each lattice vector
# atoms_labels: atom labels based on lattice vector classification
vectors_list, vector_labels_list, atoms_labels = classify_vectors(
    vector_centers,  atoms.to_numpy(),  neighbor_dis,
    neighbor_pos, cutoff, dist_multi=1)

# Step 6: Interactive atom selector GUI for the user to select specific lattice vector classes
# selected_vectors_list: lattice vectors selected by user
# selected_vector_labels_list: labels of selected vectors
# selected_atoms: atom positions corresponding to the selected vectors
atomselector = AtomSelector(vectors_list, vector_labels_list, atoms.to_numpy(), atoms_labels, denoised_image)
selected_vectors_list, selected_vector_labels_list, selected_atoms = atomselector.get_result()

# Step 7: Define a set of standard lattice vectors for strain calculation
standard_a, standard_b, label_a, label_b = standard_vector(cropped_image, vector_centers)

# Step 8: Calculate strain tensor components for selected atoms using the standard lattice vectors
# strain_i = [eyx, eyy, exx, exy, emax, emin, theta]
strain = calculate_strain(selected_vectors_list, selected_vector_labels_list, standard_a, standard_b, label_a, label_b)

# Step 9: Visualize strain maps and principal strain directions on the denoised image
viewer = plot_strain(denoised_image, selected_atoms, strain, standard_a, standard_b)