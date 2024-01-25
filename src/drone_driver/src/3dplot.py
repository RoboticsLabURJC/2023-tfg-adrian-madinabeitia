import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from PIL import Image
from imblearn.over_sampling import RandomOverSampler

def updateNumAngular(angularVel, value):
    label = 0
    if value > 0.50:
        angularVel[2] += 1
        label = 2
    elif value > 0.25:
        angularVel[1] += 1
        label = 1
    else:
        angularVel[0] += 1
        label = 0
    return angularVel, label

def read_image_folder_sorted_by_timestamp(folder_path):
    images = []

    try:
        # Listar archivos en la carpeta
        files = os.listdir(folder_path)

        # Filtrar solo archivos con extensión .jpg
        jpg_files = [file for file in files if file.endswith('.jpg')]

        # Ordenar archivos por timestamp
        sorted_files = sorted(jpg_files, key=lambda x: int(os.path.splitext(x)[0]))

        # Leer archivos en orden
        for file_name in sorted_files:
            file_path = os.path.join(folder_path, file_name)

            # Leer la imagen usando Pillow (PIL)
            image = Image.open(file_path)
            image_array = np.array(image)
            images.append(image_array)

    except FileNotFoundError:
        print("Error: Carpeta no encontrada.")

    return images


def read_folder_sorted_by_timestamp(folder_path):
    angularPositiveVels = [0, 0, 0]
    angularNegativeVels = [0, 0, 0]
    labels = []

    try:
        # Listar archivos en la carpeta
        files = os.listdir(folder_path)

        # Filtrar solo archivos con extensión .txt
        txt_files = [file for file in files if file.endswith('.txt')]

        # Ordenar archivos por timestamp
        sorted_files = sorted(txt_files, key=lambda x: int(os.path.splitext(x)[0]))

        # Leer archivos en orden
        for file_name in sorted_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                content = file.read()
                numbers = [float(num) for num in content.split(',')]

                # Updates the count
                if numbers[1] > 0:
                    angularPositiveVels, lab = updateNumAngular(angularPositiveVels, numbers[1])
                    labels.append(lab)
                else:
                    angularNegativeVels, lab = updateNumAngular(angularNegativeVels, abs(numbers[1]))
                    labels.append(-lab)

    except FileNotFoundError:
        print("Error: Carpeta no encontrada.")
    
    return angularPositiveVels, angularNegativeVels, labels

def plot_3d_bars(x, y, z, colors, xlabel='X', ylabel='Y', zlabel='Z', title='3D Plot'):
    # Create the figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Adjust the heights of the bars to go from 0 to their respective z values
    bottoms = np.zeros_like(z)
    width = depth = 0.8

    # Create the 3D bars with different colors for each row
    for i, color in zip(range(3), colors):
        indices = y == i
        ax.bar3d(x[indices], y[indices], bottoms[indices], dx=width, dy=depth, dz=z[indices], color=color)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Show the plot
    plt.show()


def oversample_images(images, labels):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(images.reshape(-1, 1), labels)

    # Se devuelven las rutas de las imágenes y las etiquetas resampleadas
    return X_resampled.flatten(), y_resampled


def main():
    # Ejemplo de uso
    labels_path = '../dataset/simple_circuit/labels'
    images_path = '../dataset/simple_circuit/frontal_images'

    angVelPos, angVelNeg, labels = read_folder_sorted_by_timestamp(labels_path)
    images = read_image_folder_sorted_by_timestamp(images_path)

    # Data for the columns with linear
    x = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    z = np.array([0, 0, 0, angVelNeg[0], angVelNeg[1], angVelNeg[2], angVelPos[0], angVelPos[1], angVelPos[2]])

    # Colors for each row
    colors = ['green', 'lightblue', 'blue']

    # Llamar a la función de graficado
    plot_3d_bars(x, y, z, colors, xlabel='Min mel    Medium vel   Max vel ', ylabel='Linear, -Ang, +Ang', zlabel='Samples', title='Simple circuit')

    # resampled_images, resampled_labels = oversample_images(np.array(images), np.array(labels))

if __name__ == "__main__":
    main()
