import os
from PIL import Image


# Path to the new dataset folder
new_dataset_folder = r'GVSS_Vision_Data/Dataset_Final_grs'
output_folder = 'Canvas_dataset'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the views and their positions in the grid
views = ['bathroom', 'bedroom', 'frontal', 'kitchen']
positions = {
    'bathroom': (0, 0),
    'bedroom': (1, 0),
    'frontal': (0, 1),
    'kitchen': (1, 1)
}

# List all the images in the new dataset folder
images = [img for img in os.listdir(new_dataset_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Create a dictionary to store images for each house
house_images = {}
for image in images:
    house_number = int(image.split('_')[0])
    if house_number not in house_images:
        house_images[house_number] = {}
    for view in views:
        if view in image:
            house_images[house_number][view] = os.path.join(new_dataset_folder, image)

# Function to create a blank image
def create_blank_image(size, color=(0,0,0)):
    return Image.new('RGB', size, color)

# Function to resize image without maintaining aspect ratio
def resize_image(image, size):
    return image.resize(size, Image.LANCZOS)

# Process each house
for house_number, images in house_images.items():
    # Create a blank canvas for the 2x2 grid
    cell_size = (256, 256)
    canvas_size = (cell_size[0] * 2, cell_size[1] * 2)
    canvas = create_blank_image(canvas_size)

    for view, pos in positions.items():
        if view in images:
            img = Image.open(images[view])
            img = resize_image(img, cell_size)
        else:
            img = create_blank_image(cell_size)
        
        # Paste the image on the canvas at the correct position
        canvas.paste(img, (pos[0] * cell_size[0], pos[1] * cell_size[1]))

    # Save the canvas image
    canvas.save(os.path.join(output_folder, f'{house_number}.jpg'))

print("Canvas images created for each house.")
