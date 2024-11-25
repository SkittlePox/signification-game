import wandb
import os
import numpy as np
import uuid
from PIL import Image

def download_speaker_examples(run_id, directory):
    api = wandb.Api()
    run = api.run(run_id)
    for file in run.files():
        if "speaker_examples" in str(file):
            file.download(root=directory)


def make_speaker_example_graphic(directory, start_epoch=299, interval_epoch=20, count=30, log_interval=5, height_dx=30): # TODO: This function will take speaker images and turn them into nice graphics
    image_dir = os.path.join(directory, "media/images/env/")
    files = os.listdir(image_dir)

    output_dir = os.path.join(directory, "data_vis/")

    fname_template = "speaker_examples_"

    sorted_files = sorted([f for f in files if fname_template in f],
                         key=lambda x: int(x.split(fname_template)[1].split('_')[0]))
    
    
    start_index=int((start_epoch-(log_interval-1))/log_interval)
    mid_index=int(start_index + count * interval_epoch / log_interval)
    interval_index=int(interval_epoch/log_interval)

    indices = range(start_index, mid_index, interval_index) if count > 0 else []
    image_files = [sorted_files[i] for i in indices if i < len(sorted_files)]

    # Read and concatenate images
    images = []
    for i, f in enumerate(image_files):
        img = Image.open(os.path.join(image_dir, f))
        img_array = np.array(img)
        if i == len(image_files) - 1:
            images.append(img_array[:height_dx+2])
        else:
            images.append(img_array[:height_dx])
        print(f)

    combined = np.concatenate(images, axis=0)
    combined_image = Image.fromarray(combined)
    
    graphic_name = f"{fname_template}{start_epoch}s_{interval_epoch}i_{count}c_{str(uuid.uuid4())[:6]}"
    combined_image.save(output_dir + f"{graphic_name}.png")

    # Save image filenames
    with open(output_dir + f"{graphic_name}_image_list.txt", "w") as f:
        f.write(f"start_epoch: {start_epoch}\n")
        f.write(f"interval_epoch: {interval_epoch}\n") 
        f.write(f"count: {count}\n")
        for img_file in image_files:
            f.write(f"{img_file}\n")


if __name__=="__main__":
    # download_speaker_examples(run_id="signification-team/signification-game/avnly640", directory="./drawn-shape-1950/")
    make_speaker_example_graphic(directory="./drawn-shape-1950/", start_epoch=299, interval_epoch=100, count=10)
