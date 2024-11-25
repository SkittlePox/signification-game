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


def make_speaker_example_graphic(directory, count=5, log_interval=5, height_dx=30, method="uniform", **kwargs):
    image_dir = os.path.join(directory, "media/images/env/")
    files = os.listdir(image_dir)

    output_dir = os.path.join(directory, "data_vis/")

    fname_template = "speaker_examples_"

    sorted_files = sorted([f for f in files if fname_template in f],
                         key=lambda x: int(x.split(fname_template)[1].split('_')[0]))
    
    
    if method == "uniform":
        start_epoch = kwargs["start_epoch"]
        interval_epoch = kwargs["interval_epoch"]

        start_index=int((start_epoch-(log_interval-1))/log_interval)
        mid_index=int(start_index + count * interval_epoch / log_interval)
        interval_index=int(interval_epoch/log_interval)

        indices = range(start_index, mid_index, interval_index) if count > 0 else []
        image_files = [sorted_files[i] for i in indices if i < len(sorted_files)]
        
        graphic_name = f"{fname_template}{start_epoch}s_{interval_epoch}i_{count}c_{str(uuid.uuid4())[:6]}"

    elif method == "1/x":
        start_epoch = kwargs["start_epoch"]
        epoch_span = kwargs["epoch_span"]
        x_stretch = kwargs["x_stretch"]

        index_span = int(epoch_span / log_interval)
        start_index = int((start_epoch-(log_interval-1))/log_interval)
        # print(start_index)

        indices_of_interest = range(len(sorted_files))[start_index: start_index+index_span+1]

        index_values = np.array([(1.0/(x+x_stretch)) for x in indices_of_interest])

        # print(index_values)
        # return
        
        tot_sum = sum(index_values)
        desired_interval = tot_sum / count

        running_sum = 0.0
        indices = [indices_of_interest[0]]
        for i, v in enumerate(index_values):
            running_sum += v
            if running_sum >= desired_interval:
                indices.append(indices_of_interest[i])
                running_sum = 0.0

        # print(indices)
        image_files = [sorted_files[i] for i in indices if i < len(sorted_files)]
        graphic_name = f"{fname_template}{start_epoch}s_{epoch_span}s_{x_stretch}x_{count}c_{str(uuid.uuid4())[:6]}"

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
    
    
    combined_image.save(output_dir + f"{graphic_name}.png")

    # Save image filenames
    with open(output_dir + f"{graphic_name}_image_list.txt", "w") as f:
        for img_file in image_files:
            f.write(f"{img_file}\n")


if __name__=="__main__":
    # download_speaker_examples(run_id="signification-team/signification-game/avnly640", directory="./drawn-shape-1950/")
    make_speaker_example_graphic(directory="./drawn-shape-1950/", start_epoch=299, count=10, interval_epoch=125)
    # make_speaker_example_graphic(directory="./drawn-shape-1950/", start_epoch=299, count=10, epoch_span=1300, x_stretch=100.0, method="1/x")
