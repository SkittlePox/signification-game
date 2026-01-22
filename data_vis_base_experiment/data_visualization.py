import wandb
import os
import numpy as np
import uuid
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import matplotlib.animation as animation
import matplotlib
import matplotlib as mpl
import seaborn as sns
import json

from utils import get_sweep_dirs

# os.chdir(os.path.join(os.getcwd(), "data_vis_base_experiment/runs"))  # For debug!
os.chdir(os.path.join(os.getcwd(), "runs"))


def download_speaker_examples(run_id, directory, tom_examples_only=True):
    fname_fragment = "tom_speaker_examples" if tom_examples_only else "speaker_examples"
    os.makedirs(directory, exist_ok=True)
    api = wandb.Api()
    run = api.run(run_id)
    files = run.files()
    print("Downloading for run", directory)
    for file in tqdm(files, desc="Downloading speaker examples"):
        if fname_fragment in str(file):
            file.download(root=directory, exist_ok=True)

def download_probe_data(run_id, directory, which_speakers=[0]):
    os.makedirs(directory, exist_ok=True)
    api = wandb.Api()
    run = api.run(run_id)
    history = run.scan_history()
    print("Downloading for run", directory)
    for sp_num in which_speakers:
        probe_entropy = [row[f"probe/entropy/speaker {sp_num} average"] for row in tqdm(history, desc="Downloading probe data")]
        probe_entropy_df = pd.DataFrame(probe_entropy)
        probe_entropy_df.to_csv(os.path.join(directory, f"probe_entropy_speaker_{sp_num}.csv"), index=False)
    probe_entropy = [row[f"probe/entropy/all speakers average"] for row in tqdm(history, desc="Downloading probe data")]
    probe_entropy_df = pd.DataFrame(probe_entropy)
    probe_entropy_df.to_csv(os.path.join(directory, f"probe_entropy_all_speakers.csv"), index=False)

def download_pr_data(run_id, directory, referents=list(range(10)), listeners=()):
    os.makedirs(directory, exist_ok=True)
    api = wandb.Api()
    run = api.run(run_id)
    history = run.scan_history()
    print("Downloading for run", directory)
    for ref_num in referents:
        probe_entropy = [row[f"inference/all listeners referent {ref_num}"] for row in tqdm(history, desc="Downloading probe data")]
        probe_entropy_df = pd.DataFrame(probe_entropy)
        probe_entropy_df.to_csv(os.path.join(directory, f"inference_pr_referent_{ref_num}.csv"), index=False)
        for lis_num in listeners:
            probe_entropy = [row[f"inference/listener {lis_num} referent {ref_num}"] for row in tqdm(history, desc="Downloading probe data")]
            probe_entropy_df = pd.DataFrame(probe_entropy)
            probe_entropy_df.to_csv(os.path.join(directory, f"inference_pr_listener_{lis_num}_referent_{ref_num}.csv"), index=False)

def download_reward_data(run_id, directory):
    os.makedirs(directory, exist_ok=True)
    api = wandb.Api()
    run = api.run(run_id)
    print("Downloading for run", directory)
    history = run.scan_history()
    # for sp_num in which_speakers:
    #     probe_entropy = [row[f"probe/entropy/speaker {sp_num} average"] for row in tqdm(history, desc="Downloading probe data")]
    #     probe_entropy_df = pd.DataFrame(probe_entropy)
    #     probe_entropy_df.to_csv(os.path.join(directory, f"probe_entropy_speaker_{sp_num}.csv"), index=False)
    rewards = [row[f"reward/mean reward by image source/speaker images all listeners"] for row in tqdm(history, desc="Downloading reward data")]
    rewards_df = pd.DataFrame(rewards)
    rewards_df.to_csv(os.path.join(directory, f"reward_for_speaker_images_all_listeners.csv"), index=False)


def download_spline_data(run_id, directory):
    os.makedirs(directory, exist_ok=True)
    api = wandb.Api()
    run = api.run(run_id)
    print("Downloading for run", directory)
    history = run.scan_history()

    # Define metrics to download
    metrics = [
        ("spline wasserstein distances/all speakers mean", "spline_wasserstein_distances_mean.csv", "spline wasserstein means"),
        ("spline wasserstein distances/all speakers std dev", "spline_wasserstein_distances_stddevs.csv", "spline wasserstein std devs"),
        ("spline wasserstein distances/all speakers cv", "spline_wasserstein_distances_cv.csv", "spline wasserstein cvs"),
        ("spline wasserstein distances invariant/all speakers mean", "spline_wasserstein_distances_mean_invariant.csv", "invariant spline wasserstein means"),
        ("spline wasserstein distances invariant/all speakers std dev", "spline_wasserstein_distances_stddevs_invariant.csv", "invariant spline wasserstein std devs"),
        ("spline wasserstein distances invariant/all speakers cv", "spline_wasserstein_distances_cv_invariant.csv", "invariant spline wasserstein cvs"),
        # ("spline visual distances/all speakers multsum mean", "spline_visual_multsum_distances_mean.csv", "spline multsum means"),
        # ("spline visual distances/all speakers multsum std dev", "spline_visual_multsum_distances_stddevs.csv", "spline multsum std devs"),
        ("spline w2 distances variance weighted invariant/all speakers mean", "spline_w2_distances_invariant.csv", "spline w2 distances invariant"),
        ("spline w2 distances variance weighted/all speakers mean", "spline_w2_distances.csv", "spline w2 distances"),
        ("spline w2 distances variance weighted invariant/all speakers std dev", "spline_w2_distances_invariant_std_dev.csv", "spline w2 distances invariant std dev"),
        ("spline w2 distances variance weighted/all speakers std dev", "spline_w2_distances_std_dev.csv", "spline w2 distances std dev"),
    ]

    for metric_key, filename, description in metrics:
        try:
            data = [row[metric_key] for row in tqdm(history, desc=f"Downloading {description}")]
            data_df = pd.DataFrame(data)
            data_df.to_csv(os.path.join(directory, filename), index=False)
            print(f"✓ Successfully saved {filename}")
        except KeyError as e:
            print(f"Warning: Could not find key '{metric_key}': {e}")

def download_communication_success_data(run_id, directory, referents=list(range(10))):
    os.makedirs(directory, exist_ok=True)
    api = wandb.Api()
    run = api.run(run_id)
    history = run.scan_history()
    print("Downloading for run", directory)
    for ref_num in referents:
        success_data = [row[f"success/average success/all speakers referent {ref_num}"] for row in tqdm(history, desc=f"Downloading communication success data for referent {ref_num}")]
        success_data_df = pd.DataFrame(success_data)
        success_data_df.to_csv(os.path.join(directory, f"success_rate_referent_{ref_num}.csv"), index=False)
    success_data = [row[f"success/average success/all speakers"] for row in tqdm(history, desc="Downloading average communication success data")]
    success_data_df = pd.DataFrame(success_data)
    success_data_df.to_csv(os.path.join(directory, f"success_rate_all_referents.csv"), index=False)


def make_speaker_example_graphic(directory, count=5, log_interval=5, image_dim=28, method="uniform", fname_prefix="tom_", speaker_selection=None, referent_selection=None, one_sign=None, vertical=True, **kwargs):
    height_dx = image_dim + 2   # Assuming 2px border
    image_dir = os.path.join(directory, "media/images/env/")
    files = os.listdir(image_dir)

    output_dir = os.path.join(directory, "data_vis/")
    os.makedirs(output_dir, exist_ok=True)

    fname_template = fname_prefix+"speaker_examples_"

    sorted_files = sorted([f for f in files if f.startswith(fname_template)],
                         key=lambda x: int(x.split(fname_template)[1].split('_')[0]))
    
    # print(len(sorted_files))
    
    if method == "uniform":
        start_epoch = kwargs["start_epoch"]
        interval_epoch = kwargs["interval_epoch"]

        start_index=int((start_epoch-(log_interval-1))/log_interval)
        mid_index=int(start_index + count * interval_epoch / log_interval)
        interval_index=int(interval_epoch/log_interval)

        indices = range(start_index, mid_index, interval_index) if count > 0 else []
        image_files = [sorted_files[i] for i in indices if i < len(sorted_files)]
        
        graphic_name = f"{fname_template}{start_epoch}s_{interval_epoch}i_{count}c_{str(uuid.uuid4())[:5]}"

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
        graphic_name = f"{fname_template}{start_epoch}s_{epoch_span}s_{x_stretch}x_{count}c_{str(uuid.uuid4())[:5]}"

    # Read and concatenate images
    if speaker_selection == None and one_sign == None:   # TODO: eventually merge these two branches
        images = []
        for i, f in enumerate(image_files):
            img = Image.open(os.path.join(image_dir, f))
            img_array = np.array(img)
            local_height_dx = height_dx+2 if i == len(image_files) - 1 else height_dx
            images.append(img_array[:local_height_dx])
            print(f)
    elif one_sign != None:
        images = []
        for i, f in enumerate(image_files):
            img = Image.open(os.path.join(image_dir, f))
            img_array = np.array(img)
            # local_height_dx = height_dx+2
            local_width_dx = height_dx if i == len(image_files) - 1 else height_dx
            images.append(img_array[height_dx*one_sign[1]:height_dx*one_sign[1]+height_dx, height_dx*one_sign[0]:height_dx*one_sign[0]+local_width_dx])
            print(f)
        graphic_name = "single_"+graphic_name
    else:
        images = []
        for i, f in enumerate(image_files):
            img = Image.open(os.path.join(image_dir, f))
            img_array = np.array(img)
            local_height_dx = height_dx if i == len(image_files) - 1 else height_dx
            row_imgs = []
            for ii, j in zip(referent_selection, speaker_selection):
                local_width_dx = height_dx+2 if ii == referent_selection[-1] else height_dx
                row_imgs.append(img_array[height_dx*j:height_dx*j+local_height_dx, height_dx*ii:height_dx*ii+local_width_dx])
            row_img = np.concatenate(row_imgs, axis=1)            
            images.append(row_img)
            print(f)


    combined = np.concatenate(images, axis=0 if vertical else 1)
    combined_image = Image.fromarray(combined)
    combined_image.save(output_dir + f"{graphic_name}.png")

    # Save image filenames
    with open(output_dir + f"{graphic_name}_image_list.txt", "w") as f:
        for img_file in image_files:
            f.write(f"{img_file}\n")


def make_multi_speaker_example_graphic_single_sign(directories, one_sign, count=5, log_interval=5, image_dim=32, method="uniform", fname_prefix="tom_", vertical=True, **kwargs):
    height_dx = image_dim + 2   # Assuming 2px border
    fname_template = fname_prefix+"speaker_examples_"

    all_image_files = []
    image_dirs = []

    for directory in directories:
        image_dir = os.path.join(directory, "media/images/env/")
        files = os.listdir(image_dir)

        sorted_files = sorted([f for f in files if f.startswith(fname_template)],
                            key=lambda x: int(x.split(fname_template)[1].split('_')[0]))
        
        # print(len(sorted_files))
        
        if method == "uniform":
            start_epoch = kwargs["start_epoch"]
            interval_epoch = kwargs["interval_epoch"]

            start_index=int((start_epoch-(log_interval-1))/log_interval)
            mid_index=int(start_index + count * interval_epoch / log_interval)
            interval_index=int(interval_epoch/log_interval)

            indices = range(start_index, mid_index, interval_index) if count > 0 else []
            image_files = [sorted_files[i] for i in indices if i < len(sorted_files)]
            
            graphic_name = f"{fname_template}onesign_{str(one_sign)}_{start_epoch}s_{interval_epoch}i_{count}c_{str(uuid.uuid4())[:5]}"

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
            graphic_name = f"{fname_template}onesign_{str(one_sign)}_{start_epoch}s_{epoch_span}s_{x_stretch}x_{count}c_{str(uuid.uuid4())[:5]}"
        
        all_image_files.append(image_files)
        image_dirs.append(image_dir)


    # Assuming one sign
    images = []
    for run_ix, image_files in enumerate(all_image_files):
        col_imgs = []
        for i, f in enumerate(image_files):
            img = Image.open(os.path.join(image_dirs[run_ix], f))
            img_array = np.array(img)
            # local_height_dx = height_dx+2
            local_width_dx = height_dx if i == len(image_files) - 1 else height_dx
            col_imgs.append(img_array[height_dx*one_sign[1]:height_dx*one_sign[1]+height_dx, height_dx*one_sign[0]:height_dx*one_sign[0]+local_width_dx])
        # col_img = np.concatenate(col_)
        col_img = np.concatenate(col_imgs, axis=0)
        images.append(col_img)

    combined = np.concatenate(images, axis=1)

    combined_image = Image.fromarray(combined)
    combined_image.save("../joint-plots/" + f"{graphic_name}.png")

    # Save image filenames
    with open("../joint-plots/configs/" + f"{graphic_name}_image_list.txt", "w") as f:
        for i, image_files in enumerate(all_image_files):
            f.write(f"{image_dirs[i]}\n")
            for img_file in image_files:    
                f.write(f"{img_file}\n")

def make_graphics_part1():
    # (Runs 1950: manipulation, 1931: whitesum, 1934: negative whitesum, 1940: auto-centering, 1944: curvature, 1945: negative curvature)

    # Download data for Part1 all V1
    # download_speaker_examples(run_id="signification-team/signification-game/avnly640", directory="./drawn-shape-1950/")
    # download_speaker_examples(run_id="signification-team/signification-game/ni0g5mz6", directory="./distinctive-mountain-1931/")
    # download_speaker_examples(run_id="signification-team/signification-game/dlezjmad", directory="./true-forest-1934/")
    # download_speaker_examples(run_id="signification-team/signification-game/51sgma7e", directory="./jolly-donkey-1940/")
    # download_speaker_examples(run_id="signification-team/signification-game/d0gvcotl", directory="./playful-oath-1944/")
    # download_speaker_examples(run_id="signification-team/signification-game/2vy8mbfi", directory="./treasured-sound-1945/")
    
    # Download data for Part1b V2 (squeezed and unsqueezed manip-coop at epoch 1k)
    # download_speaker_examples(run_id="signification-team/signification-game/4qlamhgk", directory="./warm-night-1962/")
    # download_speaker_examples(run_id="signification-team/signification-game/pgyqk1o8", directory="./azure-leaf-1963/")
    # download_speaker_examples(run_id="signification-team/signification-game/0i5kqgie", directory="./iconic-resonance-1964/")
    # download_speaker_examples(run_id="signification-team/signification-game/6tune16z", directory="./vivid-water-1965/")
    # download_speaker_examples(run_id="signification-team/signification-game/xwens41t", directory="./cool-cosmos-1966/")
    # download_speaker_examples(run_id="signification-team/signification-game/kphyfba3", directory="./devout-shadow-1967/")
    # download_speaker_examples(run_id="signification-team/signification-game/fn48wc5z", directory="./ethereal-dust-1968/")
    # download_speaker_examples(run_id="signification-team/signification-game/14tntarn", directory="./absurd-feather-1969/")

    # Download data for Part1b V3 (squeezed manip-coop at epoch 600)
    # download_speaker_examples(run_id="signification-team/signification-game/0u0y0lk4", directory="./soft-sun-1970/")
    # download_speaker_examples(run_id="signification-team/signification-game/m1t9k3wp", directory="./atomic-firefly-1971/")
    # download_speaker_examples(run_id="signification-team/signification-game/yw1uvwal", directory="./divine-deluge-1972/")
    # download_speaker_examples(run_id="signification-team/signification-game/wjtjyd8u", directory="./whole-firefly-1973/")
    download_speaker_examples(run_id="signification-team/signification-game/0in3o71n", directory="./cool-armadillo-1975/")
    download_speaker_examples(run_id="signification-team/signification-game/bzsyxf7i", directory="./glad-grass-1976/")


    # Make graphics for 1950 - manipulation
    # make_speaker_example_graphic(directory="./drawn-shape-1950/", start_epoch=299, count=10, interval_epoch=125)
    # make_speaker_example_graphic(directory="./drawn-shape-1950/", start_epoch=299, count=10, epoch_span=1300, x_stretch=100.0, method="1/x")

    # Make graphics for Part1b  (Completely cooperative)
    # directories = {"./true-forest-1934/", "./jolly-donkey-1940/", "./playful-oath-1944/", "./treasured-sound-1945/"}
    # for directory in directories:
    #     make_speaker_example_graphic(directory, start_epoch=299, count=10, interval_epoch=300)
    #     make_speaker_example_graphic(directory, start_epoch=299, count=20, interval_epoch=150)
    #     make_speaker_example_graphic(directory, start_epoch=299, count=10, epoch_span=3000, x_stretch=100.0, method="1/x")
    #     make_speaker_example_graphic(directory, start_epoch=299, count=20, epoch_span=3000, x_stretch=100.0, method="1/x")
    #     make_speaker_example_graphic(directory, start_epoch=299, count=10, epoch_span=3000, x_stretch=0.0, method="1/x")
    #     make_speaker_example_graphic(directory, start_epoch=299, count=20, epoch_span=3000, x_stretch=0.0, method="1/x")

    # Make graphics for Part1b  (Manip-Coop at epoch 1000) Squeezed (odd numbered)
    # directories = ["./azure-leaf-1963/", "./vivid-water-1965/", "./devout-shadow-1967/", "./absurd-feather-1969/"]
    # for directory in directories[2:]:
    #     make_speaker_example_graphic(directory, start_epoch=199, count=10, interval_epoch=300)
    #     make_speaker_example_graphic(directory, start_epoch=199, count=20, interval_epoch=150)
    #     make_speaker_example_graphic(directory, start_epoch=199, count=10, epoch_span=3000, x_stretch=100.0, method="1/x")
    #     make_speaker_example_graphic(directory, start_epoch=199, count=20, epoch_span=3000, x_stretch=100.0, method="1/x")
    #     make_speaker_example_graphic(directory, start_epoch=199, count=10, epoch_span=3000, x_stretch=0.0, method="1/x")
    #     make_speaker_example_graphic(directory, start_epoch=199, count=20, epoch_span=3000, x_stretch=0.0, method="1/x")

    
    # Make graphics for Part1b  (Manip-Coop at epoch 600) Squeezed
    directories = ["./soft-sun-1970/", "./atomic-firefly-1971/", "./divine-deluge-1972/", "./whole-firefly-1973/", "./cool-armadillo-1975/", "./glad-grass-1976/"]
    for directory in directories[-2:]:
        make_speaker_example_graphic(directory, start_epoch=199, count=10, interval_epoch=300)
        make_speaker_example_graphic(directory, start_epoch=199, count=20, interval_epoch=150)
        make_speaker_example_graphic(directory, start_epoch=199, count=10, epoch_span=3000, x_stretch=100.0, method="1/x")
        make_speaker_example_graphic(directory, start_epoch=199, count=20, epoch_span=3000, x_stretch=100.0, method="1/x")
        make_speaker_example_graphic(directory, start_epoch=199, count=10, epoch_span=3000, x_stretch=0.0, method="1/x")
        make_speaker_example_graphic(directory, start_epoch=199, count=20, epoch_span=3000, x_stretch=0.0, method="1/x")

    # (Runs 1950: manipulation, 1931: whitesum, 1934: negative whitesum, 1940: auto-centering, 1944: curvature, 1945: negative curvature)
    # Manip-coop negative curvature: 1973, manip-coop size penalty: 1975
    download_probe_data(run_id="signification-team/signification-game/avnly640", directory="./drawn-shape-1950/")
    download_probe_data(run_id="signification-team/signification-game/wjtjyd8u", directory="./whole-firefly-1973/")
    download_probe_data(run_id="signification-team/signification-game/0in3o71n", directory="./cool-armadillo-1975/")

    download_probe_data(run_id="signification-team/signification-game/2vy8mbfi", directory="./treasured-sound-1945/") # Coop - Curve penalty
    download_probe_data(run_id="signification-team/signification-game/dlezjmad", directory="./true-forest-1934/") # Coop - Size penalty

    make_probe_plot(directories=["./drawn-shape-1950/", "./cool-armadillo-1975/", "./whole-firefly-1973/", "./true-forest-1934/", "./treasured-sound-1945/"],
                    labels=["Manipulation", "Manip-coop - Size Penalty", "Manip-coop - Curve Penalty", "Coop - Size Penalty", "Coop - Curve Penalty"],
                    num_epochs=3300)


def make_graphics_part2():
    # (Runs 1950: manipulation, 1931: whitesum, 1934: negative whitesum, 1940: auto-centering, 1944: curvature, 1945: negative curvature)

    ## Download data for Part2
    # Curvature runs
    # download_speaker_examples(run_id="signification-team/signification-game/jgbklnk8", directory="./sweet-shape-2348/", tom_examples_only=True)   # speaker_selection=[11, 0, 12, 2, 2, 12, 0, 14, 2, 12]
    # download_speaker_examples(run_id="signification-team/signification-game/p1jvmtsq", directory="./worldly-lion-2349/", tom_examples_only=True)    # has larger speaker l2 norm
    # download_speaker_examples(run_id="signification-team/signification-game/1xty9ob3", directory="./frosty-silence-2354/", tom_examples_only=True)    # has larger speaker l2 norm and 3000 epochs
    # download_speaker_examples(run_id="signification-team/signification-game/desfenmt", directory="./tough-cloud-2359/", tom_examples_only=True)    # has larger speaker l2 norm and 3000 epochs and more intense penalty
    # No Penalty runs
    # download_speaker_examples(run_id="signification-team/signification-game/cmrqqctn", directory="./dark-cosmos-2353/", tom_examples_only=True)
    # Whitesum runs
    # download_speaker_examples(run_id="signification-team/signification-game/ni2dajf2", directory="./glad-dew-2358/", tom_examples_only=True)    # has larger speaker l2 norm and 3000 epochs and new whitesum penalty
    
    # Gut runs
    # download_speaker_examples(run_id="signification-team/signification-game/6vtdcxr5", directory="./dazzling-meadow-2352/", tom_examples_only=False)

    # download_probe_data(run_id="signification-team/signification-game/p1jvmtsq", directory="./worldly-lion-2349/")
    # download_probe_data(run_id="signification-team/signification-game/1xty9ob3", directory="./frosty-silence-2354/")
    # download_probe_data(run_id="signification-team/signification-game/cmrqqctn", directory="./dark-cosmos-2353/")
    # download_probe_data(run_id="signification-team/signification-game/6vtdcxr5", directory="./dazzling-meadow-2352/")
    # download_probe_data(run_id="signification-team/signification-game/kytkioqx", directory="./dazzling-puddle-2413/")

    # download_reward_data(run_id="signification-team/signification-game/1xty9ob3", directory="./frosty-silence-2354/")
    # download_reward_data(run_id="signification-team/signification-game/cmrqqctn", directory="./dark-cosmos-2353/")
    # download_reward_data(run_id="signification-team/signification-game/6vtdcxr5", directory="./dazzling-meadow-2352/")
    # download_reward_data(run_id="signification-team/signification-game/kytkioqx", directory="./dazzling-puddle-2413/")

    # download_pr_data(run_id="signification-team/signification-game/cmrqqctn", directory="./dark-cosmos-2353/", listeners=(7,))

    # Make evolution graphics
    directories = ["./frosty-silence-2354/", "./dark-cosmos-2353/", "./dazzling-meadow-2352/", "./tough-cloud-2359/", "./glad-dew-2358/"]
    # speaker_selections = [[12, 8, 12, 2, 2, 12, 0, 14, 2, 12],
    #                       [12, 8, 12, 2, 2, 12, 0, 14, 2, 12]]
    
    speaker_selection = [12, 8, 12, 2, 2, 12, 0, 14, 2, 12]
    # abbreviated_speaker_selection = [12, 2, 12, 2, 12]
    # abbreviated_referent_selection = [0, 4, 5, 8, 9]
    name_prefixes = ["tom_", "", "tom_", "tom_"]
    for directory, fname_prefix in list(zip(directories, name_prefixes))[:1]:
        make_speaker_example_graphic(directory, image_dim=32, fname_prefix=fname_prefix, speaker_selection=speaker_selection, referent_selection=list(range(10)), start_epoch=149, count=15, interval_epoch=180)
        make_speaker_example_graphic(directory, image_dim=32, fname_prefix=fname_prefix, speaker_selection=speaker_selection, referent_selection=list(range(10)), start_epoch=149, count=15, epoch_span=2800, x_stretch=100.0, method="1/x")
        make_speaker_example_graphic(directory, image_dim=32, fname_prefix=fname_prefix, speaker_selection=speaker_selection, referent_selection=list(range(10)), start_epoch=149, count=15, epoch_span=2800, x_stretch=0.0, method="1/x")
        # make_speaker_example_graphic(directory, image_dim=32, fname_prefix="tom_", one_sign=(5, 12), vertical=False, start_epoch=149, count=20, interval_epoch=75)
        # make_speaker_example_graphic(directory, image_dim=32, fname_prefix="tom_", one_sign=(5, 12), vertical=False, start_epoch=149, count=10, epoch_span=1800, x_stretch=200.0, method="1/x")
        # make_speaker_example_graphic(directory, image_dim=32, fname_prefix="tom_", one_sign=(5, 12), vertical=False, start_epoch=149, count=10, epoch_span=1800, x_stretch=0.0, method="1/x")

    # make_probe_plot(directories=["./worldly-lion-2349/"],
    #     labels=["Iconicity"],
    #     all_speakers_avg=True,
    #     num_epochs=1800,
    #     epoch_start=150,
    #     markers_on=np.array([150, 260, 685, 1040, 1470, 1720])-150)

    # make_reward_plot(directories=("./dazzling-meadow-2352/", "./dark-cosmos-2353/", "./dazzling-puddle-2413/"),
    #     labels=("Behaviorist", "Inferential", "Inferential - no P_ref"),
    #     num_epochs=2800,
    #     epoch_start=0)
    
    # make_probe_plot(directories=("./dazzling-meadow-2352/", "./dark-cosmos-2353/", "./dazzling-puddle-2413/"),
    #     labels=("Behaviorist", "Inferential", "Inferential - no P_ref"),
    #     all_speakers_avg=True,
    #     num_epochs=2800,
    #     epoch_start=0)

    # ("Bicycle", "Butterfly", "Camel", "Crab", "Dolphin", "Palm Tree", "Rocket", "Snail", "Snake", "Spider") # list(range(10))

    # make_pr_plot(directory="./dark-cosmos-2353/",
    #     referent_labels=("Snail", "Dolphin", "Palm Tree", "Rocket", "Spider"),
    #     referent_nums=(7, 4, 5, 6, 9),
    #     num_epochs=3000,
    #     epoch_start=0,
    #     agent_num=7,
    #     log_scale=True)

    # make_animation(directory="./dark-cosmos-2353/", label="Inferential")

    # download_communication_success_data(run_id="signification-team/signification-game/q2jk11i4", directory="./youthful-dust-2425/")
    # download_communication_success_data(run_id="signification-team/signification-game/zsbo8nlx", directory="./resilient-violet-2427/")
    # download_communication_success_data(run_id="signification-team/signification-game/7qvg2jqt", directory="./absurd-pyramid-2428/")

    # download_probe_data(run_id="signification-team/signification-game/q2jk11i4", directory="./youthful-dust-2425/")
    # download_probe_data(run_id="signification-team/signification-game/zsbo8nlx", directory="./resilient-violet-2427/")
    # download_probe_data(run_id="signification-team/signification-game/7qvg2jqt", directory="./absurd-pyramid-2428/")

    # make_com_success_plot(directories=("./youthful-dust-2425/", "./resilient-violet-2427/", "./absurd-pyramid-2428/"),
    #     labels=("Behaviorist", "Inferential", "Inf. - P_ref"),
    #     num_epochs=2800,
    #     epoch_start=0)

    # make_probe_plot(directories=("./youthful-dust-2425/", "./resilient-violet-2427/", "./absurd-pyramid-2428/"),
    #     labels=("Behaviorist", "Inferential", "Inf. - P_ref"),
    #     all_speakers_avg=True,
    #     num_epochs=2800,
    #     epoch_start=0)

    # download_pr_data(run_id="signification-team/signification-game/zsbo8nlx", directory="./resilient-violet-2427/", listeners=(7,))

    # make_pr_plot(directory="./resilient-violet-2427/",
    #     referent_labels=("Snail", "Dolphin", "Palm Tree", "Rocket", "Spider"),
    #     referent_nums=(7, 4, 5, 6, 9),
    #     num_epochs=3000,
    #     epoch_start=0,
    #     agent_num=7,
    #     log_scale=True)


def remake_graphics_part1():
    # (Runs 1950: manipulation, 1931: whitesum, 1934: negative whitesum, 1940: auto-centering, 1944: curvature, 1945: negative curvature)

    # The re-runs aren't always ordered perfectly, but I have verified that they match the intended experiment parameters
    # 1975 - Manip-coop size. 2368-2372
    # 1973 - Manip-coop curve. 2373-2377
    # 1934 - Coop size. 2378-2382
    # 1945 - Coop curve. 2383-2387

    ### Re-runs of 1950: 2363-2367, 2388-2392
    manipulation_runs = ["./comic-rain-2363/", "./rosy-field-2364/", "./dainty-surf-2364/", "./jolly-waterfall-2366/", "./blooming-donkey-2367/", "./tough-jazz-2388/", "./fancy-glitter-2388/", "./lyric-firebrand-2388/", "./woven-lion-2391/", "./rose-shape-2392/"]
    # download_probe_data(run_id="signification-team/signification-game/rnucselq", directory="./comic-rain-2363/")
    # download_probe_data(run_id="signification-team/signification-game/s69h3sh7", directory="./rosy-field-2364/")
    # download_probe_data(run_id="signification-team/signification-game/yf15il82", directory="./dainty-surf-2364/")
    # download_probe_data(run_id="signification-team/signification-game/xqzzhed0", directory="./jolly-waterfall-2366/")
    # download_probe_data(run_id="signification-team/signification-game/uilb1k7z", directory="./blooming-donkey-2367/")
    # download_probe_data(run_id="signification-team/signification-game/0ypainpy", directory="./tough-jazz-2388/")
    # download_probe_data(run_id="signification-team/signification-game/l3qkj31h", directory="./fancy-glitter-2388/")
    # download_probe_data(run_id="signification-team/signification-game/wnvdroiz", directory="./lyric-firebrand-2388/")
    # download_probe_data(run_id="signification-team/signification-game/9gd4pq1t", directory="./woven-lion-2391/")
    # download_probe_data(run_id="signification-team/signification-game/spcb6kxy", directory="./rose-shape-2392/")
    # download_speaker_examples(run_id="signification-team/signification-game/s69h3sh7", directory="./rosy-field-2364/")

    ### Re-runs of 1975: 2368-2372, 2393-2397
    manip_coop_size_runs = ["./lilac-pond-2368/", "./dauntless-pine-2369/", "./different-flower-2370/", "./glowing-bird-2371/", "./olive-fog-2372/", "./rural-grass-2393/", "./rosy-bee-2395/", "./treasured-serenity-2394/", "./smart-leaf-2396/", "./gallant-sponge-2397/"]
    # download_probe_data(run_id="signification-team/signification-game/p2nna36h", directory="./lilac-pond-2368/")
    # download_probe_data(run_id="signification-team/signification-game/jvpihsmi", directory="./dauntless-pine-2369/")
    # download_probe_data(run_id="signification-team/signification-game/24uekkvx", directory="./different-flower-2370/")
    # download_probe_data(run_id="signification-team/signification-game/tjaw0n1s", directory="./glowing-bird-2371/")
    # download_probe_data(run_id="signification-team/signification-game/rsyjg7o0", directory="./olive-fog-2372/")
    # download_probe_data(run_id="signification-team/signification-game/7an6dm5y", directory="./rural-grass-2393/")
    # download_probe_data(run_id="signification-team/signification-game/28lyzq3d", directory="./rosy-bee-2395/")
    # download_probe_data(run_id="signification-team/signification-game/anpl4r4s", directory="./treasured-serenity-2394/")
    # download_probe_data(run_id="signification-team/signification-game/a0661enj", directory="./smart-leaf-2396/")
    # download_probe_data(run_id="signification-team/signification-game/kpfjld6h", directory="./gallant-sponge-2397/")

    ### Re-runs of 1973: 2373-2377, 2398-2402
    manip_coop_curve_runs = ["./proud-violet-2373/", "./worthy-lake-2373/", "./good-elevator-2375/", "./light-pond-2376/", "./twilight-capybara-2377/", "./dazzling-firebrand-2398/", "./earnest-surf-2399/", "./serene-lion-2400/", "./eager-dew-2400/", "./wise-capybara-2403/"]
    # download_probe_data(run_id="signification-team/signification-game/ou6fhk9c", directory="./proud-violet-2373/")
    # download_probe_data(run_id="signification-team/signification-game/takhyi9a", directory="./worthy-lake-2373/")
    # download_probe_data(run_id="signification-team/signification-game/pa66d5fw", directory="./good-elevator-2375/")
    # download_probe_data(run_id="signification-team/signification-game/pb176dbx", directory="./light-pond-2376/")
    # download_probe_data(run_id="signification-team/signification-game/7wfskzt7", directory="./twilight-capybara-2377/")
    # download_probe_data(run_id="signification-team/signification-game/nb2shync", directory="./dazzling-firebrand-2398/")
    # download_probe_data(run_id="signification-team/signification-game/ve2ec3em", directory="./earnest-surf-2399/")
    # download_probe_data(run_id="signification-team/signification-game/8nlr7lox", directory="./serene-lion-2400/")
    # download_probe_data(run_id="signification-team/signification-game/b09wtocw", directory="./eager-dew-2400/")
    # download_probe_data(run_id="signification-team/signification-game/e3iptn8e", directory="./wise-capybara-2403/")

    ### Re-runs of 1934: 2378-2382, 2403-2407
    coop_size_runs = ["./kind-terrain-2378/", "./grateful-vortex-2379/", "./twilight-sound-2379/", "./robust-planet-2381/", "./spring-silence-2382/", "./smooth-durian-2402/", "./crisp-elevator-2404/", "./fancy-dragon-2405/", "./firm-deluge-2406/", "./genial-meadow-2407/"]
    # download_probe_data(run_id="signification-team/signification-game/81vlracz", directory="./kind-terrain-2378/")
    # download_probe_data(run_id="signification-team/signification-game/5yblw5tr", directory="./grateful-vortex-2379/")
    # download_probe_data(run_id="signification-team/signification-game/da2jf77x", directory="./twilight-sound-2379/")
    # download_probe_data(run_id="signification-team/signification-game/hty54uqe", directory="./robust-planet-2381/")
    # download_probe_data(run_id="signification-team/signification-game/uo75i1aw", directory="./spring-silence-2382/")
    # download_probe_data(run_id="signification-team/signification-game/ssei5nhb", directory="./smooth-durian-2402/")
    # download_probe_data(run_id="signification-team/signification-game/7wvdxdd8", directory="./crisp-elevator-2404/")
    # download_probe_data(run_id="signification-team/signification-game/cxjsd0bv", directory="./fancy-dragon-2405/")
    # download_probe_data(run_id="signification-team/signification-game/7wjq2453", directory="./firm-deluge-2406/")
    # download_probe_data(run_id="signification-team/signification-game/79ljdh2y", directory="./genial-meadow-2407/")
    # download_speaker_examples(run_id="signification-team/signification-game/81vlracz", directory="./kind-terrain-2378/")

    ### Re-runs of 1945: 2383-2387, 2408-2412
    coop_curve_runs = ["./polished-wave-2383/", "./eager-thunder-2384/", "./grateful-fire-2385/", "./rich-bee-2386/", "./hearty-grass-2387/", "./deep-fire-2408/", "./logical-smoke-2409/", "./vocal-yogurt-2410/", "./breezy-plasma-2411/", "./hearty-tree-2412/"]
    # download_probe_data(run_id="signification-team/signification-game/ubzkvnrm", directory="./polished-wave-2383/")
    # download_probe_data(run_id="signification-team/signification-game/5lud7k0f", directory="./eager-thunder-2384/")
    # download_probe_data(run_id="signification-team/signification-game/aieg7rt5", directory="./grateful-fire-2385/")
    # download_probe_data(run_id="signification-team/signification-game/61kerwgm", directory="./rich-bee-2386/")
    # download_probe_data(run_id="signification-team/signification-game/qkg1mtca", directory="./hearty-grass-2387/")
    # download_probe_data(run_id="signification-team/signification-game/o8p0nug9", directory="./deep-fire-2408/")
    # download_probe_data(run_id="signification-team/signification-game/uup1s1wu", directory="./logical-smoke-2409/")
    # download_probe_data(run_id="signification-team/signification-game/eeey2myl", directory="./vocal-yogurt-2410/")
    # download_probe_data(run_id="signification-team/signification-game/9hq2r29s", directory="./breezy-plasma-2411/")
    # download_probe_data(run_id="signification-team/signification-game/etmfynm2", directory="./hearty-tree-2412/")
    # download_speaker_examples(run_id="signification-team/signification-game/ubzkvnrm", directory="./polished-wave-2383/")


    # make_avg_probe_plot([manipulation_runs,
    #                      manip_coop_size_runs,
    #                      manip_coop_curve_runs,
    #                      coop_size_runs,
    #                      coop_curve_runs],
    #                      ["Manipulation",
    #                        "Manip→Coop - Size Penalty",
    #                        "Manip→Coop - Curve Penalty",
    #                        "Coop - Size Penalty",
    #                        "Coop - Curve Penalty"],
    #                          rolling_window=100, t_val=1.833)

    make_simple_animation(directory="./comic-rain-2363/", fname_prefix="", image_dim=28, labels=False)
    # make_simple_animation(directory="./kind-terrain-2378/", fname_prefix="", image_dim=28, labels=False)
    # make_simple_animation(directory="./polished-wave-2383/", fname_prefix="", image_dim=28, labels=False)

    # speaker_selection = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # directories = ["./comic-rain-2363/", "./polished-wave-2383/"]
    
    # for directory in directories[-2:]:
    #     make_speaker_example_graphic(directory, image_dim=28, speaker_selection=speaker_selection, referent_selection=list(range(10)), start_epoch=149, count=15, interval_epoch=180)
    #     make_speaker_example_graphic(directory, image_dim=28, speaker_selection=speaker_selection, referent_selection=list(range(10)), start_epoch=149, count=15, epoch_span=2800, x_stretch=100.0, method="1/x")
    #     make_speaker_example_graphic(directory, image_dim=28, speaker_selection=speaker_selection, referent_selection=list(range(10)), start_epoch=149, count=15, epoch_span=2800, x_stretch=0.0, method="1/x")
    #     make_speaker_example_graphic(directory, start_epoch=199, count=10, interval_epoch=300, speaker_selection=speaker_selection, referent_selection=list(range(10)), fname_prefix="", image_dim=28)
    #     make_speaker_example_graphic(directory, start_epoch=199, count=20, interval_epoch=150, speaker_selection=speaker_selection, referent_selection=list(range(10)), fname_prefix="", image_dim=28)
    #     make_speaker_example_graphic(directory, start_epoch=199, count=10, epoch_span=3000, x_stretch=100.0, method="1/x", speaker_selection=speaker_selection, referent_selection=list(range(10)), fname_prefix="", image_dim=28)
    #     make_speaker_example_graphic(directory, start_epoch=199, count=20, epoch_span=3000, x_stretch=100.0, method="1/x", speaker_selection=speaker_selection, referent_selection=list(range(10)), fname_prefix="", image_dim=28)
    #     make_speaker_example_graphic(directory, start_epoch=199, count=10, epoch_span=3000, x_stretch=0.0, method="1/x", speaker_selection=speaker_selection, referent_selection=list(range(10)), fname_prefix="", image_dim=28)
    #     make_speaker_example_graphic(directory, start_epoch=199, count=20, epoch_span=3000, x_stretch=0.0, method="1/x", speaker_selection=speaker_selection, referent_selection=list(range(10)), fname_prefix="", image_dim=28)


def make_animation(directory, label, num_epochs=2800, epoch_start=0, fname_prefix="tom_", image_dim=32, referent_selection=list(range(10)), speaker_selection=list(np.zeros(10, dtype=int))):
    height_dx = image_dim + 2   # Assuming 2px border

    FPS=20
    
    # Load reward data
    reward_data = pd.read_csv(os.path.join(directory, f"reward_for_speaker_images_all_listeners.csv"))

    # Load probe data
    probe_data = pd.read_csv(os.path.join(directory, f"probe_entropy_all_speakers.csv"))

    # Load image data
    image_dir = os.path.join(directory, "media/images/env/")
    files = os.listdir(image_dir)
    fname_template = fname_prefix+"speaker_examples_"

    sorted_files = sorted([f for f in files if f.startswith(fname_template)],
                         key=lambda x: int(x.split(fname_template)[1].split('_')[0]))
    sorted_files = [os.path.join(image_dir, f) for f in sorted_files]
    
    # Grab only every 5th reward datapoint
    reward_data = reward_data.iloc[::5]
    probe_data = probe_data.iloc[::5]
    
    print(len(reward_data))
    print(len(sorted_files))

    # Initialize figure
    fig, axes = plt.subplots(3, 1, figsize=(4, 5.5))
    probe_ax = axes[0]
    probe_ax.set_xlim(0, 600)
    probe_ax.set_ylim(0.0, 1.0)
    probe_ax.set_title("Symbolicity")
    probe_ax.set_xlabel("Epoch")
    img_ax = axes[1]
    reward_ax = axes[2]
    reward_ax.set_xlim(0, 600)
    reward_ax.set_ylim(-0.1, 1.0)
    reward_ax.set_title("Communication Success")
    reward_ax.set_xlabel("Epoch")
    # reward_ax.set_ylabel("Reward")
    reward_line, = reward_ax.plot([], [], lw=2)
    probe_line, = probe_ax.plot([], [], lw=2)

    def update(frame):
        # Load and update image
        # img = plt.imread(sorted_files[frame])

        ##### Crop img here if you want.
        img = Image.open(sorted_files[frame])
        img_array = np.array(img)
        # local_height_dx = height_dx if i == len(image_files) - 1 else height_dx
        row_imgs = []
        for ii, j in zip(referent_selection, speaker_selection):
            local_width_dx = height_dx #if ii == referent_selection[-1] else height_dx
            row_imgs.append(img_array[height_dx*j:height_dx*j+height_dx, height_dx*ii:height_dx*ii+local_width_dx])
        row_img = np.concatenate(row_imgs, axis=1)
        ###########

        img_ax.clear()
        img_ax.imshow(row_img)
        img_ax.axis("off")
        img_ax.set_title(f"Signals at Epoch {frame}")
        # ("Bicycle", "Butterfly", "Camel", "Crab", "Dolphin", "Palm Tree", "Rocket", "Snail", "Snake", "Spider")
        img_ax.text(-2, 45, " ".join(("Bicycle", "Butterfly", "Camel", " Crab  ", "Dolphin", " Tree ", " Rocket", "  Snail  ", "Snake  ", "Spider")), size="xx-small")
        
        # Update reward curve
        reward_line.set_data(range(frame + 1), reward_data[:frame + 1])
        probe_line.set_data(range(frame + 1), probe_data[:frame + 1])
        return probe_line, img_ax, reward_line
    
    # print(animation.writers.list())
    
    ani = animation.FuncAnimation(fig, update, frames=600, interval=1000//FPS)

    ani.save(f"../joint-plots/vid_{directory.split('-')[-1][:-1]}.mp4", writer="ffmpeg", fps=FPS)

    print("Saved file")


def make_multi_animation(directories, labels, num_epochs=2800, epoch_start=0, fname_prefixes=["", "tom_"], image_dim=32, referent_selection=list(range(10)), speaker_selection=list(np.zeros(10, dtype=int))):
    height_dx = image_dim + 2   # Assuming 2px border

    FPS=20
    
    # Load reward data
    reward_datas = [pd.read_csv(os.path.join(directory, "success_rate_all_referents.csv")) for directory in directories]

    # Load probe data
    probe_datas = [pd.read_csv(os.path.join(directory, f"probe_entropy_all_speakers.csv")) for directory in directories]

    # Load image data
    image_dirs = [os.path.join(directory, "media/images/env/") for directory in directories]
    
    files_list = [os.listdir(image_dir) for image_dir in image_dirs]
    fname_templates = [fname_prefix+"speaker_examples_" for fname_prefix in fname_prefixes]

    sorted_files_list = [sorted([os.path.join(image_dir, f) for f in files if f.startswith(fname_template)],
                         key=lambda x: int(x.split(fname_template)[1].split('_')[0])) for files, image_dir, fname_template in zip(files_list, image_dirs, fname_templates)]
    
    
    # Grab only every 5th reward datapoint
    reward_datas = [reward_data.iloc[::5] for reward_data in reward_datas]
    probe_datas = [probe_data.iloc[::5] for probe_data in probe_datas]
    
    # print(len(reward_data))
    # print(len(sorted_files))

    # Initialize figure
    fig, axes = plt.subplots(3, 1, figsize=(4, 7))
    probe_ax = axes[0]
    probe_ax.set_xlim(0, 3000)
    probe_ax.set_ylim(0.0, 1.0)
    probe_ax.set_title("Symbolicity")
    probe_ax.set_xlabel("Epoch")
    img_ax = axes[1]
    reward_ax = axes[2]
    reward_ax.set_xlim(0, 3000)
    reward_ax.set_ylim(-0.1, 1.0)
    reward_ax.set_title("Communication Success")
    reward_ax.set_xlabel("Epoch")
    # reward_ax.set_ylabel("Reward")

    reward_lines = [reward_ax.plot([], [], lw=2, label=label) for label in labels]
    probe_lines = [probe_ax.plot([], [], lw=2, label=label) for label in labels]

    probe_ax.legend(loc="upper right", fontsize="medium")
    reward_ax.legend(loc="lower right", fontsize="medium")

    def update(frame):
        # Load and update image
        # img = plt.imread(sorted_files[frame])

        ##### Crop img here if you want.
        full_row_imgs = []
        for sorted_files in sorted_files_list:
            img = Image.open(sorted_files[frame])
            img_array = np.array(img)
            # local_height_dx = height_dx if i == len(image_files) - 1 else height_dx
            row_imgs = []
            for ii, j in zip(referent_selection, speaker_selection):
                local_width_dx = height_dx #if ii == referent_selection[-1] else height_dx
                row_imgs.append(img_array[height_dx*j:height_dx*j+height_dx, height_dx*ii:height_dx*ii+local_width_dx])
            row_img = np.concatenate(row_imgs, axis=1)
            full_row_imgs.append(row_img)
        ###########

        row_img = np.concatenate(full_row_imgs, axis=0)

        img_ax.clear()
        img_ax.imshow(row_img)
        img_ax.axis("off")
        img_ax.set_title(f"Signals at Epoch {frame*5}")

        img_ax.text(-2, 116, " ".join(("Bicycle", "Butterfly", "Camel", " Crab  ", "Dolphin", " Tree ", " Rocket", "  Snail  ", "Snake  ", "Spider")), size="xx-small")
        img_ax.text(-52, 20, "Behaviorist", size="xx-small")
        img_ax.text(-50, 53, "Inf. –P_Ref", size="xx-small")
        img_ax.text(-50, 86, "Inferential", size="xx-small")

        # Update reward curve
        for reward_line, reward_data in zip(reward_lines, reward_datas):
            reward_line[0].set_data(np.array(range(frame + 1)) * 5, reward_data[:frame + 1])
        
        for probe_line, probe_data in zip(probe_lines, probe_datas):
            probe_line[0].set_data(np.array(range(frame + 1)) * 5, probe_data[:frame + 1])
        
        return [line[0] for line in reward_lines + probe_lines] + [img_ax]
    
    # print(animation.writers.list())
    
    ani = animation.FuncAnimation(fig, update, frames=600, interval=1000//FPS)

    uuidstr = str(uuid.uuid4())[:5]

    save_suffix = "_".join([d.split('-')[-1][:4] for d in directories])
    ani.save(f"../joint-plots/vid_multi_{save_suffix}_{uuidstr}.mp4", writer="ffmpeg", fps=FPS)

    print("Saved file")

def make_multi_animation_noprobe(directories, labels, num_imgs=None, fname_prefixes=["", "tom_"], image_dim=32, referent_selection=list(range(10)), speaker_selection=list(np.zeros(10, dtype=int))):
    height_dx = image_dim + 2   # Assuming 2px border

    FPS=20
    
    # Load reward data
    reward_datas = [pd.read_csv(os.path.join(directory, "success_rate_all_referents.csv")) for directory in directories]

    # Load image data
    image_dirs = [os.path.join(directory, "media/images/env/") for directory in directories]
    
    files_list = [os.listdir(image_dir) for image_dir in image_dirs]
    fname_templates = [fname_prefix+"speaker_examples_" for fname_prefix in fname_prefixes]

    sorted_files_list = [sorted([os.path.join(image_dir, f) for f in files if f.startswith(fname_template)],
                         key=lambda x: int(x.split(fname_template)[1].split('_')[0])) for files, image_dir, fname_template in zip(files_list, image_dirs, fname_templates)]
    
    if num_imgs:
        sorted_files_list = sorted_files_list[:num_imgs]

    # Grab only every 5th reward datapoint
    reward_datas = [reward_data.iloc[::5] for reward_data in reward_datas]
    
    # print(len(reward_data))
    # print(len(sorted_files))

    # Initialize figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 2))
    img_ax = axes[0]
    reward_ax = axes[1]
    reward_ax.set_xlim(0, 3000)
    reward_ax.set_ylim(-0.1, 1.0)
    reward_ax.set_title("Communication Success")
    reward_ax.set_xlabel("Epoch")
    # reward_ax.set_ylabel("Reward")

    reward_lines = [reward_ax.plot([], [], lw=2, label=label) for label in labels]

    reward_ax.legend(loc="lower right", fontsize="medium")

    def update(frame):
        # Load and update image
        # img = plt.imread(sorted_files[frame])

        ##### Crop img here if you want.
        full_row_imgs = []
        for sorted_files in sorted_files_list:
            img = Image.open(sorted_files[frame])
            img_array = np.array(img)
            # local_height_dx = height_dx if i == len(image_files) - 1 else height_dx
            row_imgs = []
            for ii, j in zip(referent_selection, speaker_selection):
                local_width_dx = height_dx #if ii == referent_selection[-1] else height_dx
                row_imgs.append(img_array[height_dx*j:height_dx*j+height_dx, height_dx*ii:height_dx*ii+local_width_dx])
            row_img = np.concatenate(row_imgs, axis=1)
            full_row_imgs.append(row_img)
        ###########

        row_img = np.concatenate(full_row_imgs, axis=0)

        img_ax.clear()
        img_ax.imshow(row_img)
        img_ax.axis("off")
        img_ax.set_title(f"Signals at Epoch {frame*5}")

        img_ax.text(-2, 78, " ".join(("Bicycle", "Butterfly", "Camel", " Crab   ", "Dolphin", " Tree ", "   Rocket", "  Snail  ", "Snake  ", "Spider")), size="xx-small")
        img_ax.text(-52, 20, "Behaviorist", size="xx-small")
        img_ax.text(-50, 53, "Inf. –P_Ref", size="xx-small")

        # Update reward curve
        for reward_line, reward_data in zip(reward_lines, reward_datas):
            reward_line[0].set_data(np.array(range(frame + 1)) * 5, reward_data[:frame + 1])
        
        return [line[0] for line in reward_lines] + [img_ax]
    
    # print(animation.writers.list())
    
    ani = animation.FuncAnimation(fig, update, frames=600, interval=1000//FPS)

    uuidstr = str(uuid.uuid4())[:5]

    save_suffix = "_".join([d.split('-')[-1][:4] for d in directories])
    ani.save(f"../joint-plots/vid_multi_noprobe_{save_suffix}_{uuidstr}.mp4", writer="ffmpeg", fps=FPS)

    print("Saved file")


def make_simple_animation(directory, labels=True, frames=600, fname_prefix="tom_", image_dim=32, referent_selection=list(range(10)), speaker_selection=list(np.zeros(10, dtype=int)), fname_suffix_info="", fps=20, cmap='viridis'):
    height_dx = image_dim + 2   # Assuming 2px border
    if fname_suffix_info != "":
        fname_suffix_info = "_"+fname_suffix_info
    
    # Load image data
    image_dir = os.path.join(directory, "media/images/env/")
    files = os.listdir(image_dir)
    fname_template = fname_prefix+"speaker_examples_"

    sorted_files = sorted([f for f in files if f.startswith(fname_template)],
                         key=lambda x: int(x.split(fname_template)[1].split('_')[0]))
    sorted_files = [os.path.join(image_dir, f) for f in sorted_files]

    # print(sorted_files)
    

    # Initialize figure
    fig, axe = plt.subplots(figsize=(4, 2))
    img_ax = axe

    def update(frame):
        # Load and update image
        # img = plt.imread(sorted_files[frame])

        ##### Crop img here if you want.
        img = Image.open(sorted_files[frame])
        img_array = np.array(img)
        # local_height_dx = height_dx if i == len(image_files) - 1 else height_dx
        row_imgs = []
        for ii, j in zip(referent_selection, speaker_selection):
            local_width_dx = height_dx #if ii == referent_selection[-1] else height_dx
            row_imgs.append(img_array[height_dx*j:height_dx*j+height_dx, height_dx*ii:height_dx*ii+local_width_dx])
        row_img = np.concatenate(row_imgs, axis=1)
        ###########

        img_ax.clear()
        img_ax.imshow(row_img, cmap=cmap)
        img_ax.axis("off")
        img_ax.set_title(f"Signals at Epoch {frame * 5}")
        # ("Bicycle", "Butterfly", "Camel", "Crab", "Dolphin", "Palm Tree", "Rocket", "Snail", "Snake", "Spider")
        if labels:
            img_ax.text(-2, 45, " ".join(("Bicycle", "Butterfly", "Camel", " Crab  ", "Dolphin", " Tree ", " Rocket", "  Snail  ", "Snake  ", "Spider")), size="xx-small")

        return img_ax
    
    # print(animation.writers.list())
    
    ani = animation.FuncAnimation(fig, update, frames=min(frames, len(sorted_files)), interval=1000//fps)
    
    if frames > len(sorted_files):
        print(f"You are asking for {frames} frames but there are only {len(sorted_files)} images")

    uuidstr = str(uuid.uuid4())[:5]

    ani.save(f"../joint-plots/vid_simple_{directory.split('-')[-1][:-1]}{fname_suffix_info}_{uuidstr}.mp4", writer="ffmpeg", fps=fps)

    print("Saved file")


def make_simple_animation_same_sign_multi_agent(directories, referent_coordinates=((0,1), (4,5)), labels=[], fname_prefix="tom_", epochs=3000, image_dim=32, start_epoch=500):
    all_referents = ["Bicycle", "Butterfly", "Camel", "Crab", "Dolphin", "Palm Tree", "Rocket", "Snail", "Snake", "Spider"]

    # Maintain a canonical order when displaying the plots
    referent_coordinates.sort(key=lambda x: x[1])

    referent_labels = [all_referents[ref[1]] for ref in referent_coordinates]
    
    height_dx = image_dim + 2   # Assuming 2px border

    FPS=20

    sorted_files_by_dir = []
    
    for directory in directories:
        # Load image data
        image_dir = os.path.join(directory, "media/images/env/")
        files = os.listdir(image_dir)
        fname_template = fname_prefix+"speaker_examples_"

        sorted_files = sorted([f for f in files if f.startswith(fname_template)],
                            key=lambda x: int(x.split(fname_template)[1].split('_')[0]))
        sorted_files = [os.path.join(image_dir, f) for f in sorted_files]
        sorted_files_by_dir.append(sorted_files)
    

    # Initialize figure
    fig, axe = plt.subplots(figsize=(4, 2))
    img_ax = axe

    pos = img_ax.get_position()
    img_ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.9])

    def update(frame):
        # Load and update image
        # img = plt.imread(sorted_files[frame])

        ##### Crop img here if you want.
        row_imgs = []
        for sorted_files in sorted_files_by_dir:
            img = Image.open(sorted_files[frame])
            img_array = np.array(img)
            # local_height_dx = height_dx if i == len(image_files) - 1 else height_dx
            col_images = []
            for (x, y) in referent_coordinates:
                local_width_dx = height_dx #if ii == referent_selection[-1] else height_dx
                col_images.append(img_array[height_dx*x:height_dx*x+height_dx, height_dx*y:height_dx*y+local_width_dx])
            col_image = np.concatenate(col_images, axis=0)
            row_imgs.append(col_image)
        row_img = np.concatenate(row_imgs, axis=1)
        
        ###########

        img_ax.clear()
        img_ax.imshow(row_img, cmap="viridis")
        img_ax.axis("off")
        img_ax.set_title(f"Signals at Epoch {frame * 20 + start_epoch}", pad=15)
        # ("Bicycle", "Butterfly", "Camel", "Crab", "Dolphin", "Palm Tree", "Rocket", "Snail", "Snake", "Spider")
        if labels:
            img_ax.text(-35, 90, "\n\n\n\n".join(referent_labels), size="xx-small")
            img_ax.text(0, -3, "   " + "     ".join(f"Agent {i}" for i in range(7)), size="xx-small")
        

        return img_ax
    
    # print(animation.writers.list())
    
    ani = animation.FuncAnimation(fig, update, frames=(6000 - start_epoch) // 20 + 1, interval=1000//FPS)

    uuidstr = str(uuid.uuid4())[:5]

    directory_names = "_".join(directory.split('-')[-1][:-1] for directory in directories)
    ani.save(f"../joint-plots/vid_multi_onesign_{directory_names}_{uuidstr}.mp4", writer="ffmpeg", fps=FPS)

    print(f"Saved file: ../joint-plots/vid_multi_onesign_{directory_names}_{uuidstr}.mp4")


def make_labeled_animation(directories, referent_coordinates, run_labels, fname_prefix="tom_", start_epoch=500, epochs=3000, image_dim=32):
    all_referents = ["Bicycle", "Butterfly", "Camel", "Crab", "Dolphin", "Palm Tree", "Rocket", "Snail", "Snake", "Spider"]

    # Maintain a canonical order when displaying the plots
    referent_coordinates.sort(key=lambda x: x[1])

    referent_labels = [all_referents[ref[1]] for ref in referent_coordinates]
    
    height_dx = image_dim + 2   # Assuming 2px border

    FPS=20

    sorted_files_by_dir = []
    
    for directory in directories:
        # Load image data
        image_dir = os.path.join(directory, "media/images/env/")
        files = os.listdir(image_dir)
        fname_template = fname_prefix+"speaker_examples_"

        sorted_files = sorted([f for f in files if f.startswith(fname_template)],
                            key=lambda x: int(x.split(fname_template)[1].split('_')[0]))
        sorted_files = [os.path.join(image_dir, f) for f in sorted_files]
        sorted_files_by_dir.append(sorted_files)

    num_runs = len(directories)
    num_referents = len(referent_coordinates) # Makes more sense if referents are unique

    # Initialize figure
    fig, axe = plt.subplots(num_referents, num_runs, figsize=(num_runs * 3, num_referents * 3))
    img_ax = axe

    end_epoch = epochs
    step_size = 10

    num_frames = (end_epoch - start_epoch) // step_size

    def update(frame):
        curr_epoch = start_epoch + (frame * step_size)

        for i in range(num_referents):
            for j in range(num_runs):
                ax = img_ax[i, j]
                run_files = sorted_files_by_dir[j]
                img = Image.open(run_files[frame])
                img_array = np.array(img)
                row_coord, col_coord = referent_coordinates[i]
                small_img = img_array[row_coord * height_dx : (row_coord + 1) * height_dx, col_coord * height_dx : (col_coord + 1) * height_dx]
                ax.imshow(small_img, cmap="viridis")
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis='both', which='both', length=0)

        ###########
        for idx, label in enumerate(referent_labels):
            ax = img_ax[idx, 0]
            ax.set_ylabel(f"{referent_labels[idx]}", rotation=0, labelpad=55, fontsize=18)

        for idx, label in enumerate(directories):
            ax = img_ax[0, idx]
            ax.set_title(f"{run_labels[idx]}", fontsize=18, pad=10)

        fig.suptitle(f"Signals at Epoch {curr_epoch}", fontsize=24, y=0.98)  # Move suptitle down slightly
        fig.subplots_adjust(left=0.15, top=0.85, wspace=0, hspace=0)
        return img_ax

    print(directories)

    # print(animation.writers.list())
    
    ani = animation.FuncAnimation(fig, update, frames=epochs//step_size, interval=1000//FPS)

    uuidstr = str(uuid.uuid4())[:5]

    directory_names = "_".join(directory.split('-')[-1][:-1] for directory in directories)
    ani.save(f"../joint-plots/vid_multi_onesign_{directory_names}_{uuidstr}.mp4", writer="ffmpeg", fps=FPS)

    print(f"Saved file: ../joint-plots/vid_multi_onesign_{directory_names}_{uuidstr}.mp4")

def make_pr_plot(directory, referent_labels, referent_nums, num_epochs=None, epoch_start=0, agent_num=None, log_scale=False):
    if agent_num:
        datas = [pd.read_csv(os.path.join(directory, f"inference_pr_listener_{agent_num}_referent_{ref_num}.csv")) for ref_num in referent_nums]
    else:
        datas = [pd.read_csv(os.path.join(directory, f"inference_pr_referent_{ref_num}.csv")) for ref_num in referent_nums]
    
    sns.set_theme(style="darkgrid")

    # Plot the data with larger font
    plt.yscale("log", base=np.e)
    plt.xscale("log")
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_xscale("log")
    if log_scale:
        # ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 1, 0.2)**2))
        ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 1, 0.2)))
    fig.patch.set_facecolor('#f3f3f3ff')  # Set the background color of the figure

    colors = [sns.color_palette("deep")[0], sns.color_palette("deep")[1], sns.color_palette("deep")[2], sns.color_palette("deep")[3], sns.color_palette("deep")[4]]
    colors = ["black", 
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("flare", as_cmap=True)(50),
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("crest", as_cmap=True)(50)]
    paired = sns.color_palette("Paired")
    colors = [paired[0], paired[2], paired[3], paired[4], paired[5]]

    sns.color_palette("flare", as_cmap=True)

    for i, data in enumerate(datas):
        if num_epochs is not None:
            data = data.head(num_epochs)
            data = data.tail(len(data)-epoch_start)
        ax.plot(data.rolling(window=100).mean(), label=referent_labels[i], color=sns.color_palette("husl", len(referent_labels))[i], linewidth=2, alpha=0.9)
        # ax.plot(, label=labels[i], color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.5)
        
        # else:
        #     marker_style = dict(
        #         marker=7,  # Change to preferred marker shape
        #         markersize=12,  # Marker size
        #         markerfacecolor="black",  # Marker face color
        #         markeredgecolor="black",  # Marker edge color
        #         markeredgewidth=1.5  # Marker edge width
        #     )

        #     ax.plot(data, color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.7, markevery=markers_on, **marker_style)

    # ax.set_title(f'Probe Entropy for Speaker Signals', fontsize=16)
    # ax.set_xlabel('Epoch', fontsize=16)
    # ax.set_ylabel('Entropy', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=15, loc=1)
    
    fig.tight_layout()
    uuidstr = str(uuid.uuid4())[:5]
    if agent_num:
        plt.savefig(os.path.join("../joint-plots/", f"inference_prs_for_listener_{agent_num}_all_referents_{uuidstr}.png"))    
    else:
        plt.savefig(os.path.join("../joint-plots/", f"inference_prs_for_all_referents_{uuidstr}.png"))

    config = {
        "directories": directory,
        "labels": referent_labels,
        "num_epochs": num_epochs,
    }

    with open(f'../joint-plots/configs/config_{uuidstr}.json', 'w') as f:
        json.dump(config, f)

    print(f'../joint-plots/configs/config_{uuidstr}.json')

def make_avg_pr_across_referents_plot(directorybunch, labels, referent_names, ref_nums=list(range(10)), num_epochs=None, epoch_start=0, markers_on=[], rolling_window=None, t_val=2.262, y_offsets=list(np.zeros(10))):
    entropies = []
    cis = []
    for ref_num in ref_nums:
        for directories in directorybunch:
            # directories = ["/users/bspiegel/signification-game/data_vis_base_experiment/"+d[2:] for d in directories]  # Useful for debug
            data_for_group = [pd.read_csv(os.path.join(directory, f"inference_pr_referent_{ref_num}.csv")) for directory in directories]
            merged_datas = pd.concat(data_for_group, axis=1, keys=range(len(data_for_group)))
            mean_entropy = merged_datas.mean(axis=1)
            ci_entropy = t_val * merged_datas.sem(axis=1)
            if rolling_window:
                mean_entropy = mean_entropy.rolling(window=rolling_window, center=True).mean()
                ci_entropy = ci_entropy.rolling(window=rolling_window, center=True).mean()
            entropies.append(mean_entropy)
            cis.append(ci_entropy)
    
    sns.set_theme(style="darkgrid")

    # Plot the data with larger font
    # plt.yscale("log", base=np.e)
    # plt.xscale("log")
    fig, ax = plt.subplots(figsize=(6, 5))
    # ax.set_xscale("log")
    fig.patch.set_facecolor('#f3f3f3ff')  # Set the background color of the figure

    colors = [sns.color_palette("deep")[0], sns.color_palette("deep")[1], sns.color_palette("deep")[2], sns.color_palette("deep")[3], sns.color_palette("deep")[4]]
    colors = ["black", 
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("flare", as_cmap=True)(50),
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("crest", as_cmap=True)(50)]
    paired = sns.color_palette("Paired")
    colors = [paired[0], paired[2], paired[3], paired[4], paired[5]]

    sns.color_palette("flare", as_cmap=True)

    for i, (entropy, ci) in enumerate(zip(entropies, cis)):
        if num_epochs is not None:
            entropy = entropy.head(num_epochs)
            entropy = entropy.tail(len(entropy)-epoch_start)
            ci = ci.head(num_epochs)
            ci = ci.tail(len(ci)-epoch_start)
        # if len(markers_on) > 0:
        #     marker_style = dict(
        #         marker=7,  # Change to preferred marker shape
        #         markersize=12,  # Marker size
        #         markerfacecolor="black",  # Marker face color
        #         markeredgecolor="black",  # Marker edge color
        #         markeredgewidth=1.5  # Marker edge width
        #     )

        #     ax.plot(data, color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.7, markevery=markers_on, **marker_style)

        ax.plot(entropy, label=f"{labels[i//len(ref_nums)]} referent {i}", color=sns.color_palette("tab10")[i % len(ref_nums)], linewidth=2, alpha=0.5)
        ax.fill_between(list(range(len(ci))), entropy-ci, entropy+ci, color=sns.color_palette("tab10")[i % len(ref_nums)], alpha=0.15)

        last_x = entropy.index[0]
        last_y = entropy.iloc[rolling_window // 2 + 1]
        # print(last_x)
        # print(last_y)

        ax.annotate(f'{referent_names[i]}: {last_y:.2f}',
                    xy=(last_x, last_y),
                    xytext=(-10, y_offsets[i]),  # 10 pixels to the left
                    textcoords='offset points',
                    va='center',
                    ha='right',
                    color=sns.color_palette("tab10")[i % len(ref_nums)],
                    annotation_clip=False,
                    fontsize=16)

    # ax.set_title(f'Probe Entropy for Speaker Signals', fontsize=16)
    # ax.set_xlabel('Epoch', fontsize=16)
    # ax.set_ylabel('Entropy', fontsize=16)
    ax.yaxis.tick_right()
    ax.tick_params(axis='both', which='major', labelsize=18)
    # ax.legend()
    # plt.legend(fontsize=16, loc='upper left')

    # Get current y limits and extend upper bound by 20%
    # ymin, ymax = ax.get_ylim()
    # ax.set_ylim(ymin, ymax * 1.2)
    
    fig.tight_layout()
    uuidstr = str(uuid.uuid4())[:5]
    plt.savefig(os.path.join("../joint-plots/", f"avg_inference_prs_across_referents_{uuidstr}.png"))

    config = {
        "directories": directories,
        "labels": labels,
        "num_epochs": num_epochs,
        "ref_nums": ref_nums
    }

    with open(f'../joint-plots/configs/config_{uuidstr}.json', 'w') as f:
        json.dump(config, f)

    print(f'../joint-plots/configs/config_{uuidstr}.json')


def make_reward_plot(directories, labels, num_epochs=None, epoch_start=0, markers_on=[]):
    datas = [pd.read_csv(os.path.join(directory, f"reward_for_speaker_images_all_listeners.csv")) for directory in directories]
    sns.set_theme(style="darkgrid")

    # Plot the data with larger font
    fig, ax = plt.subplots(figsize=(5, 2.25))
    fig.patch.set_facecolor('#f3f3f3ff')  # Set the background color of the figure

    colors = [sns.color_palette("deep")[0], sns.color_palette("deep")[1], sns.color_palette("deep")[2], sns.color_palette("deep")[3], sns.color_palette("deep")[4]]
    colors = ["black", 
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("flare", as_cmap=True)(50),
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("crest", as_cmap=True)(50)]
    paired = sns.color_palette("Paired")
    colors = [paired[0], paired[2], paired[3], paired[4], paired[5]]

    sns.color_palette("flare", as_cmap=True)

    for i, data in enumerate(datas):
        if num_epochs is not None:
            data = data.head(num_epochs)
            data = data.tail(len(data)-epoch_start)
        ax.plot(data, label=labels[i], color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.5)
        # ax.plot(data.rolling(window=100).mean(), label=labels[i], color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.5)
        
        # else:
        #     marker_style = dict(
        #         marker=7,  # Change to preferred marker shape
        #         markersize=12,  # Marker size
        #         markerfacecolor="black",  # Marker face color
        #         markeredgecolor="black",  # Marker edge color
        #         markeredgewidth=1.5  # Marker edge width
        #     )

        #     ax.plot(data, color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.7, markevery=markers_on, **marker_style)

    # ax.set_title(f'Probe Entropy for Speaker Signals', fontsize=16)
    # ax.set_xlabel('Epoch', fontsize=16)
    # ax.set_ylabel('Entropy', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(fontsize=15)
    
    fig.tight_layout()
    uuidstr = str(uuid.uuid4())[:5]
    plt.savefig(os.path.join("../joint-plots/", f"reward_for_speaker_images_all_listeners_{uuidstr}.png"))

    config = {
        "directories": directories,
        "labels": labels,
        "num_epochs": num_epochs,
    }

    with open(f'../joint-plots/configs/config_{uuidstr}.json', 'w') as f:
        json.dump(config, f)

    print(f'../joint-plots/configs/config_{uuidstr}.json')

def make_com_success_plot(directories, labels, num_epochs=None, epoch_start=0, markers_on=[]):
    datas = [pd.read_csv(os.path.join(directory, f"success_rate_all_referents.csv")) for directory in directories]
    sns.set_theme(style="darkgrid")

    # Plot the data with larger font
    fig, ax = plt.subplots(figsize=(5, 2.25))
    fig.patch.set_facecolor('#f3f3f3ff')  # Set the background color of the figure

    colors = [sns.color_palette("deep")[0], sns.color_palette("deep")[1], sns.color_palette("deep")[2], sns.color_palette("deep")[3], sns.color_palette("deep")[4]]
    colors = ["black", 
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("flare", as_cmap=True)(50),
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("crest", as_cmap=True)(50)]
    paired = sns.color_palette("Paired")
    colors = [paired[0], paired[2], paired[3], paired[4], paired[5]]

    sns.color_palette("flare", as_cmap=True)

    for i, data in enumerate(datas):
        if num_epochs is not None:
            data = data.head(num_epochs)
            data = data.tail(len(data)-epoch_start)
        ax.plot(data, label=labels[i], color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.5)
        # ax.plot(data.rolling(window=100).mean(), label=labels[i], color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.5)
        
        # else:
        #     marker_style = dict(
        #         marker=7,  # Change to preferred marker shape
        #         markersize=12,  # Marker size
        #         markerfacecolor="black",  # Marker face color
        #         markeredgecolor="black",  # Marker edge color
        #         markeredgewidth=1.5  # Marker edge width
        #     )

        #     ax.plot(data, color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.7, markevery=markers_on, **marker_style)

    # ax.set_title(f'Probe Entropy for Speaker Signals', fontsize=16)
    # ax.set_xlabel('Epoch', fontsize=16)
    # ax.set_ylabel('Entropy', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(fontsize=15)
    
    fig.tight_layout()
    uuidstr = str(uuid.uuid4())[:5]
    plt.savefig(os.path.join("../joint-plots/", f"success_rate_all_referents_{uuidstr}.png"))

    config = {
        "directories": directories,
        "labels": labels,
        "num_epochs": num_epochs,
    }

    with open(f'../joint-plots/configs/config_{uuidstr}.json', 'w') as f:
        json.dump(config, f)

    print(f'../joint-plots/configs/config_{uuidstr}.json')

def make_avg_com_success_plot(directorybunch, labels, ref_num=0, all_speakers_avg=False, num_epochs=None, epoch_start=0, markers_on=[], rolling_window=None, t_val=2.262):
    entropies = []
    cis = []
    for directories in directorybunch:
        # directories = ["/users/bspiegel/signification-game/data_vis_base_experiment/"+d[2:] for d in directories]  # Useful for debug
        data_for_group = [pd.read_csv(os.path.join(directory, f"success_rate_referent_{ref_num}.csv" if not all_speakers_avg else "success_rate_all_referents.csv")) for directory in directories]
        merged_datas = pd.concat(data_for_group, axis=1, keys=range(len(data_for_group)))
        mean_entropy = merged_datas.mean(axis=1)
        ci_entropy = t_val * merged_datas.sem(axis=1)
        if rolling_window:
            mean_entropy = mean_entropy.rolling(window=rolling_window, center=True).mean()
            ci_entropy = ci_entropy.rolling(window=rolling_window, center=True).mean()
        entropies.append(mean_entropy)
        cis.append(ci_entropy)
    
    sns.set_theme(style="darkgrid")

    # Plot the data with larger font
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor('#f3f3f3ff')  # Set the background color of the figure

    colors = [sns.color_palette("deep")[0], sns.color_palette("deep")[1], sns.color_palette("deep")[2], sns.color_palette("deep")[3], sns.color_palette("deep")[4]]
    colors = ["black", 
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("flare", as_cmap=True)(50),
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("crest", as_cmap=True)(50)]
    paired = sns.color_palette("Paired")
    colors = [paired[0], paired[2], paired[3], paired[4], paired[5]]

    sns.color_palette("flare", as_cmap=True)

    for i, (entropy, ci) in enumerate(zip(entropies, cis)):
        if num_epochs is not None:
            entropy = entropy.head(num_epochs)
            entropy = entropy.tail(len(entropy)-epoch_start)
            ci = ci.head(num_epochs)
            ci = ci.tail(len(ci)-epoch_start)
        # if len(markers_on) > 0:
        #     marker_style = dict(
        #         marker=7,  # Change to preferred marker shape
        #         markersize=12,  # Marker size
        #         markerfacecolor="black",  # Marker face color
        #         markeredgecolor="black",  # Marker edge color
        #         markeredgewidth=1.5  # Marker edge width
        #     )

        #     ax.plot(data, color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.7, markevery=markers_on, **marker_style)

        ax.plot(entropy, label=labels[i], color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.5)
        ax.fill_between(list(range(len(ci))), entropy-ci, entropy+ci, color=sns.color_palette("Set1")[i], alpha=0.15)

    # ax.set_title(f'Probe Entropy for Speaker Signals', fontsize=16)
    # ax.set_xlabel('Epoch', fontsize=16)
    # ax.set_ylabel('Entropy', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(fontsize=15, loc='lower right')

    # Get current y limits and extend upper bound by 20%
    ymin, ymax = ax.get_ylim()
    # ax.set_ylim(ymin, 1.0)
    
    fig.tight_layout()
    uuidstr = str(uuid.uuid4())[:5]
    plt.savefig(os.path.join("../joint-plots/", f"avg_success_rate_referent_{ref_num}_{uuidstr}.png" if not all_speakers_avg else f"avg_success_rate_all_referents_{uuidstr}.png"))

    config = {
        "directories": directories,
        "labels": labels,
        "num_epochs": num_epochs,
        "ref_num": ref_num
    }

    with open(f'../joint-plots/configs/config_{uuidstr}.json', 'w') as f:
        json.dump(config, f)

    print(f"avg_success_rate_referent_{ref_num}_{uuidstr}.png" if not all_speakers_avg else f"avg_success_rate_all_referents_{uuidstr}.png")

def make_avg_com_success_across_referents_plot(directorybunch, labels, referent_names, ref_nums=list(range(10)), num_epochs=None, epoch_start=0, markers_on=[], rolling_window=None, t_val=2.262, y_offsets=list(np.zeros(10))):
    entropies = []
    cis = []
    for ref_num in ref_nums:
        for directories in directorybunch:
            # directories = ["/users/bspiegel/signification-game/data_vis_base_experiment/"+d[2:] for d in directories]  # Useful for debug
            data_for_group = [pd.read_csv(os.path.join(directory, f"success_rate_referent_{ref_num}.csv")) for directory in directories]
            merged_datas = pd.concat(data_for_group, axis=1, keys=range(len(data_for_group)))
            mean_entropy = merged_datas.mean(axis=1)
            ci_entropy = t_val * merged_datas.sem(axis=1)
            if rolling_window:
                mean_entropy = mean_entropy.rolling(window=rolling_window, center=True).mean()
                ci_entropy = ci_entropy.rolling(window=rolling_window, center=True).mean()
            entropies.append(mean_entropy)
            cis.append(ci_entropy)
    
    sns.set_theme(style="darkgrid")

    # Plot the data with larger font
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#f3f3f3ff')  # Set the background color of the figure

    colors = [sns.color_palette("deep")[0], sns.color_palette("deep")[1], sns.color_palette("deep")[2], sns.color_palette("deep")[3], sns.color_palette("deep")[4]]
    colors = ["black", 
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("flare", as_cmap=True)(50),
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("crest", as_cmap=True)(50)]
    paired = sns.color_palette("Paired")
    colors = [paired[0], paired[2], paired[3], paired[4], paired[5]]

    sns.color_palette("flare", as_cmap=True)

    for i, (entropy, ci) in enumerate(zip(entropies, cis)):
        if num_epochs is not None:
            entropy = entropy.head(num_epochs)
            entropy = entropy.tail(len(entropy)-epoch_start)
            ci = ci.head(num_epochs)
            ci = ci.tail(len(ci)-epoch_start)
        # if len(markers_on) > 0:
        #     marker_style = dict(
        #         marker=7,  # Change to preferred marker shape
        #         markersize=12,  # Marker size
        #         markerfacecolor="black",  # Marker face color
        #         markeredgecolor="black",  # Marker edge color
        #         markeredgewidth=1.5  # Marker edge width
        #     )

        #     ax.plot(data, color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.7, markevery=markers_on, **marker_style)

        ax.plot(entropy, label=f"{labels[i//len(ref_nums)]} referent {i}", color=sns.color_palette("tab10")[i % len(ref_nums)], linewidth=2, alpha=0.5)
        ax.fill_between(list(range(len(ci))), entropy-ci, entropy+ci, color=sns.color_palette("tab10")[i % len(ref_nums)], alpha=0.15)

        last_x = entropy.index[-1]
        last_y = entropy.iloc[len(entropy) - 1 - rolling_window // 2]
        # print(last_x)
        # print(last_y)
        ax.annotate(f'{referent_names[i]}: {last_y:.2f}',
                    xy=(last_x, last_y),
                    xytext=(10, y_offsets[i]),  # 10 pixels to the right
                    textcoords='offset points',
                    va='center',
                    ha='left',
                    color=sns.color_palette("tab10")[i % len(ref_nums)],
                    annotation_clip=False,
                    fontsize=16)

    # ax.set_title(f'Probe Entropy for Speaker Signals', fontsize=16)
    # ax.set_xlabel('Epoch', fontsize=16)
    # ax.set_ylabel('Entropy', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # ax.legend()
    # plt.legend(fontsize=16, loc='upper left')

    # Get current y limits and extend upper bound by 20%
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)
    
    fig.tight_layout()
    uuidstr = str(uuid.uuid4())[:5]
    plt.savefig(os.path.join("../joint-plots/", f"avg_success_rate_across_referents_{uuidstr}.png"))

    config = {
        "directories": directories,
        "labels": labels,
        "num_epochs": num_epochs,
        "ref_nums": ref_nums
    }

    with open(f'../joint-plots/configs/config_{uuidstr}.json', 'w') as f:
        json.dump(config, f)

    print(f"avg_success_rate_across_referents_{uuidstr}.png")

def make_probe_plot(directories, labels, sp_num=0, all_speakers_avg=False, num_epochs=None, epoch_start=0, markers_on=[]):
    datas = [pd.read_csv(os.path.join(directory, f"probe_entropy_speaker_{sp_num}.csv" if not all_speakers_avg else "probe_entropy_all_speakers.csv")) for directory in directories]
    sns.set_theme(style="darkgrid")

    # Plot the data with larger font
    fig, ax = plt.subplots(figsize=(5, 2))
    fig.patch.set_facecolor('#f3f3f3ff')  # Set the background color of the figure

    colors = [sns.color_palette("deep")[0], sns.color_palette("deep")[1], sns.color_palette("deep")[2], sns.color_palette("deep")[3], sns.color_palette("deep")[4]]
    colors = ["black", 
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("flare", as_cmap=True)(50),
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("crest", as_cmap=True)(50)]
    paired = sns.color_palette("Paired")
    colors = [paired[0], paired[2], paired[3], paired[4], paired[5]]

    sns.color_palette("flare", as_cmap=True)

    for i, data in enumerate(datas):
        if num_epochs is not None:
            data = data.head(num_epochs)
            data = data.tail(len(data)-epoch_start)
        if len(markers_on) > 0:
            marker_style = dict(
                marker=7,  # Change to preferred marker shape
                markersize=12,  # Marker size
                markerfacecolor="black",  # Marker face color
                markeredgecolor="black",  # Marker edge color
                markeredgewidth=1.5  # Marker edge width
            )

            ax.plot(data, color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.7, markevery=markers_on, **marker_style)
        else:
            # ax.plot(data, color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.15)
            ax.plot(data.rolling(window=50).mean(), label=labels[i], color=sns.color_palette("Set1")[i], linewidth=4, alpha=0.5)
            # ax.plot(data, label=labels[i], color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.5)

    # ax.set_title(f'Probe Entropy for Speaker Signals', fontsize=16)
    # ax.set_xlabel('Epoch', fontsize=16)
    # ax.set_ylabel('Entropy', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # plt.legend(fontsize=16)
    
    fig.tight_layout()
    uuidstr = str(uuid.uuid4())[:5]
    plt.savefig(os.path.join("../joint-plots/", f"probe_entropy_speaker_{sp_num}_{uuidstr}.png" if not all_speakers_avg else f"probe_entropy_all_speakers_{uuidstr}.png"))

    config = {
        "directories": directories,
        "labels": labels,
        "num_epochs": num_epochs,
        "sp_num": sp_num
    }

    with open(f'../joint-plots/configs/config_{uuidstr}.json', 'w') as f:
        json.dump(config, f)

    print(f'../joint-plots/configs/config_{uuidstr}.json')

def make_avg_probe_plot(directorybunch, labels, sp_num=0, all_speakers_avg=False, num_epochs=None, epoch_start=0, markers_on=[], rolling_window=None, t_val=2.262):
    entropies = []
    cis = []
    for directories in directorybunch:
        # directories = ["/users/bspiegel/signification-game/data_vis_base_experiment/"+d[2:] for d in directories]  # Useful for debug
        data_for_group = [pd.read_csv(os.path.join(directory, f"probe_entropy_speaker_{sp_num}.csv" if not all_speakers_avg else "probe_entropy_all_speakers.csv")) for directory in directories]
        merged_datas = pd.concat(data_for_group, axis=1, keys=range(len(data_for_group)))
        mean_entropy = merged_datas.mean(axis=1)
        ci_entropy = t_val * merged_datas.sem(axis=1)
        if rolling_window:
            mean_entropy = mean_entropy.rolling(window=rolling_window, center=True).mean()
            ci_entropy = ci_entropy.rolling(window=rolling_window, center=True).mean()
        entropies.append(mean_entropy)
        cis.append(ci_entropy)
    
    sns.set_theme(style="darkgrid")

    # Plot the data with larger font
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor('#f3f3f3ff')  # Set the background color of the figure

    colors = [sns.color_palette("deep")[0], sns.color_palette("deep")[1], sns.color_palette("deep")[2], sns.color_palette("deep")[3], sns.color_palette("deep")[4]]
    colors = ["black", 
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("flare", as_cmap=True)(50),
              sns.color_palette("flare", as_cmap=True)(100), sns.color_palette("crest", as_cmap=True)(50)]
    paired = sns.color_palette("Paired")
    colors = [paired[0], paired[2], paired[3], paired[4], paired[5]]

    sns.color_palette("flare", as_cmap=True)

    for i, (entropy, ci) in enumerate(zip(entropies, cis)):
        if num_epochs is not None:
            entropy = entropy.head(num_epochs)
            entropy = entropy.tail(len(entropy)-epoch_start)
            ci = ci.head(num_epochs)
            ci = ci.tail(len(ci)-epoch_start)
        # if len(markers_on) > 0:
        #     marker_style = dict(
        #         marker=7,  # Change to preferred marker shape
        #         markersize=12,  # Marker size
        #         markerfacecolor="black",  # Marker face color
        #         markeredgecolor="black",  # Marker edge color
        #         markeredgewidth=1.5  # Marker edge width
        #     )

        #     ax.plot(data, color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.7, markevery=markers_on, **marker_style)

        ax.plot(entropy, label=labels[i], color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.5)
        ax.fill_between(list(range(len(ci))), entropy-ci, entropy+ci, color=sns.color_palette("Set1")[i], alpha=0.15)

    # ax.set_title(f'Probe Entropy for Speaker Signals', fontsize=16)
    # ax.set_xlabel('Epoch', fontsize=16)
    # ax.set_ylabel('Entropy', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(fontsize=13, loc='upper right', handleheight=0.3)

    # Get current y limits and extend upper bound by 20%
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.75)
    
    fig.tight_layout()
    uuidstr = str(uuid.uuid4())[:5]
    plt.savefig(os.path.join("../joint-plots/", f"probe_entropy_speaker_{sp_num}_{uuidstr}.png" if not all_speakers_avg else f"probe_entropy_all_speakers_{uuidstr}.png"))

    config = {
        "directories": directories,
        "labels": labels,
        "num_epochs": num_epochs,
        "sp_num": sp_num
    }

    with open(f'../joint-plots/configs/config_{uuidstr}.json', 'w') as f:
        json.dump(config, f)

    print(f"probe_entropy_speaker_{sp_num}_{uuidstr}.png" if not all_speakers_avg else f"probe_entropy_all_speakers_{uuidstr}.png")

# def make_value_grid_plot(dataframe, metric="spline_wasserstein_distances_mean", epoch=1000):
#     # I need to grab the summary metrics at a specific epoch and add them to the dataframe

#     # load data from directory and add it to dataframe

#     metric_data = []

#     for run_info in dataframe.values.tolist():

#     # Then plot them on a grid

#     vegetables = None
#     farmers = None

#     fig, ax = plt.subplots()
#     im = ax.imshow(harvest)

#     # Show all ticks and label them with the respective list entries
#     ax.set_xticks(range(len(farmers)), labels=farmers,
#                 rotation=45, ha="right", rotation_mode="anchor")
#     ax.set_yticks(range(len(vegetables)), labels=vegetables)

#     # Loop over data dimensions and create text annotations.
#     for i in range(len(vegetables)):
#         for j in range(len(farmers)):
#             text = ax.text(j, i, harvest[i, j],
#                         ha="center", va="center", color="w")

#     ax.set_title("Harvest of local farmers (in tons/year)")
#     fig.tight_layout()
#     uuidstr = str(uuid.uuid4())[:5]
#     plt.savefig(os.path.join("../sweep-plots/", f"gridplot_{metric}_epoch_{epoch}_{uuidstr}.png"))

def make_graphics_post_conference():
    # download_reward_data(run_id="signification-team/signification-game/desfenmt", directory="./tough-cloud-2359/")
    # download_probe_data(run_id="signification-team/signification-game/ni2dajf2", directory="./glad-dew-2358/")
    # download_reward_data(run_id="signification-team/signification-game/ni2dajf2", directory="./glad-dew-2358/")

    # make_animation(directory="./dazzling-puddle-2413/", label="Inferential - No P_ref", speaker_selection=[12, 8, 12, 2, 2, 12, 0, 14, 2, 12])

    # make_animation(directory="./glad-dew-2358/", label="Inferential - Curve Penalty", speaker_selection=[12, 8, 12, 2, 2, 12, 0, 14, 2, 12])
    # make_animation(directory="./frosty-silence-2354/", label="Inferential - Size Penalty", speaker_selection=[12, 8, 12, 2, 2, 12, 0, 14, 2, 12])
    # make_simple_animation(directory="./glad-dew-2358/", label="Inferential - Curve Penalty", speaker_selection=[12, 8, 12, 2, 2, 12, 0, 14, 2, 12])
    # make_simple_animation(directory="./frosty-silence-2354/", label="Inferential - Size Penalty", speaker_selection=[12, 8, 12, 2, 2, 12, 0, 14, 2, 12])

    # make_multi_animation(directories=("./dazzling-meadow-2352/", "./frosty-silence-2354/"), labels=("Behavioral", "Inferential (ours)"), speaker_selection=[12, 8, 12, 2, 2, 12, 0, 14, 2, 12])
    # frosty-silence-2354

    ### 10 runs of behaviorist signaling with learning listeners - canvas 0.2
    behaviorist_live_listeners_runs = ["./rebel-commander-2489/",
                                        "./galactic-wars-2488/",
                                        "./civilized-senate-2487/",
                                        "./grievous-lightsaber-2486/",
                                        "./clone-tauntaun-2485/",
                                        "./holographic-ewok-2483/",
                                        "./hokey-speeder-2484/",
                                        "./light-fleet-2482/",
                                        "./elegant-jawa-2481/",
                                        "./tusken-xwing-2480/"]
    # download_communication_success_data(run_id="signification-team/signification-game/y7j9deuh", directory="./rebel-commander-2489/")
    # download_communication_success_data(run_id="signification-team/signification-game/v0oq5oam", directory="./galactic-wars-2488/")
    # download_communication_success_data(run_id="signification-team/signification-game/868c3dyx", directory="./civilized-senate-2487/")
    # download_communication_success_data(run_id="signification-team/signification-game/98cwvh50", directory="./grievous-lightsaber-2486/")
    # download_communication_success_data(run_id="signification-team/signification-game/9eo8wgbw", directory="./clone-tauntaun-2485/")
    # download_communication_success_data(run_id="signification-team/signification-game/eklqwb39", directory="./holographic-ewok-2483/")
    # download_communication_success_data(run_id="signification-team/signification-game/lqgebg27", directory="./hokey-speeder-2484/")
    # download_communication_success_data(run_id="signification-team/signification-game/r5o8a7b4", directory="./light-fleet-2482/")
    # download_communication_success_data(run_id="signification-team/signification-game/jnk99r1l", directory="./elegant-jawa-2481/")
    # download_communication_success_data(run_id="signification-team/signification-game/vhpgwje5", directory="./tusken-xwing-2480/")

    ### 10 runs of behaviorist signaling with dead listeners - canvas 0.2
    behaviorist_dead_listeners_runs = ["./galactic-parsec-2479/",
                                        "./star-nexu-2478/",
                                        "./mythical-transport-2475/",
                                        "./carbonite-womprat-2477/",
                                        "./ancient-admiral-2475/",
                                        "./legendary-bothan-2474/",
                                        "./star-master-2473/",
                                        "./ancient-tie-fighter-2472/",
                                        "./light-lightsaber-2471/",
                                        "./legendary-fleet-2470/"]
    # download_communication_success_data(run_id="signification-team/signification-game/iiaca3fq", directory="./galactic-parsec-2479/")
    # download_communication_success_data(run_id="signification-team/signification-game/6onfwmva", directory="./star-nexu-2478/")
    # download_communication_success_data(run_id="signification-team/signification-game/45p6kcc7", directory="./mythical-transport-2475/")
    # download_communication_success_data(run_id="signification-team/signification-game/8xsirdyn", directory="./carbonite-womprat-2477/")
    # download_communication_success_data(run_id="signification-team/signification-game/hs4kyj2p", directory="./ancient-admiral-2475/")
    # download_communication_success_data(run_id="signification-team/signification-game/qroyfszz", directory="./legendary-bothan-2474/")
    # download_communication_success_data(run_id="signification-team/signification-game/6n03k6b8", directory="./star-master-2473/")
    # download_communication_success_data(run_id="signification-team/signification-game/olfmgnvp", directory="./ancient-tie-fighter-2472/")
    # download_communication_success_data(run_id="signification-team/signification-game/1blbrljs", directory="./light-lightsaber-2471/")
    # download_communication_success_data(run_id="signification-team/signification-game/pd6jy740", directory="./legendary-fleet-2470/")


    ### 10 runs of inferential signaling, no penalties - canvas 0.2
    inferential_no_penalty_runs = ["./elegant-admiral-2499/",
                                    "./legendary-commander-2498/",
                                    "./imperial-senate-2497/",
                                    "./old-tauntaun-2496/",
                                    "./carbonite-astromech-2495/",
                                    "./tusken-tie-fighter-2494/",
                                    "./scruffy-looking-federation-2493/",
                                    "./dark-cantina-2492-2472/",
                                    "./old-nerf-herder-2490/",
                                    "./scruffy-looking-cantina-2490/"]
    # download_communication_success_data(run_id="signification-team/signification-game/bt0q4jzl", directory="./elegant-admiral-2499/")
    # download_communication_success_data(run_id="signification-team/signification-game/a1v8p68r", directory="./legendary-commander-2498/")
    # download_communication_success_data(run_id="signification-team/signification-game/jpc5mdyq", directory="./imperial-senate-2497/")
    # download_communication_success_data(run_id="signification-team/signification-game/re7rmsli", directory="./old-tauntaun-2496/")
    # download_communication_success_data(run_id="signification-team/signification-game/cfo76mkt", directory="./carbonite-astromech-2495/")
    # download_communication_success_data(run_id="signification-team/signification-game/ajoth1qz", directory="./tusken-tie-fighter-2494/")
    # download_communication_success_data(run_id="signification-team/signification-game/iu4zqx64", directory="./scruffy-looking-federation-2493/")
    # download_communication_success_data(run_id="signification-team/signification-game/9ljsehh6", directory="./dark-cantina-2492-2472/")
    # download_communication_success_data(run_id="signification-team/signification-game/8a606df1", directory="./old-nerf-herder-2490/")
    # download_communication_success_data(run_id="signification-team/signification-game/gb21hgmt", directory="./scruffy-looking-cantina-2490/")

    ### 10 runs of inferential signaling, no penalties, ablated Pr - canvas 0.2
    inferential_no_penalty_runs_ablated_Pr = ["./imperial-fleet-2509/",
                                                "./civilized-destroyer-2508/",
                                                "./scruffy-looking-bothan-2507/",
                                                "./dark-parsec-2506/",
                                                "./sith-trooper-2505/",
                                                "./imperial-bantha-2504/",
                                                "./jedi-tie-fighter-2503/",
                                                "./clone-commander-2502/",
                                                "./stellar-wookie-2501/",
                                                "./grievous-federation-2500/"]
    # download_communication_success_data(run_id="signification-team/signification-game/2av1uafm", directory="./imperial-fleet-2509/")
    # download_communication_success_data(run_id="signification-team/signification-game/zktrb6u0", directory="./civilized-destroyer-2508/")
    # download_communication_success_data(run_id="signification-team/signification-game/0hfwxcpx", directory="./scruffy-looking-bothan-2507/")
    # download_communication_success_data(run_id="signification-team/signification-game/ur8zsdoz", directory="./dark-parsec-2506/")
    # download_communication_success_data(run_id="signification-team/signification-game/5xs4xgfs", directory="./sith-trooper-2505/")
    # download_communication_success_data(run_id="signification-team/signification-game/ujg58y2l", directory="./imperial-bantha-2504/")
    # download_communication_success_data(run_id="signification-team/signification-game/pk1atmym", directory="./jedi-tie-fighter-2503/")
    # download_communication_success_data(run_id="signification-team/signification-game/2v3b0xon", directory="./clone-commander-2502/")
    # download_communication_success_data(run_id="signification-team/signification-game/xqd4ts0g", directory="./stellar-wookie-2501/")
    # download_communication_success_data(run_id="signification-team/signification-game/w5jjgghi", directory="./grievous-federation-2500/")

    # make_avg_com_success_plot([behaviorist_live_listeners_runs,
    #                     behaviorist_dead_listeners_runs,
    #                     inferential_no_penalty_runs,
    #                     inferential_no_penalty_runs_ablated_Pr],
    #                     ["Behaviorist",
    #                     "Behaviorist (Canalized)",
    #                     "Inferential",
    #                     "Inferential - P_ref"],
    #                     all_speakers_avg=True,
    #                     rolling_window=25, t_val=1.833)
    
    # make_avg_com_success_across_referents_plot([
    #                     behaviorist_dead_listeners_runs,],
    #                     [
    #                     "Behaviorist - Canalized"],
    #                     rolling_window=25, t_val=1.833)


    #######################################################
    ################## Canvas 0.1 below ###################
    #######################################################

    ### 10 runs of behaviorist signaling with learning listeners - canvas 0.1
    behaviorist_live_listeners_runs = ["./ethereal-butterfly-2559/",
                                        "./glorious-sea-2558/",
                                        "./lunar-forest-2557/",
                                        "./ancient-meadow-2556/",
                                        "./silvery-oath-2555/",
                                        "./youthful-morning-2554/",
                                        "./deep-galaxy-2553/",
                                        "./dry-donkey-2552/",
                                        "./copper-capybara-2551/",
                                        "./denim-dawn-2550/"]
    # download_communication_success_data(run_id="signification-team/signification-game/ea55y73e", directory="./ethereal-butterfly-2559/")
    # download_communication_success_data(run_id="signification-team/signification-game/q3wt7y7c", directory="./glorious-sea-2558/")
    # download_communication_success_data(run_id="signification-team/signification-game/bml99nmq", directory="./lunar-forest-2557/")
    # download_communication_success_data(run_id="signification-team/signification-game/hc9r6xxu", directory="./ancient-meadow-2556/")
    # download_communication_success_data(run_id="signification-team/signification-game/fxocf59g", directory="./silvery-oath-2555/")
    # download_communication_success_data(run_id="signification-team/signification-game/49hznmxu", directory="./youthful-morning-2554/")
    # download_communication_success_data(run_id="signification-team/signification-game/9tfe32h3", directory="./deep-galaxy-2553/")
    # download_communication_success_data(run_id="signification-team/signification-game/qj06v9n1", directory="./dry-donkey-2552/")
    # download_communication_success_data(run_id="signification-team/signification-game/szlcbp8n", directory="./copper-capybara-2551/")
    # download_communication_success_data(run_id="signification-team/signification-game/atqxv560", directory="./denim-dawn-2550/")
    # download_probe_data(run_id="signification-team/signification-game/ea55y73e", directory="./ethereal-butterfly-2559/")
    # download_probe_data(run_id="signification-team/signification-game/q3wt7y7c", directory="./glorious-sea-2558/")
    # download_probe_data(run_id="signification-team/signification-game/bml99nmq", directory="./lunar-forest-2557/")
    # download_probe_data(run_id="signification-team/signification-game/hc9r6xxu", directory="./ancient-meadow-2556/")
    # download_probe_data(run_id="signification-team/signification-game/fxocf59g", directory="./silvery-oath-2555/")
    # download_probe_data(run_id="signification-team/signification-game/49hznmxu", directory="./youthful-morning-2554/")
    # download_probe_data(run_id="signification-team/signification-game/9tfe32h3", directory="./deep-galaxy-2553/")
    # download_probe_data(run_id="signification-team/signification-game/qj06v9n1", directory="./dry-donkey-2552/")
    # download_probe_data(run_id="signification-team/signification-game/szlcbp8n", directory="./copper-capybara-2551/")
    # download_probe_data(run_id="signification-team/signification-game/atqxv560", directory="./denim-dawn-2550/")
    # download_speaker_examples(run_id="signification-team/signification-game/ea55y73e", directory="./ethereal-butterfly-2559/")

    ### 10 runs of behaviorist signaling with dead listeners - canvas 0.1
    behaviorist_dead_listeners_runs = ["./still-pond-2549/",
                                        "./pretty-firebrand-2548/",
                                        "./divine-snowflake-2547/",
                                        "./lively-cloud-2546/",
                                        "./lilac-frog-2545/",
                                        "./woven-galaxy-2534/",
                                        "./colorful-blaze-2533/",
                                        "./autumn-cosmos-2530/",
                                        "./lilac-music-2529/",
                                        "./zesty-dust-2528/"]
    # download_communication_success_data(run_id="signification-team/signification-game/3bjaz4re", directory="./still-pond-2549/")
    # download_communication_success_data(run_id="signification-team/signification-game/4it1pfsc", directory="./pretty-firebrand-2548/")
    # download_communication_success_data(run_id="signification-team/signification-game/508e2i0e", directory="./divine-snowflake-2547/")
    # download_communication_success_data(run_id="signification-team/signification-game/6lm3h4x1", directory="./lively-cloud-2546/")
    # download_communication_success_data(run_id="signification-team/signification-game/ji0vg89o", directory="./lilac-frog-2545/")
    # download_communication_success_data(run_id="signification-team/signification-game/4hxwrpiw", directory="./woven-galaxy-2534/")
    # download_communication_success_data(run_id="signification-team/signification-game/0t3ev7nx", directory="./colorful-blaze-2533/")
    # download_communication_success_data(run_id="signification-team/signification-game/lwt9wjtl", directory="./autumn-cosmos-2530/")
    # download_communication_success_data(run_id="signification-team/signification-game/pqz71686", directory="./lilac-music-2529/")
    # download_communication_success_data(run_id="signification-team/signification-game/9csiykvn", directory="./zesty-dust-2528/")
    # download_probe_data(run_id="signification-team/signification-game/3bjaz4re", directory="./still-pond-2549/")
    # download_probe_data(run_id="signification-team/signification-game/4it1pfsc", directory="./pretty-firebrand-2548/")
    # download_probe_data(run_id="signification-team/signification-game/508e2i0e", directory="./divine-snowflake-2547/")
    # download_probe_data(run_id="signification-team/signification-game/6lm3h4x1", directory="./lively-cloud-2546/")
    # download_probe_data(run_id="signification-team/signification-game/ji0vg89o", directory="./lilac-frog-2545/")
    # download_probe_data(run_id="signification-team/signification-game/4hxwrpiw", directory="./woven-galaxy-2534/")
    # download_probe_data(run_id="signification-team/signification-game/0t3ev7nx", directory="./colorful-blaze-2533/")
    # download_probe_data(run_id="signification-team/signification-game/lwt9wjtl", directory="./autumn-cosmos-2530/")
    # download_probe_data(run_id="signification-team/signification-game/pqz71686", directory="./lilac-music-2529/")
    # download_probe_data(run_id="signification-team/signification-game/9csiykvn", directory="./zesty-dust-2528/")
    # download_speaker_examples(run_id="signification-team/signification-game/3bjaz4re", directory="./still-pond-2549/")

    ### 10 runs of inferential signaling, no penalties - canvas 0.1
    inferential_no_penalty_runs = ["./fancy-monkey-2544/",
                                    "./lucky-sky-2543/",
                                    "./royal-blaze-2542/",
                                    "./visionary-butterfly-2541/",
                                    "./smart-resonance-2540/",
                                    "./cosmic-thunder-2539/",
                                    "./smooth-dew-2538/",
                                    "./dainty-waterfall-2537/",
                                    "./youthful-star-2536/",
                                    "./dutiful-planet-2535/"]
    # download_communication_success_data(run_id="signification-team/signification-game/wet3y5g6", directory="./fancy-monkey-2544/")
    # download_communication_success_data(run_id="signification-team/signification-game/d5hqyufc", directory="./lucky-sky-2543/")
    # download_communication_success_data(run_id="signification-team/signification-game/xyceym5k", directory="./royal-blaze-2542/")
    # download_communication_success_data(run_id="signification-team/signification-game/al94zbi7", directory="./visionary-butterfly-2541/")
    # download_communication_success_data(run_id="signification-team/signification-game/mjylwu94", directory="./smart-resonance-2540/")
    # download_communication_success_data(run_id="signification-team/signification-game/h6h7nq1p", directory="./cosmic-thunder-2539/")
    # download_communication_success_data(run_id="signification-team/signification-game/5taxgi5l", directory="./smooth-dew-2538/")
    # download_communication_success_data(run_id="signification-team/signification-game/g8sgtojm", directory="./dainty-waterfall-2537/")
    # download_communication_success_data(run_id="signification-team/signification-game/cll0cq9m", directory="./youthful-star-2536/")
    # download_communication_success_data(run_id="signification-team/signification-game/sgk40864", directory="./dutiful-planet-2535/")
    # download_pr_data(run_id="signification-team/signification-game/wet3y5g6", directory="./fancy-monkey-2544/")
    # download_pr_data(run_id="signification-team/signification-game/d5hqyufc", directory="./lucky-sky-2543/")
    # download_pr_data(run_id="signification-team/signification-game/xyceym5k", directory="./royal-blaze-2542/")
    # download_pr_data(run_id="signification-team/signification-game/al94zbi7", directory="./visionary-butterfly-2541/")
    # download_pr_data(run_id="signification-team/signification-game/mjylwu94", directory="./smart-resonance-2540/")
    # download_pr_data(run_id="signification-team/signification-game/h6h7nq1p", directory="./cosmic-thunder-2539/")
    # download_pr_data(run_id="signification-team/signification-game/5taxgi5l", directory="./smooth-dew-2538/")
    # download_pr_data(run_id="signification-team/signification-game/g8sgtojm", directory="./dainty-waterfall-2537/")
    # download_pr_data(run_id="signification-team/signification-game/cll0cq9m", directory="./youthful-star-2536/")
    # download_pr_data(run_id="signification-team/signification-game/sgk40864", directory="./dutiful-planet-2535/")
    # download_probe_data(run_id="signification-team/signification-game/wet3y5g6", directory="./fancy-monkey-2544/")
    # download_probe_data(run_id="signification-team/signification-game/d5hqyufc", directory="./lucky-sky-2543/")
    # download_probe_data(run_id="signification-team/signification-game/xyceym5k", directory="./royal-blaze-2542/")
    # download_probe_data(run_id="signification-team/signification-game/al94zbi7", directory="./visionary-butterfly-2541/")
    # download_probe_data(run_id="signification-team/signification-game/mjylwu94", directory="./smart-resonance-2540/")
    # download_probe_data(run_id="signification-team/signification-game/h6h7nq1p", directory="./cosmic-thunder-2539/")
    # download_probe_data(run_id="signification-team/signification-game/5taxgi5l", directory="./smooth-dew-2538/")
    # download_probe_data(run_id="signification-team/signification-game/g8sgtojm", directory="./dainty-waterfall-2537/")
    # download_probe_data(run_id="signification-team/signification-game/cll0cq9m", directory="./youthful-star-2536/")
    # download_probe_data(run_id="signification-team/signification-game/sgk40864", directory="./dutiful-planet-2535/")
    # download_speaker_examples(run_id="signification-team/signification-game/mjylwu94", directory="./smart-resonance-2540/", tom_examples_only=True)

    ### 10 runs of inferential signaling, no penalties, ablated Pr - canvas 0.1
    inferential_no_penalty_runs_ablated_Pr = ["./ruby-butterfly-2569/",
                                                "./earthy-universe-2568/",
                                                "./swift-cherry-2567/",
                                                "./honest-aardvark-2566/",
                                                "./stoic-cherry-2565/",
                                                "./mild-aardvark-2564/",
                                                "./logical-river-2563/",
                                                "./super-thunder-2562/",
                                                "./fast-armadillo-2561/",
                                                "./zany-smoke-2560/"]
    # download_communication_success_data(run_id="signification-team/signification-game/2mhk24u0", directory="./ruby-butterfly-2569/")
    # download_communication_success_data(run_id="signification-team/signification-game/xyb3o3yf", directory="./earthy-universe-2568/")
    # download_communication_success_data(run_id="signification-team/signification-game/z4vgrcn7", directory="./swift-cherry-2567/")
    # download_communication_success_data(run_id="signification-team/signification-game/sxkv8ypd", directory="./honest-aardvark-2566/")
    # download_communication_success_data(run_id="signification-team/signification-game/p6xuay4v", directory="./stoic-cherry-2565/")
    # download_communication_success_data(run_id="signification-team/signification-game/6f5wc6k4", directory="./mild-aardvark-2564/")
    # download_communication_success_data(run_id="signification-team/signification-game/gkxs9f51", directory="./logical-river-2563/")
    # download_communication_success_data(run_id="signification-team/signification-game/jlp9jzod", directory="./super-thunder-2562/")
    # download_communication_success_data(run_id="signification-team/signification-game/50te50qt", directory="./fast-armadillo-2561/")
    # download_communication_success_data(run_id="signification-team/signification-game/ulve0qoh", directory="./zany-smoke-2560/")
    # download_probe_data(run_id="signification-team/signification-game/2mhk24u0", directory="./ruby-butterfly-2569/")
    # download_probe_data(run_id="signification-team/signification-game/xyb3o3yf", directory="./earthy-universe-2568/")
    # download_probe_data(run_id="signification-team/signification-game/z4vgrcn7", directory="./swift-cherry-2567/")
    # download_probe_data(run_id="signification-team/signification-game/sxkv8ypd", directory="./honest-aardvark-2566/")
    # download_probe_data(run_id="signification-team/signification-game/p6xuay4v", directory="./stoic-cherry-2565/")
    # download_probe_data(run_id="signification-team/signification-game/6f5wc6k4", directory="./mild-aardvark-2564/")
    # download_probe_data(run_id="signification-team/signification-game/gkxs9f51", directory="./logical-river-2563/")
    # download_probe_data(run_id="signification-team/signification-game/jlp9jzod", directory="./super-thunder-2562/")
    # download_probe_data(run_id="signification-team/signification-game/50te50qt", directory="./fast-armadillo-2561/")
    # download_probe_data(run_id="signification-team/signification-game/ulve0qoh", directory="./zany-smoke-2560/")
    # download_speaker_examples(run_id="signification-team/signification-game/2mhk24u0", directory="./ruby-butterfly-2569/", tom_examples_only=True)


    # make_avg_com_success_plot([behaviorist_live_listeners_runs,
    #                     inferential_no_penalty_runs,
    #                     inferential_no_penalty_runs_ablated_Pr],
    #                     ["Behaviorist",
    #                     "Inferential",
    #                     "Inf. no P_ref"],
    #                     all_speakers_avg=True,
    #                     rolling_window=25, t_val=1.833)

    referent_names = ("Bicycle", "Butterfly", "Camel", "Crab", "Dolphin", "Tree", "Rocket", "Snail", "Snake", "Spider")

    # make_avg_com_success_across_referents_plot([
    #                     behaviorist_dead_listeners_runs,],
    #                     [
    #                     "Behaviorist (Canalized)"],
    #                     referent_names,
    #                     rolling_window=25, t_val=1.833, y_offsets=[0, 0, 0, -6, 0, 0, 0, 12, -12, 14])


    # make_avg_pr_across_referents_plot([
    #                     inferential_no_penalty_runs,],
    #                     [
    #                     "Inferential",],
    #                     referent_names,
    #                     num_epochs=650,
    #                     rolling_window=25, t_val=1.833, y_offsets=[-2, 0, 0, 0, 0, 0, 0, 0, 0, -12])

    # make_avg_probe_plot([behaviorist_live_listeners_runs,
    #                      behaviorist_dead_listeners_runs,
    #                     inferential_no_penalty_runs,
    #                     inferential_no_penalty_runs_ablated_Pr],
    #                      ["Behaviorist",
    #                       "Behaviorist (Canalized)",
    #                     "Inferential",
    #                     "Inf. no P_ref"],
    #                     all_speakers_avg=True,
    #                     rolling_window=100, t_val=1.833)

    speaker_selection = [0, 0, 2, 4, 8, 6, 6, 6, 14, 4]
    # directories = ["./ethereal-butterfly-2559/", "./still-pond-2549/"]
    
    # for directory in directories[-2:]:
    #     make_speaker_example_graphic(directory, image_dim=32, speaker_selection=speaker_selection, referent_selection=list(range(10)), start_epoch=149, count=15, interval_epoch=180)
    #     make_speaker_example_graphic(directory, image_dim=32, speaker_selection=speaker_selection, referent_selection=list(range(10)), start_epoch=149, count=15, epoch_span=2800, x_stretch=100.0, method="1/x")
    #     make_speaker_example_graphic(directory, image_dim=32, speaker_selection=speaker_selection, referent_selection=list(range(10)), start_epoch=149, count=15, epoch_span=2800, x_stretch=0.0, method="1/x")
        # make_speaker_example_graphic(directory, start_epoch=199, count=10, interval_epoch=300, speaker_selection=speaker_selection, referent_selection=list(range(10)), fname_prefix="tom_", image_dim=32)
        # make_speaker_example_graphic(directory, start_epoch=199, count=20, interval_epoch=150, speaker_selection=speaker_selection, referent_selection=list(range(10)), fname_prefix="tom_", image_dim=32)
        # make_speaker_example_graphic(directory, start_epoch=199, count=10, epoch_span=3000, x_stretch=100.0, method="1/x", speaker_selection=speaker_selection, referent_selection=list(range(10)), fname_prefix="tom_", image_dim=32)
        # make_speaker_example_graphic(directory, start_epoch=199, count=20, epoch_span=3000, x_stretch=100.0, method="1/x", speaker_selection=speaker_selection, referent_selection=list(range(10)), fname_prefix="tom_", image_dim=32)
        # make_speaker_example_graphic(directory, start_epoch=199, count=10, epoch_span=3000, x_stretch=0.0, method="1/x", speaker_selection=speaker_selection, referent_selection=list(range(10)), fname_prefix="tom_", image_dim=32)
        # make_speaker_example_graphic(directory, start_epoch=199, count=20, epoch_span=3000, x_stretch=0.0, method="1/x", speaker_selection=speaker_selection, referent_selection=list(range(10)), fname_prefix="tom_", image_dim=32)

    make_multi_animation_noprobe(directories=["./ethereal-butterfly-2559/", "./ruby-butterfly-2569/"], labels=["Behaviorist", "Inferential – P_ref"], fname_prefixes=["", "tom_"], speaker_selection=speaker_selection)
    # make_multi_animation(directories=["./ethereal-butterfly-2559/",  "./ruby-butterfly-2569/", "./smart-resonance-2540/"], labels=["Behaviorist", "Inferential – P_ref", "Inferential"], fname_prefixes=["", "tom_", "tom_"], speaker_selection=speaker_selection)


def make_graphics_fall_2025():
    ### Visual penalty runs 4 splines
    # download_speaker_examples(run_id="signification-team/signification-game/30uim87h", directory="./stellar-vortex-2631/")       # Post-Draft-Part2-R26b - cifar10b tom agents 4 splines 0.1 canvas spline similarity penalty -0.3
    # download_speaker_examples(run_id="signification-team/signification-game/5jqw6pwa", directory="./warm-thunder-2630/")         # Post-Draft-Part2-R26a - cifar10b tom agents 4 splines 0.1 canvas spline similarity penalty 0.3
    # download_speaker_examples(run_id="signification-team/signification-game/qp4s2a6a", directory="./true-universe-2629/")        # Post-Draft-Part2-R24b - cifar10b tom agents 4 splines 0.1 canvas right angle penalty -0.1
    # download_speaker_examples(run_id="signification-team/signification-game/ndhauk3k", directory="./glorious-dew-2628/")         # Post-Draft-Part2-R24a - cifar10b tom agents 4 splines 0.1 canvas right angle penalty 0.1
    # download_speaker_examples(run_id="signification-team/signification-game/veqkm6ry", directory="./misunderstood-meadow-2627/") # Post-Draft-Part2-R23b - cifar10b tom agents 4 splines 0.1 canvas whitesum penalty -0.5
    # download_speaker_examples(run_id="signification-team/signification-game/w6bvvkke", directory="./expert-pond-2626/")          # Post-Draft-Part2-R23a - cifar10b tom agents 4 splines 0.1 canvas whitesum penalty 0.5
    # download_speaker_examples(run_id="signification-team/signification-game/dxmau0va", directory="./stoic-bush-2624/")           # no shuffle rerun - Post-Draft-Part2-R21c - cifar10b tom agents 4 splines 0.1 canvas curve penalty 0.001
    # download_speaker_examples(run_id="signification-team/signification-game/oceg4oyg", directory="./hardy-cosmos-2623/")         # no shuffle rerun - Post-Draft-Part2-R21b - cifar10b tom agents 4 splines 0.1 canvas curve penalty -0.01
    # download_speaker_examples(run_id="signification-team/signification-game/h8ibznoh", directory="./summer-oath-2625/")          # Post-Draft-Part2-R22b - cifar10b tom agents 4 splines 0.2 canvas no penalties

    directories = ("./stellar-vortex-2631/",
                    "./warm-thunder-2630/",
                    "./true-universe-2629/",
                    "./glorious-dew-2628/",
                    "./misunderstood-meadow-2627/",
                    "./expert-pond-2626/",                    
                    "./stoic-bush-2624/",
                    "./hardy-cosmos-2623/",
                    "./summer-oath-2625/")


    ### Visual penalty runs 3 splines
    # download_speaker_examples(run_id="signification-team/signification-game/bdc2ns0f", directory="./solar-dragon-2640/")            # Post-Draft-Part2-R26b - cifar10b tom agents 3 splines 0.3 canvas spline similarity penalty -0.3
    # download_speaker_examples(run_id="signification-team/signification-game/jaigtgpe", directory="./pleasant-aardvark-2637/")       # Post-Draft-Part2-R24a - cifar10b tom agents 3 splines 0.3 canvas right angle penalty 0.1
    # download_speaker_examples(run_id="signification-team/signification-game/lw59sb9i", directory="./stellar-snow-2638/")            # Post-Draft-Part2-R26a - cifar10b tom agents 3 splines 0.3 canvas spline similarity penalty 0.3
    # download_speaker_examples(run_id="signification-team/signification-game/t6kyuyjx", directory="./bright-silence-2638/")          # Post-Draft-Part2-R23b - cifar10b tom agents 3 splines 0.3 canvas whitesum penalty -0.5
    # download_speaker_examples(run_id="signification-team/signification-game/0mxqdjk5", directory="./helpful-tree-2636/")            # Post-Draft-Part2-R24b - cifar10b tom agents 3 splines 0.3 canvas right angle penalty -0.1
    # download_speaker_examples(run_id="signification-team/signification-game/uh9h2zgn", directory="./ruby-snow-2634/")               # Post-Draft-Part2-R23a - cifar10b tom agents 3 splines 0.3 canvas whitesum penalty 0.5
    # download_speaker_examples(run_id="signification-team/signification-game/jaf2ejp2", directory="./silver-violet-2633/")           # Post-Draft-Part2-R21c - cifar10b tom agents 3 splines 0.3 canvas curve penalty 0.001
    # download_speaker_examples(run_id="signification-team/signification-game/r5bnxfd1", directory="./feasible-aardvark-2632/")       # Post-Draft-Part2-R21b - cifar10b tom agents 3 splines 0.3 canvas curve penalty -0.01
    # download_speaker_examples(run_id="signification-team/signification-game/20xx8eba", directory="./peach-shadow-2634/")            # Post-Draft-Part2-R22b - cifar10b tom agents 3 splines 0.3 canvas no penalties


    directories = ("./solar-dragon-2640/",
                    "./pleasant-aardvark-2637/",
                    "./stellar-snow-2638/",
                    "./bright-silence-2638/",
                    "./helpful-tree-2636/",
                    "./ruby-snow-2634/",                    
                    "./silver-violet-2633/",
                    "./feasible-aardvark-2632/",
                    "./peach-shadow-2634/")


    ### Visual penalty runs 3 splines - XL (6000 epochs)

    # Excluded
    # download_speaker_examples(run_id="signification-team/signification-game/q38rm6d9", directory="./comic-meadow-2649/")            # XL Post-Draft-Part2-R25b - cifar10b tom agents 3 splines 0.3 canvas right angle or straight penalty -0.1
    # download_speaker_examples(run_id="signification-team/signification-game/i63san15", directory="./desert-sponge-2646/")           # XL Post-Draft-Part2-R25a - cifar10b tom agents 3 splines 0.3 canvas right angle or straight penalty 0.1

    # download_speaker_examples(run_id="signification-team/signification-game/b29hyp3d", directory="./sunny-wind-2646/")              # XL Post-Draft-Part2-R24a - cifar10b tom agents 3 splines 0.3 canvas right angle penalty 0.1
    # download_speaker_examples(run_id="signification-team/signification-game/k95y3zqx", directory="./divine-oath-2646/")             # XL Post-Draft-Part2-R24b - cifar10b tom agents 3 splines 0.3 canvas right angle penalty -0.1
    # download_speaker_examples(run_id="signification-team/signification-game/eejb1l80", directory="./grateful-firefly-2651/")        # XL Post-Draft-Part2-R26b - cifar10b tom agents 3 splines 0.3 canvas spline similarity penalty -0.3
    # download_speaker_examples(run_id="signification-team/signification-game/n1jn54v1", directory="./dutiful-glade-2650/")           # XL Post-Draft-Part2-R26a - cifar10b tom agents 3 splines 0.3 canvas spline similarity penalty 0.3
    # download_speaker_examples(run_id="signification-team/signification-game/kbt3q1rj", directory="./ruby-dew-2645/")                # XL Post-Draft-Part2-R23b - cifar10b tom agents 3 splines 0.3 canvas whitesum penalty -0.5
    # download_speaker_examples(run_id="signification-team/signification-game/iiacje3z", directory="./unique-sky-2643/")              # XL Post-Draft-Part2-R23a - cifar10b tom agents 3 splines 0.3 canvas whitesum penalty 0.5
    # download_speaker_examples(run_id="signification-team/signification-game/uiax9t3x", directory="./azure-thunder-2644/")           # XL Post-Draft-Part2-R21c - cifar10b tom agents 3 splines 0.3 canvas curve penalty 0.001
    # download_speaker_examples(run_id="signification-team/signification-game/yn1yg2xe", directory="./daily-snowball-2641/")          # XL Post-Draft-Part2-R21b - cifar10b tom agents 3 splines 0.3 canvas curve penalty -0.01
    # download_speaker_examples(run_id="signification-team/signification-game/5ds59x9l", directory="./distinctive-haze-2642/")        # XL Post-Draft-Part2-R22b - cifar10b tom agents 3 splines 0.3 canvas no penalties

    directories = ("./sunny-wind-2646/",
                    "./divine-oath-2646/",
                    # "./grateful-firefly-2651/",
                    # "./dutiful-glade-2650/",
                    "./ruby-dew-2645/",
                    "./unique-sky-2643/",                    
                    "./azure-thunder-2644/",
                    "./daily-snowball-2641/",
                    "./distinctive-haze-2642/")


    # make_multi_speaker_example_graphic_single_sign(directories, one_sign=(4,4), start_epoch=449, count=5, epoch_span=5500, x_stretch=100.0, method="1/x")
    # make_multi_speaker_example_graphic_single_sign(directories, one_sign=(4,4), start_epoch=449, count=5, epoch_span=5500, x_stretch=0.0, method="1/x")
    # make_multi_speaker_example_graphic_single_sign(directories, one_sign=(4,4), start_epoch=449, count=5, interval_epoch=1100)
    # make_multi_speaker_example_graphic_single_sign(directories, one_sign=(5,10), start_epoch=949, count=20, interval_epoch=125)
    # make_multi_speaker_example_graphic_single_sign(directories, one_sign=(5,10), start_epoch=949, count=10, epoch_span=2550, x_stretch=100.0, method="1/x")
    # make_multi_speaker_example_graphic_single_sign(directories, one_sign=(5,10), start_epoch=949, count=20, epoch_span=2550, x_stretch=100.0, method="1/x")
    # make_multi_speaker_example_graphic_single_sign(directories, one_sign=(5,10), start_epoch=949, count=10, epoch_span=2550, x_stretch=0.0, method="1/x")
    # make_multi_speaker_example_graphic_single_sign(directories, one_sign=(5,10), start_epoch=949, count=20, epoch_span=2550, x_stretch=0.0, method="1/x")


    referent_coordinates = [(3, 5), (0, 0), (3, 4)]
    run_labels = ["Baseline", "Right Angle\nReward", "Right Angle\nPenalty", "Whitesum\nPenalty", "Whitesum\nReward", "Curvature\nReward", "Curvature\nPenalty"]

    animation_inds = [8, 0, 1, 4, 5, 6, 7]
    animation_dirs = [directories[i] for i in animation_inds]

    all_labels = ["Bicycle", "Dolphin", "Palm Tree"]

#    make_labeled_animation(animation_dirs, referent_coordinates=referent_coordinates, run_labels=run_labels, epochs=5500)
    make_simple_animation_same_sign_multi_agent(animation_dirs, referent_coordinates=referent_coordinates, epochs=6000, labels=run_labels, start_epoch=500)
    # make_labeled_animation(animation_dirs, num_epochs=5500, epoch_start=500, fname_prefix="tom_", image_dim=32, referent_selection=list(range(10)), speaker_selection=list(np.zeros(10, dtype=int))):

def make_graphics_newyear_2026():
    # download_speaker_examples(run_id="signification-team/phonology-study/el0lk7n8", directory="./comic-moon-918/", tom_examples_only=True)
    # download_speaker_examples(run_id="signification-team/phonology-study/jp39a4yv", directory="./stoic-dust-980/", tom_examples_only=True)
    # download_speaker_examples(run_id="signification-team/phonology-study/g9tmjm5g", directory="./comic-bird-993/", tom_examples_only=True)


    # make_simple_animation(directory="./comic-moon-918/", labels=False, speaker_selection=[2]*10, frames=500, fname_suffix_info="row2", fps=20, cmap='cividis')
    # make_simple_animation(directory="./comic-moon-918/", labels=False, speaker_selection=[4]*10, frames=500, fname_suffix_info="row4", fps=20, cmap='cividis')

    # make_simple_animation(directory="./stoic-dust-980/", labels=False, speaker_selection=[6]*10, frames=600, fname_suffix_info="row6", fps=20, cmap='cividis')
    # make_simple_animation(directory="./stoic-dust-980/", labels=False, speaker_selection=[4]*10, frames=500, fname_suffix_info="row4", fps=20, cmap='summer')

    make_simple_animation(directory="./comic-bird-993/", labels=False, speaker_selection=[6]*10, frames=600, fname_suffix_info="row6f", fps=20, cmap='summer')
    make_simple_animation(directory="./comic-bird-993/", labels=False, speaker_selection=[8]*10, frames=600, fname_suffix_info="row8f", fps=20, cmap='summer')


def make_phonology_graphics():
    df = pd.read_csv("../sweep-run-list/wandb_export_2025-12-20T11_48_15.271-05_00.csv")
    filtered_df = df[["Name", "ID", "Tags", "LISTENER_ARCH", "SPEAKER_ARCH"]]

    # for run_info in filtered_df.values.tolist():
    #     download_spline_data(run_id=f"signification-team/phonology-study/{run_info[1]}", directory=f"./{run_info[0]}/")

    # make_value_grid_plot(filtered_df)


if __name__=="__main__":
    # make_graphics_post_conference()
    # remake_graphics_part1()
    # make_graphics_part2()
    # make_phonology_graphics()
    make_graphics_newyear_2026()
    ## Don't forget to `module load ffmpeg``!
    
    
