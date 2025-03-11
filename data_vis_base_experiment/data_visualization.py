import wandb
import os
import numpy as np
import uuid
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
import json

def download_speaker_examples(run_id, directory, tom_examples_only=False):
    fname_fragment = "tom_speaker_examples" if tom_examples_only else "speaker_examples"
    os.makedirs(directory, exist_ok=True)
    api = wandb.Api()
    run = api.run(run_id)
    files = run.files()
    for file in tqdm(files, desc="Downloading speaker examples"):
        if fname_fragment in str(file):
            file.download(root=directory)

def download_probe_data(run_id, directory, which_speakers=[0]):
    os.makedirs(directory, exist_ok=True)
    api = wandb.Api()
    run = api.run(run_id)
    history = run.scan_history()
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
    history = run.scan_history()
    # for sp_num in which_speakers:
    #     probe_entropy = [row[f"probe/entropy/speaker {sp_num} average"] for row in tqdm(history, desc="Downloading probe data")]
    #     probe_entropy_df = pd.DataFrame(probe_entropy)
    #     probe_entropy_df.to_csv(os.path.join(directory, f"probe_entropy_speaker_{sp_num}.csv"), index=False)
    probe_entropy = [row[f"reward/mean reward by image source/speaker images all listeners"] for row in tqdm(history, desc="Downloading probe data")]
    probe_entropy_df = pd.DataFrame(probe_entropy)
    probe_entropy_df.to_csv(os.path.join(directory, f"reward_for_speaker_images_all_listeners.csv"), index=False)


def make_speaker_example_graphic(directory, count=5, log_interval=5, image_dim=28, method="uniform", fname_prefix="", speaker_selection=None, referent_selection=None, one_sign=None, vertical=True, **kwargs):
    height_dx = image_dim + 2   # Assuming 2px border
    image_dir = os.path.join(directory, "media/images/env/")
    files = os.listdir(image_dir)

    output_dir = os.path.join(directory, "data_vis/")
    os.makedirs(output_dir, exist_ok=True)

    fname_template = fname_prefix+"speaker_examples_"

    sorted_files = sorted([f for f in files if f.startswith(fname_template)],
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

    # download_reward_data(run_id="signification-team/signification-game/1xty9ob3", directory="./frosty-silence-2354/")
    # download_reward_data(run_id="signification-team/signification-game/cmrqqctn", directory="./dark-cosmos-2353/")
    # download_reward_data(run_id="signification-team/signification-game/6vtdcxr5", directory="./dazzling-meadow-2352/")

    # download_pr_data(run_id="signification-team/signification-game/cmrqqctn", directory="./dark-cosmos-2353/", listeners=(7,))

    # Make evolution graphics
    # directories = ["./dark-cosmos-2353/", "./dazzling-meadow-2352/", "./tough-cloud-2359/", "./glad-dew-2358/"]
    # speaker_selections = [[12, 8, 12, 2, 2, 12, 0, 14, 2, 12],
    #                       [12, 8, 12, 2, 2, 12, 0, 14, 2, 12]]
    
    # speaker_selection = [12, 8, 12, 2, 2, 12, 0, 14, 2, 12]
    # abbreviated_speaker_selection = [12, 2, 12, 2, 12]
    # abbreviated_referent_selection = [0, 4, 5, 8, 9]
    # name_prefixes = ["tom_", "", "tom_", "tom_"]
    # for directory, fname_prefix in list(zip(directories, name_prefixes))[1:]:
    #     make_speaker_example_graphic(directory, image_dim=32, fname_prefix=fname_prefix, speaker_selection=speaker_selection, referent_selection=list(range(10)), start_epoch=149, count=15, interval_epoch=180)
    #     make_speaker_example_graphic(directory, image_dim=32, fname_prefix=fname_prefix, speaker_selection=speaker_selection, referent_selection=list(range(10)), start_epoch=149, count=15, epoch_span=2800, x_stretch=100.0, method="1/x")
    #     make_speaker_example_graphic(directory, image_dim=32, fname_prefix=fname_prefix, speaker_selection=speaker_selection, referent_selection=list(range(10)), start_epoch=149, count=15, epoch_span=2800, x_stretch=0.0, method="1/x")
        # make_speaker_example_graphic(directory, image_dim=32, fname_prefix="tom_", one_sign=(5, 12), vertical=False, start_epoch=149, count=20, interval_epoch=75)
        # make_speaker_example_graphic(directory, image_dim=32, fname_prefix="tom_", one_sign=(5, 12), vertical=False, start_epoch=149, count=10, epoch_span=1800, x_stretch=200.0, method="1/x")
        # make_speaker_example_graphic(directory, image_dim=32, fname_prefix="tom_", one_sign=(5, 12), vertical=False, start_epoch=149, count=10, epoch_span=1800, x_stretch=0.0, method="1/x")

    # make_probe_plot(directories=["./worldly-lion-2349/"],
    #     labels=["Iconicity"],
    #     all_speakers_avg=True,
    #     num_epochs=1800,
    #     epoch_start=150,
    #     markers_on=np.array([150, 260, 685, 1040, 1470, 1720])-150)

    # make_reward_plot(directories=("./dazzling-meadow-2352/", "./dark-cosmos-2353/"),
    #     labels=("Instinctual", "Inferential"),
    #     num_epochs=2800,
    #     epoch_start=0)
    
    # make_probe_plot(directories=("./dazzling-meadow-2352/", "./dark-cosmos-2353/"),
    #     labels=("Instinctual", "Inferential"),
    #     all_speakers_avg=True,
    #     num_epochs=2800,
    #     epoch_start=0)

    # ("Bicycle", "Butterfly", "Camel", "Crab", "Dolphin", "Palm Tree", "Rocket", "Snail", "Snake", "Spider") # list(range(10))

    make_pr_plot(directory="./dark-cosmos-2353/",
        referent_labels=("Snail", "Dolphin", "Palm Tree", "Rocket", "Spider"),
        referent_nums=(7, 4, 5, 6, 9),
        num_epochs=3000,
        epoch_start=0,
        agent_num=7,
        log_scale=True)


def remake_graphics_part1():
    # (Runs 1950: manipulation, 1931: whitesum, 1934: negative whitesum, 1940: auto-centering, 1944: curvature, 1945: negative curvature)

    ### Re-runs of 1950: 2363-2367
    manipulation_runs = ["./comic-rain-2363/", "./rosy-field-2364/", "./dainty-surf-2364/", "./jolly-waterfall-2366/", "./blooming-donkey-2367/"]
    # download_probe_data(run_id="signification-team/signification-game/rnucselq", directory="./comic-rain-2363/")
    # download_probe_data(run_id="signification-team/signification-game/s69h3sh7", directory="./rosy-field-2364/")
    # download_probe_data(run_id="signification-team/signification-game/yf15il82", directory="./dainty-surf-2364/")
    # download_probe_data(run_id="signification-team/signification-game/xqzzhed0", directory="./jolly-waterfall-2366/")
    # download_probe_data(run_id="signification-team/signification-game/uilb1k7z", directory="./blooming-donkey-2367/")

    make_avg_probe_plot([manipulation_runs], ["Manipulation"], num_epochs=3300, rolling_window=100)



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
    plt.legend(fontsize=16, loc=1)
    
    fig.tight_layout()
    uuidstr = str(uuid.uuid4())[:4]
    if agent_num:
        plt.savefig(os.path.join("./joint-plots/", f"inference_prs_for_listener_{agent_num}_all_referents_{uuidstr}.png"))    
    else:
        plt.savefig(os.path.join("./joint-plots/", f"inference_prs_for_all_referents_{uuidstr}.png"))

    config = {
        "directories": directory,
        "labels": referent_labels,
        "num_epochs": num_epochs,
    }

    with open(f'./joint-plots/config_{uuidstr}.json', 'w') as f:
        json.dump(config, f)

    print(f'./joint-plots/config_{uuidstr}.json')

def make_reward_plot(directories, labels, num_epochs=None, epoch_start=0, markers_on=[]):
    datas = [pd.read_csv(os.path.join(directory, f"reward_for_speaker_images_all_listeners.csv")) for directory in directories]
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
    plt.legend(fontsize=16)
    
    fig.tight_layout()
    uuidstr = str(uuid.uuid4())[:4]
    plt.savefig(os.path.join("./joint-plots/", f"reward_for_speaker_images_all_listeners_{uuidstr}.png"))

    config = {
        "directories": directories,
        "labels": labels,
        "num_epochs": num_epochs,
    }

    with open(f'./joint-plots/config_{uuidstr}.json', 'w') as f:
        json.dump(config, f)

    print(f'./joint-plots/config_{uuidstr}.json')


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
            # ax.plot(data.rolling(window=100).mean(), label=labels[i], color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.5)
            ax.plot(data, label=labels[i], color=sns.color_palette("Set1")[i], linewidth=2, alpha=0.5)

    # ax.set_title(f'Probe Entropy for Speaker Signals', fontsize=16)
    # ax.set_xlabel('Epoch', fontsize=16)
    # ax.set_ylabel('Entropy', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # plt.legend(fontsize=16)
    
    fig.tight_layout()
    uuidstr = str(uuid.uuid4())[:4]
    plt.savefig(os.path.join("./joint-plots/", f"probe_entropy_speaker_{sp_num}_{uuidstr}.png" if not all_speakers_avg else f"probe_entropy_all_speakers_{uuidstr}.png"))

    config = {
        "directories": directories,
        "labels": labels,
        "num_epochs": num_epochs,
        "sp_num": sp_num
    }

    with open(f'./joint-plots/config_{uuidstr}.json', 'w') as f:
        json.dump(config, f)

    print(f'./joint-plots/config_{uuidstr}.json')


def make_avg_probe_plot(directorybunch, labels, sp_num=0, all_speakers_avg=False, num_epochs=None, epoch_start=0, markers_on=[], rolling_window=None):
    entropies = []
    cis = []
    for directories in directorybunch:
        # directories = ["/users/bspiegel/signification-game/data_vis_base_experiment/"+d[2:] for d in directories]  # Useful for debug
        data_for_group = [pd.read_csv(os.path.join(directory, f"probe_entropy_speaker_{sp_num}.csv" if not all_speakers_avg else "probe_entropy_all_speakers.csv")) for directory in directories]
        merged_datas = pd.concat(data_for_group, axis=1, keys=range(len(data_for_group)))
        mean_entropy = merged_datas.mean(axis=1)
        ci_entropy = 1.96 * merged_datas.sem(axis=1)
        if rolling_window:
            mean_entropy = mean_entropy.rolling(window=rolling_window, center=True).mean()
            ci_entropy = ci_entropy.rolling(window=rolling_window, center=True).mean()
        entropies.append(mean_entropy)
        cis.append(ci_entropy)
    
    sns.set_theme(style="darkgrid")

    # Plot the data with larger font
    fig, ax = plt.subplots(figsize=(6, 6))
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
    # plt.legend(fontsize=16)
    
    fig.tight_layout()
    uuidstr = str(uuid.uuid4())[:4]
    plt.savefig(os.path.join("./joint-plots/", f"probe_entropy_speaker_{sp_num}_{uuidstr}.png" if not all_speakers_avg else f"probe_entropy_all_speakers_{uuidstr}.png"))

    config = {
        "directories": directories,
        "labels": labels,
        "num_epochs": num_epochs,
        "sp_num": sp_num
    }

    with open(f'./joint-plots/config_{uuidstr}.json', 'w') as f:
        json.dump(config, f)

    print(f'./joint-plots/config_{uuidstr}.json')


if __name__=="__main__":
    remake_graphics_part1()
    
