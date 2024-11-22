import wandb

def download_speaker_examples(run_id, directory):
    api = wandb.Api()
    run = api.run(run_id)
    for file in run.files():
        if "speaker_examples" in str(file):
            file.download(root=directory)


if __name__=="__main__":
    download_speaker_examples(run_id="signification-team/signification-game/avnly640", directory="./drawn-shape-1950/")
