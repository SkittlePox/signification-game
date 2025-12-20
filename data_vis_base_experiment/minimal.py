import wandb

run_id = "signification-team/phonology-study/9v0cqy5b"

api = wandb.Api()
run = api.run(run_id)

history = run.scan_history()

data = [row["spline wasserstein distances/all speakers mean"] for row in history]

print(data)

# This data should be what is in this panel: https://wandb.ai/signification-team/phonology-study/runs/9v0cqy5b/panel/j3jecwkfu?nw=nwusertoes
# The data is logged every 5 epochs, starting at epoch 4, so 4, 9, 14, 19, etc. You can verify this by exporting the panel data as csv.
# But the data collected via run.scan_history() is clearly not correct:
# [None, None, None, 0.45441022515296936, None, None, 0.4634958505630493, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.6643260717391968,

# Downloading other data that is recorded every epoch works as expected, e.g. "reward/mean reward/all listeners" downloads as expected
