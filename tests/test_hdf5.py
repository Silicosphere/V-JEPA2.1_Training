from src.datasets.video_dataset import VideoDataset as V

data = V(['/data/asem.a.abdelaziz/gp-2027a/ego4d_lists/ego4d_low.csv'], frames_per_clip=16, frame_step=4)
idx = 1
buffer, label, indices = data[idx]
print(f"Fetching clip {idx + 1}")
print(buffer[0].shape)
