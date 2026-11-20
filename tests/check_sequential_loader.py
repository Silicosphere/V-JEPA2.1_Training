"""Check that make_videodataset(shuffle=False) yields CSV rows in order.

Builds the real dataloader for every rank, records which CSV row each sample
came from, and compares that against the expected order (rank r gets rows
r, r + world_size, r + 2 * world_size, ...). As a sanity check it runs the same
comparison with shuffle=True, which must fail.

By default nothing is decoded, so this only tests sampler + DataLoader order
and runs in seconds. Pass --decode to load the real clips as well; then a row
that fails to load (and is replaced by a random row) is reported.

Usage:
    python tests/check_sequential_loader.py --csv /path/to/list.csv --world-size 4
"""

import argparse
import sys

import src.datasets.video_dataset as vd

_VideoDataset = vd.VideoDataset


class ProbeDataset(_VideoDataset):
    decode = False

    def get_item_video(self, index):
        if self.decode and super().get_item_video(index) is None:
            return None
        return (index,)  # row that actually loaded

    def __getitem__(self, index):
        return index, super().__getitem__(index)[0]


def collect(args, shuffle, rank, epoch):
    """Return (requested rows, loaded rows) seen by one rank in one epoch."""
    vd.VideoDataset = ProbeDataset
    try:
        _, loader, sampler = vd.make_videodataset(
            data_paths=[args.csv],
            batch_size=args.batch_size,
            frames_per_clip=args.frames_per_clip,
            frame_step=args.frame_step,
            rank=rank,
            world_size=args.world_size,
            num_workers=args.num_workers,
            persistent_workers=False,
            pin_mem=False,
            shuffle=shuffle,
        )
    finally:
        vd.VideoDataset = _VideoDataset
    sampler.set_epoch(epoch)
    requested, loaded = [], []
    for b, (req, got) in enumerate(loader):
        requested += req.tolist()
        loaded += got.tolist()
        if b + 1 >= args.max_batches:
            break
    return requested, loaded, len(loader.dataset)


def expected_rows(n, rank, world_size, count):
    # DistributedSampler(shuffle=False) pads by wrapping to the start
    return [(rank + k * world_size) % n for k in range(count)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--frames-per-clip", type=int, default=16)
    p.add_argument("--frame-step", type=int, default=4)
    p.add_argument("--decode", action="store_true", help="also decode the clips")
    args = p.parse_args()
    ProbeDataset.decode = args.decode

    ok = True
    for rank in range(args.world_size):
        req0, got0, n = collect(args, shuffle=False, rank=rank, epoch=0)
        req1, _, _ = collect(args, shuffle=False, rank=rank, epoch=1)
        want = expected_rows(n, rank, args.world_size, len(req0))

        in_order = req0 == want
        same_each_epoch = req0 == req1
        retried = [(r, g) for r, g in zip(req0, got0) if r != g]
        ok &= in_order and same_each_epoch and not retried

        print(f"rank {rank}: first rows {req0[:10]}")
        print(f"  in order:            {in_order}")
        print(f"  same every epoch:    {same_each_epoch}")
        if args.decode:
            print(f"  failed loads:        {len(retried)} {retried[:5]}")
        if not in_order:
            bad = next(i for i, (a, b) in enumerate(zip(req0, want)) if a != b)
            print(f"  first mismatch at position {bad}: got {req0[bad]}, expected {want[bad]}")

    # The check must be able to detect random order
    req_shuf, _, _ = collect(args, shuffle=True, rank=0, epoch=0)
    caught = req_shuf != expected_rows(n, 0, args.world_size, len(req_shuf))
    print(f"sanity: shuffle=True detected as out of order: {caught}")
    ok &= caught

    print(f"\n{'PASS' if ok else 'FAIL'} ({len(req0)} samples per rank checked, {n} rows in CSV)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
