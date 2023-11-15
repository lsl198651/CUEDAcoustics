#!/usr/bin/env python
from team_code import train_challenge_model_full
import datetime
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--prev_run", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.prev_run is None:
        current_time = datetime.datetime.now()
        filename =  current_time.strftime("%Y-%m-%d__%H-%M-%S")
    else:
        filename = args.prev_run
    model_folder = "model_runs/" + filename

    murmur_score, outcome_score = train_challenge_model_full(args.data, 
                                model_folder, 
                                args.verbose,
                                gpu=not args.cpu, 
                                load_old_file=args.prev_run is not None,
                                quick=args.quick)

    if not args.prev_run:
        with open("model/results.txt", "a") as myfile:
            myfile.write(f"\n{model_folder}\t\t{murmur_score:.3f}\t{outcome_score:.0f}\t{'q' if args.quick else ''}")

