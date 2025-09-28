import time


def print_run_header(args, output_file):
    print("=" * 60)
    print(" Running experiment ")
    print("=" * 60)
    print(f" Model ID       : {args.model_id}")
    print(f" Generation     : {args.method}")
    print(f" Device         : {args.device}")
    print(f" Max tokens     : {args.max_tokens}")
    print(f" Output file    : {output_file}")
    print("=" * 60)


def print_run_summary(start_time, total_samples, output_file):
    total_elapsed = time.time() - start_time
    avg_time = total_elapsed / total_samples if total_samples > 0 else 0

    print("=" * 60)
    print(" Run summary ")
    print("=" * 60)
    print(f" Samples processed : {total_samples}")
    print(f" Total time        : {total_elapsed/60:.1f} min")
    print(f" Avg per sample    : {avg_time:.2f} sec")
    print(f" Output saved to   : {output_file}")
    print("=" * 60)
