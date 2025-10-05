import time
import sys
from loguru import logger


def print_run_chair_header(args, output_file):
    print("=" * 60)
    print(" Running experiment ")
    print("=" * 60)
    print(f" Model ID       : {args.model_id}")
    print(f" Generation     : {args.method}")
    print(f" Device         : {args.device}")
    print(f" Max tokens     : {args.max_tokens}")
    print(f" Output file    : {output_file}")
    print("=" * 60)


def print_run_chair_summary(start_time, total_samples, output_file):
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


def print_run_pope_header(args, output):
    print("=" * 50)
    print(f"Running POPE evaluation")
    print(f"Model: {args.model_id}")
    print(f"Method: {args.method}")
    print(f"POPE Type: {args.pope_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output dir: {output}")
    print("=" * 50)


def print_run_pope_summary(start_time, total_samples, output_file):
    total_elapsed = time.time() - start_time
    avg_time = total_elapsed / total_samples if total_samples > 0 else 0

    print("=" * 50)
    print(" Run summary ")
    print("=" * 50)
    print(f" Samples processed : {total_samples}")
    print(f" Total time        : {total_elapsed/60:.1f} min")
    print(f" Avg per sample    : {avg_time:.2f} sec")
    print(f" Output saved to   : {output_file}")
    print("=" * 50)


def setup_logger(debug: bool = False, log_file: str = None, silent: bool = False):
    """Configure Loguru logging globally. Use `silent=True` to suppress all output."""
    logger.remove()

    if silent:
        # Don’t add any sink → no output anywhere
        return

    # Normal mode (info or debug)
    level = "DEBUG" if debug else "INFO"
    fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=fmt, enqueue=True)

    if log_file:
        logger.add(log_file, level=level, rotation="20 MB")

    logger.info(f"Logger initialized at level: {level}")
