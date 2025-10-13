import time
import sys
from loguru import logger


def print_run_chair_header(args, output_file):
    logger.info("=" * 60)
    logger.info(" Running experiment ")
    logger.info("=" * 60)
    logger.info(f" Model ID       : {args.model_id}")
    logger.info(f" Generation     : {args.method}")
    logger.info(f" Device         : {args.device}")
    logger.info(f" Max tokens     : {args.max_tokens}")
    logger.info(f" Output file    : {output_file}")
    logger.info("=" * 60)


def print_run_chair_summary(start_time, total_samples, output_file):
    total_elapsed = time.time() - start_time
    avg_time = total_elapsed / total_samples if total_samples > 0 else 0

    logger.info("=" * 60)
    logger.info(" Run summary ")
    logger.info("=" * 60)
    logger.info(f" Samples processed : {total_samples}")
    logger.info(f" Total time        : {total_elapsed/60:.1f} min")
    logger.info(f" Avg per sample    : {avg_time:.2f} sec")
    logger.info(f" Output saved to   : {output_file}")
    logger.info("=" * 60)


def print_run_pope_header(args, output):
    logger.info("=" * 50)
    logger.info(f"Running POPE evaluation")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Method: {args.method}")
    logger.info(f"POPE Type: {args.pope_type}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Output dir: {output}")
    logger.info("=" * 50)


def print_run_pope_summary(start_time, total_samples, output_file):
    total_elapsed = time.time() - start_time
    avg_time = total_elapsed / total_samples if total_samples > 0 else 0

    logger.info("=" * 50)
    logger.info(" Run summary ")
    logger.info("=" * 50)
    logger.info(f" Samples processed : {total_samples}")
    logger.info(f" Total time        : {total_elapsed/60:.1f} min")
    logger.info(f" Avg per sample    : {avg_time:.2f} sec")
    logger.info(f" Output saved to   : {output_file}")
    logger.info("=" * 50)


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
