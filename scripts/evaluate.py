"""
Evaluation script for lake ice cover predictions.

This script evaluates trained models across multiple lakes and metrics including:
- Overall accuracy and IoU
- Fraction Ice Cover (FIC) timeseries metrics
- Spatial pattern similarity (SSIM, Kendall Tau-B, Local Moran's I)
- Phenology event timing (freeze-up/break-up dates)

Configuration:
    Edit the variables below to customize the evaluation:
    - SITES: List of lakes to evaluate
    - COMPARISONS: Observation vs model pairs to compare
    - EVALUATORS: Metrics to compute

Usage:
    # Run full evaluation
    python scripts/evaluate.py --name LIF_DL_Best
    
    # Save intermediate data for figure generation
    python scripts/evaluate.py --name LIF_DL_Best --save-intermediate
    
    # Compute variable importance
    python scripts/evaluate.py --name LIF_DL_Best --variable-importance
"""

import argparse
import logging
from pathlib import Path

from src.eval.engine import DataLoader, ValidationEngine, EvaluationContext
from src.utils.constants import EVAL_START, EVAL_END
from src.eval.evaluators.overall import OverallEvaluator
from src.eval.evaluators.fic import FICEvaluator
from src.eval.evaluators.spatial import SpatialEvaluator
from src.eval.evaluators.phenology import PhenologyEvaluator
from src.eval.io import save_results_as_csv, save_intermediate_data


# ============================================================================
# EVALUATION CONFIGURATION - Edit these to customize what gets evaluated
# ============================================================================

# Sites to evaluate (comment out any you don't want to run)
SITES = [
    "great_bear_lake",
    "great_slave_lake",
    "lake_athabasca",
    "lake_winnipeg",
    "reindeer_lake"
]

# Observation vs model comparisons to perform
COMPARISONS = [
    ("ims", "lif_dl"),
    ("ims", "flake"),
    ("cis", "lif_dl"),
    ("cis", "flake")
]

# Evaluators to run (comment out any you don't want)
EVALUATORS = ['overall', 'fic', 'spatial', 'phenology']

# ============================================================================


def setup_logging(output_dir=None):
    """
    Configure logging for evaluation script.
    
    Args:
        output_dir: Optional Path to evaluations directory for log file.
                   If provided, logs to both console and file.
                   If None, logs only to console.
    """
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if output directory provided)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = output_dir / 'evaluation.log'
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Log that we're saving to file
        logger.info(f"Logging to: {log_file}")
    
    return logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate lake ice cover predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the model run to evaluate (e.g., LIF_DL_Best)"
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate data (FIC timeseries, spatial maps)"
    )
    parser.add_argument(
        "--variable-importance",
        action="store_true",
        help="Compute variable importance using gradient-based attribution"
    )
    
    return parser.parse_args()


def initialize_evaluators():
    """Initialize evaluator objects based on EVALUATORS configuration."""
    all_evaluators = {
        'overall': OverallEvaluator(),
        'fic': FICEvaluator(),
        'spatial': SpatialEvaluator(),
        'phenology': PhenologyEvaluator()
    }
    
    # Filter to requested evaluators
    selected = {name: all_evaluators[name] 
               for name in EVALUATORS 
               if name in all_evaluators}
    
    if not selected:
        raise ValueError(
            f"No valid evaluators specified in EVALUATORS. "
            f"Available: {list(all_evaluators.keys())}"
        )
    
    return selected, list(selected.values())


def print_metric_summary(comp_label, comp_results, logger):
    """Print concise summary of evaluation metrics."""
    logger.info(f"  {comp_label}:")
    
    if 'error' in comp_results:
        logger.info(f"    Error: {comp_results['error']['error']}")
        return
    
    for eval_name, eval_result in comp_results.items():
        if eval_result['status'] == 'skipped':
            # Silently skip - don't log anything for skipped evaluators
            continue
        elif eval_result['status'] != 'ok':
            logger.info(f"    {eval_name}: Error - {eval_result['error']}")
            continue
            
        metrics = eval_result['metrics']
        
        # Show key metrics only
        if eval_name == 'overall':
            acc = metrics.get('All_Accuracy', metrics.get('Overall_Accuracy', 0))
            iou = metrics.get('All_IoU', metrics.get('IoU', 0))
            logger.info(f"    {eval_name}: Accuracy={acc:.3f}, IoU={iou:.3f}")
        
        elif eval_name == 'fic':
            mae = metrics.get('MAE', 0)
            rmse = metrics.get('RMSE', 0)
            logger.info(f"    {eval_name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        elif eval_name == 'spatial':
            fus_ssim = metrics.get('FUS_SSIM', 0)
            bus_ssim = metrics.get('BUS_SSIM', 0)
            logger.info(f"    {eval_name}: FUS_SSIM={fus_ssim:.3f}, BUS_SSIM={bus_ssim:.3f}")
        
        elif eval_name == 'phenology':
            bus_mae = metrics.get('Breakup_Start_MAE', 0)
            fus_mae = metrics.get('Freezeup_Start_MAE', 0)
            logger.info(f"    {eval_name}: BUS_MAE={bus_mae:.1f}d, FUS_MAE={fus_mae:.1f}d")


def evaluate_site(site, loader, engine, logger):
    """
    Evaluate a single site.
    
    Returns
    -------
    tuple
        (site_results dict, EvaluationContext)
    """
    logger.info(f"\nSite: {site}")
    logger.info("="*60)
    
    # Load all sources
    logger.info("Loading data sources...")
    sources = loader.load_all_sources(site)
    logger.info(f"  Loaded {len(sources)} sources: {', '.join(sources.keys())}")
    
    # Load baseline data for MASE
    baseline_sources = loader.load_baseline_year(site)
    if baseline_sources:
        logger.info(f"  Loaded baseline data: {', '.join(baseline_sources.keys())}")
    
    # Load mask
    mask = loader.load_lake_mask(site)
    logger.info(f"  Loaded mask: {mask.shape}")
    
    # Create context
    context = EvaluationContext(
        sources=sources,
        site=site,
        mask=mask,
        metadata={'baseline_sources': baseline_sources}
    )
    
    # Run evaluations
    logger.info("Running evaluations...")
    site_results = engine.run(
        context=context,
        comparisons=COMPARISONS,
        evaluator_filter=EVALUATORS
    )
    
    # Print results
    for comp_label, comp_results in site_results.items():
        print_metric_summary(comp_label, comp_results, logger)
    
    return site_results, context


def run_variable_importance(model_name, output_dir, logger):
    """Compute and save variable importance analysis."""
    logger.info("\n" + "="*60)
    logger.info("Computing Variable Importance")
    logger.info("="*60)
    
    try:
        from src.eval.variable_importance import compute_variable_importance
        
        # With target-date semantics, EVAL_START and EVAL_END directly specify
        # the dates for which we want to compute variable importance
        vi_results = compute_variable_importance(
            model_name=model_name,
            output_dir=str(output_dir),
            test_start=EVAL_START,
            test_end=EVAL_END
        )
        
        logger.info(f"Variable importance saved to: {output_dir / 'variable_importance.csv'}")
        
    except Exception as e:
        logger.error(f"Error computing variable importance: {e}")
        logger.info("Continuing without variable importance analysis...")


def main():
    """Main evaluation pipeline."""
    # Parse arguments first
    args = parse_arguments()
    
    # Setup paths
    model_dir = Path(f"./results/{args.name}")
    if not model_dir.exists():
        # Use basic logging for error if model dir doesn't exist
        logger = setup_logging()
        logger.error(f"Model directory {model_dir} does not exist")
        return
    
    output_dir = model_dir / "evaluations"
    
    # Setup logging with file output to evaluations directory
    logger = setup_logging(output_dir)
    
    # Initialize evaluators
    evaluators_dict, evaluators_list = initialize_evaluators()
    
    # Initialize engine and loader
    loader = DataLoader(model_dir=str(model_dir))
    engine = ValidationEngine(evaluators_list)
    
    # Print configuration
    logger.info("="*60)
    logger.info("Evaluation Configuration")
    logger.info("="*60)
    logger.info(f"Model: {args.name}")
    logger.info(f"Sites: {', '.join(SITES)}")
    logger.info(f"Comparisons: {', '.join([f'{o}_vs_{p}' for o, p in COMPARISONS])}")
    logger.info(f"Evaluators: {', '.join([e.name for e in evaluators_list])}")
    logger.info("="*60)
    
    # Run evaluations
    all_results = {}
    all_contexts = {}
    
    for site in SITES:
        try:
            site_results, context = evaluate_site(
                site, loader, engine, logger
            )
            all_results[site] = site_results
            all_contexts[site] = context
            
        except Exception as e:
            logger.error(f"Error processing {site}: {str(e)}")
            all_results[site] = {"error": str(e)}
    
    # Save CSV results
    logger.info("\n" + "="*60)
    logger.info("Saving Results")
    logger.info("="*60)
    
    try:
        saved_files = save_results_as_csv(all_results, output_dir)
        logger.info(f"Results saved to: {output_dir}/")
        for file_path in saved_files:
            logger.info(f"  {file_path.name}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Save intermediate data if requested
    if args.save_intermediate:
        intermediate_dir = output_dir / "intermediate"
        
        logger.info("\n" + "="*60)
        logger.info(f"Saving Intermediate Data to: {intermediate_dir}/")
        logger.info("="*60)
        
        for site in SITES:
            if site in all_contexts:
                logger.info(f"{site}:")
                try:
                    saved = save_intermediate_data(
                        context=all_contexts[site],
                        evaluators_dict=evaluators_dict,
                        output_dir=intermediate_dir
                    )
                    logger.info(f"  Saved {len(saved)} files")
                except Exception as e:
                    logger.error(f"  Error: {e}")
        
        logger.info(f"Intermediate data saved.")
    
    # Compute variable importance if requested
    if args.variable_importance:
        run_variable_importance(args.name, output_dir, logger)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete")
    logger.info("="*60)

if __name__ == "__main__":
    main()
