"""
Automated runner to evaluate multiple DyMF model instances and generate visualizations.

Usage:
    python run_compare_and_visualize.py --auto            # evaluate all DyMF_* models in ./model/
    python run_compare_and_visualize.py ./model/DyMF_x ./model/DyMF_y

Outputs:
    - `visualizations/raw_results.csv` (summary across models)
    - `visualizations/comprehensive_comparison.png`, `performance_table.png`, `statistical_analysis.png`
    - For each model: `visualizations/<model_name>/trajectories.png`, `shot_type_analysis.png`, `performance_comparison.png`
"""
import os
import sys
import glob
from pathlib import Path

# Import functions from the existing analysis & visualization modules
from analyze_model_performance import collect_evaluation_results, create_comprehensive_comparison, create_performance_table, create_statistical_analysis
import visualize_results as vis


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_compare_and_visualize.py --auto | <model_folder1> [model_folder2 ...]")
        sys.exit(1)

    if sys.argv[1] == '--auto':
        model_folders = glob.glob('./model/DyMF_*')
        model_folders = sorted(model_folders)
        if not model_folders:
            print("No DyMF models found in ./model/")
            sys.exit(1)
        print(f"Auto-discovered {len(model_folders)} models")
    else:
        model_folders = sys.argv[1:]

    print("Evaluating and visualizing the following models:")
    for m in model_folders:
        print(' -', m)

    # 1) Collect evaluation results (this runs evaluation using the existing evaluate function)
    df = collect_evaluation_results(model_folders)

    if df is None or len(df) == 0:
        print("No results collected. Exiting.")
        sys.exit(1)

    out_dir = 'visualizations'
    os.makedirs(out_dir, exist_ok=True)

    # Save raw results
    df.to_csv(os.path.join(out_dir, 'raw_results.csv'), index=False)
    print(f"Saved aggregated results to: {os.path.join(out_dir, 'raw_results.csv')}")

    # 2) Create cross-model visualizations
    create_comprehensive_comparison(df, save_dir=out_dir)
    create_performance_table(df, save_dir=out_dir)
    create_statistical_analysis(df, save_dir=out_dir)

    # 3) For each model, create per-model visualizations in a subdirectory
    for model_folder in model_folders:
        model_name = Path(model_folder).name
        model_out = os.path.join(out_dir, model_name)
        os.makedirs(model_out, exist_ok=True)

        print(f"\nGenerating visualizations for model: {model_name}")
        try:
            # Trajectories (3 rallies by default)
            vis.visualize_trajectories(model_folder, num_rallies=3, save_path=os.path.join(model_out, 'trajectories.png'))
            # Shot type analysis
            vis.create_shot_type_analysis(model_folder, save_path=os.path.join(model_out, 'shot_type_analysis.png'))
            # Performance comparison for single model (will use placeholders if needed)
            vis.create_performance_comparison([model_folder], save_path=os.path.join(model_out, 'performance_comparison.png'))
        except Exception as e:
            print(f"  ✗ Error generating visualizations for {model_name}: {e}")
            continue

    print('\nAll done! Visualizations are saved under the `visualizations/` directory.')


if __name__ == '__main__':
    main()
