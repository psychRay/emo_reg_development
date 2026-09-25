"""Run the manuscript analyses from a single entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run figure-specific analyses")
    parser.add_argument("--figure", type=int, choices=[1, 2, 3, 4, 5, 6], required=True)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--toy", action="store_true", help="Generate deterministic toy data")
    source.add_argument("--input", type=Path, help="Analysis-ready participant table; format depends on the figure")
    parser.add_argument("--profiles", type=Path, help="Figure 2 ERQ profile CSV")
    parser.add_argument("--schaefer-atlas", type=Path,
                        help="Figure 3 Schaefer CIFTI dlabel atlas")
    parser.add_argument("--fig3-volume-dir", type=Path,
                        help="Figure 3 subject folders with beta_0001/0002/0003.nii.gz")
    parser.add_argument("--fig3-brain-mask", type=Path,
                        help="Figure 3 MNI volume mask for voxelwise GLM")
    parser.add_argument("--fig3-volume-atlas", type=Path,
                        default=Path("resources/atlases/schaefer2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"),
                        help="Figure 3 MNI volume Schaefer atlas for ROI age GAM")
    parser.add_argument("--fig3-workbench", default="wb_command",
                        help="Connectome Workbench wb_command executable for Figure 3 border files")
    parser.add_argument("--glm-results-dir", type=Path,
                        help="Existing whole-brain and age-bin GLM maps for Figure 3b/c/e")
    parser.add_argument("--reference-maps", type=Path,
                        default=Path("resources/reference_maps"),
                        help="Figure 3 reference emotion maps")
    parser.add_argument("--yeo-atlas", type=Path,
                        default=Path("resources/atlases/yeo2011/Yeo2011_7Networks_N1000.dlabel.nii"),
                        help="Figure 3 fsLR Yeo seven-network atlas")
    parser.add_argument("--fig4-age-positive-mask", type=Path,
                        help="Figure 4 positive FDR-significant age-effect mask")
    parser.add_argument("--fig4-age-negative-mask", type=Path,
                        help="Figure 4 negative FDR-significant age-effect mask")
    parser.add_argument("--fig4-mediator-mask-dir", type=Path,
                        help="Directory containing aMTG.nii.gz and aSTG.nii.gz ROI masks for Figure 4e")
    parser.add_argument("--fig4-yeo-volume-atlas", type=Path,
                        default=Path("resources/atlases/yeo2011/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz"),
                        help="Figure 4 MNI volume Yeo seven-network atlas")
    parser.add_argument("--fig5-data-dir", type=Path,
                        help="Directory containing the three Fig. 5 parcel CSVs and subs_info.csv")
    parser.add_argument("--fig5-permutations", type=int,
                        help="Override Fig. 5 model permutation count")
    parser.add_argument("--fig5-bootstraps", type=int,
                        help="Override Fig. 5 decoder/Haufe subject bootstrap count")
    parser.add_argument("--fig5-n-jobs", type=int,
                        help="Override Fig. 5 parallel jobs")
    parser.add_argument("--fig6-config", type=Path,
                        help="Figure 6 config in the format used by analysis/fig06/pipeline/run_paper.py")
    parser.add_argument("--fig6-steps", type=str,
                        help="Comma-separated Figure 6 steps to run; defaults to enabled config steps")
    parser.add_argument("--output", type=Path, help="Output directory")
    args = parser.parse_args()

    if args.figure == 1:
        if not args.toy and args.input is None:
            parser.error("Figure 1 requires --toy or --input")
        from analysis.fig01.run import run_figure_1

        output = args.output or Path("outputs") / "fig01" / ("toy" if args.toy else "real")
        run_figure_1(input_path=args.input, toy=args.toy, output_dir=output)
        if args.toy:
            from analysis.toy_outputs import finish_toy_outputs
            finish_toy_outputs(1, output)
        print(f"Figure 1 outputs: {output.resolve()}")
    elif args.figure == 2:
        if not args.toy and args.input is None:
            parser.error("Figure 2 requires --toy or --input")
        from analysis.fig02.run import run_figure_2

        output = args.output or Path("outputs") / "fig02" / ("toy" if args.toy else "real")
        run_figure_2(input_path=args.input, profiles_path=args.profiles,
                     toy=args.toy, output_dir=output)
        if args.toy:
            from analysis.toy_outputs import finish_toy_outputs
            finish_toy_outputs(2, output)
        print(f"Figure 2 outputs: {output.resolve()}")
    elif args.figure == 3:
        output = args.output or Path("outputs") / "fig03" / ("toy" if args.toy else "real")
        if args.toy or args.input is not None:
            from analysis.fig03.run import run_figure_3

            run_figure_3(input_path=args.input, toy=args.toy, output_dir=output,
                         schaefer_atlas=args.schaefer_atlas,
                         volume_dir=args.fig3_volume_dir,
                         brain_mask=args.fig3_brain_mask,
                         volume_atlas=args.fig3_volume_atlas,
                         workbench_command=args.fig3_workbench)
            if args.toy:
                from analysis.toy_outputs import finish_toy_outputs
                finish_toy_outputs(3, output)
            print(f"Figure 3a/3d outputs: {output.resolve()}")
        if args.glm_results_dir is not None:
            from analysis.fig03.volume_panels import run_volume_panels

            run_volume_panels(
                results_dir=args.glm_results_dir,
                reference_dir=args.reference_maps,
                yeo_atlas=args.yeo_atlas,
                output_dir=output,
            )
            print(f"Figure 3b/3c/3e outputs: {output.resolve()}")
        if not args.toy and args.input is None and args.glm_results_dir is None:
            parser.error("Figure 3 requires --toy, --input, or --glm-results-dir")
    elif args.figure == 4:
        if not args.toy and args.input is None:
            parser.error("Figure 4 requires --toy or --input participant manifest CSV")
        from analysis.fig04.run import run_figure_4

        output = args.output or Path("outputs") / "fig04" / ("toy" if args.toy else "real")
        run_figure_4(input_path=args.input, age_pos_mask=args.fig4_age_positive_mask,
                     age_neg_mask=args.fig4_age_negative_mask,
                     mediator_mask_dir=args.fig4_mediator_mask_dir,
                     atlas_path=args.fig4_yeo_volume_atlas, toy=args.toy, output_dir=output)
        if args.toy:
            from analysis.toy_outputs import finish_toy_outputs
            finish_toy_outputs(4, output)
        print(f"Figure 4 outputs: {output.resolve()}")
    elif args.figure == 5:
        if not args.toy and args.fig5_data_dir is None:
            parser.error("Figure 5 requires --toy or --fig5-data-dir")
        from analysis.fig05.run import run_figure_5

        output = args.output or Path("outputs") / "fig05" / ("toy" if args.toy else "real")
        run_figure_5(data_dir=args.fig5_data_dir, toy=args.toy, output_dir=output,
                     atlas_path=args.schaefer_atlas, bootstraps=args.fig5_bootstraps,
                     permutations=args.fig5_permutations, n_jobs=args.fig5_n_jobs)
        if args.toy:
            from analysis.toy_outputs import finish_toy_outputs
            finish_toy_outputs(5, output)
        print(f"Figure 5 outputs: {output.resolve()}")
    elif args.figure == 6:
        if not args.toy and args.fig6_config is None:
            parser.error("Figure 6 requires --toy or --fig6-config")
        from analysis.fig06.run import run_figure_6

        if not args.toy and args.output is not None:
            parser.error("Figure 6 real outputs use paths.matrix_dir in --fig6-config")
        if args.toy and args.fig6_steps is not None:
            parser.error("Figure 6 toy runs the complete pipeline; --fig6-steps is for participant data")
        output = args.output or Path("outputs") / "fig06" / "toy"
        actual_output = run_figure_6(config_path=args.fig6_config, toy=args.toy,
                                     output_dir=output, steps=args.fig6_steps)
        print(f"Figure 6 outputs: {actual_output.resolve()}")


if __name__ == "__main__":
    main()
