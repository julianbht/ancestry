"""CLI entrypoint for Label Studio helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.annotation.label_studio_build import build_tasks, normalize_export
from pipeline.annotation.label_studio_models import RunInputs, RunOutputs, RunResults
from pipeline.annotation.label_studio_paths import relativize
from pipeline.annotation.label_studio_settings import (
    apply_build_overrides,
    apply_normalize_overrides,
    build_settings_from_config,
    load_annotation_config,
    normalize_settings_from_config,
    tracking_settings_from_config,
)
from pipeline.annotation.label_studio_tracking import write_run_manifest
from pipeline.annotation.start_label_studio import start_label_studio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Annotation config file (default: config/annotation/step.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start Label Studio with local file serving")
    start.add_argument(
        "label_studio_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed directly to label-studio",
    )

    build = sub.add_parser("build", help="Build Label Studio tasks from cases files")
    build.add_argument("--cases", type=Path, nargs="+", default=None)
    build.add_argument("--existing-tasks", type=Path, default=None)
    build.add_argument("--output", type=Path, default=None)

    normalize = sub.add_parser(
        "normalize", help="Normalize a Label Studio export into ground truth"
    )
    normalize.add_argument("--export", type=Path, default=None)
    normalize.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    if args.command == "start":
        raise SystemExit(start_label_studio(args.label_studio_args))

    try:
        config, config_path = load_annotation_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    tracking = tracking_settings_from_config(config)

    if args.command == "build":
        config = apply_build_overrides(
            config, args.cases, args.existing_tasks, args.output
        )
        settings = build_settings_from_config(config)
        outcome = build_tasks(settings)

        inputs = RunInputs(
            cases_files=[relativize(path) for path in settings.cases_files],
            existing_tasks=(
                relativize(settings.existing_tasks)
                if settings.existing_tasks is not None
                else None
            ),
        )
        outputs = RunOutputs(tasks_output=relativize(settings.tasks_output))
        results = RunResults(
            tasks_written=outcome.tasks_written,
            skipped_existing=outcome.skipped_existing,
        )
        write_run_manifest(
            tracking=tracking,
            command="build",
            label_name=config.label_name,
            inputs=inputs,
            outputs=outputs,
            results=results,
            annotation_config_path=config_path,
            frame_crop_config_path=tracking.frame_crop_config_path,
        )
        return

    if args.command == "normalize":
        config = apply_normalize_overrides(config, args.export, args.output)
        settings = normalize_settings_from_config(config)
        outcome = normalize_export(settings, config.label_name)

        inputs = RunInputs(export_path=relativize(settings.export_path))
        outputs = RunOutputs(
            ground_truth_output=relativize(settings.ground_truth_output)
        )
        results = RunResults(
            labels_written=outcome.labels_written,
            skipped_tasks=outcome.skipped,
        )
        write_run_manifest(
            tracking=tracking,
            command="normalize",
            label_name=config.label_name,
            inputs=inputs,
            outputs=outputs,
            results=results,
            annotation_config_path=config_path,
            frame_crop_config_path=tracking.frame_crop_config_path,
        )
        return


if __name__ == "__main__":
    main()
