"""CLI entrypoint for face annotation Label Studio helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.annotation.face_annotation_build import (
    build_face_tasks,
    generate_ls_template,
    normalize_face_export,
)
from pipeline.annotation.face_annotation_settings import (
    apply_build_overrides,
    apply_normalize_overrides,
    load_face_annotation_config,
    resolved_existing_ground_truth,
    resolved_existing_tasks,
    resolved_export_path,
    resolved_ground_truth_output,
    resolved_tasks_output,
    resolved_template_output,
)
from pipeline.gramps import load_family_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build Label Studio tasks from all images in data/raw/")
    build.add_argument("--existing-tasks", type=Path, default=None)
    build.add_argument("--existing-ground-truth", type=Path, default=None)
    build.add_argument("--output", type=Path, default=None)
    build.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly select N not-yet-annotated images instead of all of them.",
    )
    build.add_argument(
        "--seed", type=int, default=None, help="Seed for --sample (reproducible selection)."
    )
    build.add_argument(
        "--include",
        nargs="+",
        default=[],
        metavar="KEY",
        help="Force-include specific raw keys even if already annotated (e.g. to redo one).",
    )

    normalize = sub.add_parser(
        "normalize", help="Normalize a Label Studio export into face ground truth"
    )
    normalize.add_argument("--export", type=Path, default=None)
    normalize.add_argument("--output", type=Path, default=None)

    sub.add_parser(
        "generate-template",
        help="Generate the Label Studio XML template from the family tree",
    )

    args = parser.parse_args()

    try:
        config, _ = load_face_annotation_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.command == "build":
        config = apply_build_overrides(
            config, args.existing_tasks, args.output, args.existing_ground_truth
        )
        build_face_tasks(
            tasks_output=resolved_tasks_output(config),
            existing_tasks=resolved_existing_tasks(config),
            annotated_ground_truth=resolved_existing_ground_truth(config),
            sample=args.sample,
            seed=args.seed,
            include=tuple(args.include),
        )

    elif args.command == "normalize":
        config = apply_normalize_overrides(config, args.export, args.output)
        normalize_face_export(
            export_path=resolved_export_path(config),
            ground_truth_output=resolved_ground_truth_output(config),
        )

    elif args.command == "generate-template":
        persons = load_family_tree(config.gramps_source)
        generate_ls_template(persons, resolved_template_output(config))


if __name__ == "__main__":
    main()
