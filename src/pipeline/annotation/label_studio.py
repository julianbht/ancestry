"""Thin re-export for Label Studio CLI helpers."""

from pipeline.annotation.label_studio_cli import main
from pipeline.annotation.start_label_studio import start_label_studio

__all__ = ["main", "start_label_studio"]


if __name__ == "__main__":
    main()
