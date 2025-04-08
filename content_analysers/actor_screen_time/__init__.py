"""
This package contains tools for actor screen time analysis and training data generation.

Includes functionality for:
- Screen time analysis (main processing, model training, video processing)
- Frames extraction and sequencing
- Training data CSV generation (positive/negative samples mapping)
"""

# Package version
__version__ = "1.1.0"  # Updated version

from .actor_screen_time import process_video, process_directory

# Frames processing imports
from .frames_extractor import FramesExtractor, extract_frames
from .frames_sequencer import FramesSequencer

# CSV mapping imports
from .mapping_csv_editor import (
    generate_csv,
    select_directories_and_generate
)


def sequence_frames():
    """Public function to access the frames sequencing functionality."""
    sequencer = FramesSequencer()
    return sequencer.select_and_rename_frames()


__all__ = [
    # Screen time analysis
    'process_video',
    'process_directory'

    # Frames processing
    'FramesExtractor',
    'extract_frames',
    'FramesSequencer',
    'sequence_frames',

    # CSV mapping
    'generate_csv',
    'select_directories_and_generate'
]

"""
content_analysers.actor_screen_time package

This package contains the actor screen time analysis module.
"""
