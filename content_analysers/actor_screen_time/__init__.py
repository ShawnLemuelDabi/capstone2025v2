# content_analysers/actor_screen_time/__init__.py
from .actor_screen_time import main, train_screen_time_model, process_video_for_screen_time, select_video_directory
from .frames_extractor import FramesExtractor, extract_frames
from .frames_sequencer import FramesSequencer

__all__ = [
    'main',
    'train_screen_time_model',
    'process_video_for_screen_time',
    'select_video_directory',
    'FramesExtractor',
    'extract_frames',
    'FramesSequencer',
    'sequence_frames'
]

def sequence_frames():
    """Public function to access the frames sequencing functionality."""
    sequencer = FramesSequencer()
    return sequencer.select_and_rename_frames()