# core.py
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from evaluation.tail_posture_analysis.utils import COLORS
from pipeline.utils.path_manager import PathManager

class TailPostureAnalysisBase:
    """Base class for tail posture analysis with core configuration."""
    
    def __init__(self, config=None):
        """Initialize with configuration parameters."""
        # Default configuration 
        self.config = {
            'resample_freq': 'D',
            'normalize': True,
            'smoothing_method': 'rolling',
            'smoothing_strength': 3,
            'days_before_list': [1, 2, 3, 5, 7, 10],
            'output_dir': 'tail_posture_descriptive_results',
            'random_seed': 42,
            'confidence_level': 0.95,
            'interpolation_method': 'linear',
            'interpolation_order': 3,
            'max_allowed_consecutive_missing_days': 3,
        }

        # Update with user config if provided
        if config is not None:
            self.config.update(config)

        # Initialize path manager
        self.path_manager = PathManager()

        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)

        # Setup logging FIRST before we try to use the logger
        self._setup_logging()

        # Save configuration (now can use logger)
        self._save_config()

        # Initialize data containers
        self.monitoring_results = None
        self.processed_results = []
        self.pre_outbreak_stats = None
        self.control_stats = None
        self.outbreak_patterns = None
        self.excluded_events_count = 0
        
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("tail_posture_analysis_descriptive.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("tail_posture_descriptive_analysis")
        
    def _save_config(self):
        """Save the current configuration to file."""
        config_path = os.path.join(self.config['output_dir'], 'analysis_config.txt')
        with open(config_path, 'w') as f:
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
        self.logger.info(f"Saved configuration to {config_path}")