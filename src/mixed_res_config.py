"""Configuration for mixed resolution operations (rescale and flip only, no crop)"""

# Default configuration
DEFAULT_MIXED_RES_CONFIG = {
    'use_random_resize': True,
    'use_multi_resolution': True,
    'use_flip': True,
    'min_scale': 0.5,
    'max_scale': 1.5,
    'resolution_scales': [0.5, 0.75, 1.0, 1.25, 1.5]
}

# Configuration for testing with minimal augmentation
LIGHT_MIXED_RES_CONFIG = {
    'use_random_resize': True,
    'use_multi_resolution': False,
    'use_flip': True,
    'min_scale': 0.8,
    'max_scale': 1.2,
    'resolution_scales': [0.8, 1.0, 1.2]
}

# Configuration for heavy augmentation
HEAVY_MIXED_RES_CONFIG = {
    'use_random_resize': True,
    'use_multi_resolution': True,
    'use_flip': True,
    'min_scale': 0.3,
    'max_scale': 2.0,
    'resolution_scales': [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
}