# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""
import os
FAIVE_ASSETS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "data")

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .assets import *


