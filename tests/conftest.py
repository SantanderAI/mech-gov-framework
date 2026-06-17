# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Shared test configuration: force a headless matplotlib backend so figure
tests run without a display (CI / offline)."""

import matplotlib

matplotlib.use("Agg")
