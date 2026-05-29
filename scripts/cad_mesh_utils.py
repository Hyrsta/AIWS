"""Shared CAD→mesh helper for scripts that run in the cadrille:latest image.

Tessellates a CadQuery/OCP compound into a trimesh.Trimesh. Centralized here so
the tessellation deflection defaults stay consistent across the CAD scripts;
callers that don't override them get the historical (0.001, 0.1) tessellation.
"""
from __future__ import annotations

import trimesh


def compound_to_mesh(compound, linear_deflection: float = 0.001, angular_deflection: float = 0.1):
    vertices, faces = compound.tessellate(linear_deflection, angular_deflection)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
