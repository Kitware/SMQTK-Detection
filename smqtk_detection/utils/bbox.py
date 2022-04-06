# Support previous use-case where this module defined `AxisAlignedBoundingBox`
# in duplication with the `smqtk_image_io` package. New users should instead
# just import the following instead of this module.
from smqtk_image_io import AxisAlignedBoundingBox  # noqa: F401
