import abc

from smqtk_core import Plugfigurable
from smqtk_detection.exceptions import NoDetectionError
from smqtk_core.dict import merge_dict
from typing import Hashable
from smqtk_classifier.interfaces.classification_element import ClassificationElement
from smqtk_image_io import AxisAlignedBoundingBox
from typing import Dict, Any, Tuple, Type, Optional


class DetectionElement (Plugfigurable):
    """
    Representation of a spatial detection.
    """

    __slots__ = ('_uuid',)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        # Override from Configurable.
        default = super(DetectionElement, cls).get_default_config()
        # Remove runtime positional argument(s).
        del default['uuid']
        return default

    # noinspection PyMethodOverriding
    @classmethod
    def from_config(  # type: ignore
            cls: Type["DetectionElement"],
            config_dict: Dict[Any, Any],
            uuid: Hashable,
            merge_default: bool = True
            ) -> "DetectionElement":
        """
        Override of
        :meth:`smqtk.utils.configuration.Configurable.from_config` with the
        added runtime argument ``uuid``. See parent method documentation for
        details.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :param collections.abc.Hashable uuid:
            UUID to assign to the produced DetectionElement.

        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.
        :type merge_default: bool

        :return: Constructed instance from the provided config.
        :rtype: DetectionElement

        """
        # Override from Configurable
        # Handle passing of runtime positional argument(s).
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)
        config_dict['uuid'] = uuid
        return super(DetectionElement, cls).from_config(config_dict,
                                                        merge_default=False)

    def __init__(self, uuid: Hashable) -> None:
        """
        Initialize a new detection element with the given ``uuid``.

        All DetectionElement classes will take a ``uuid`` parameter as the
        first positional argument. This parameter is not configurable and is
        only specified at runtime. Implementing classes should not include
        ``uuid`` in ``get_config`` returns.

        :param collections.abc.Hashable uuid:
            Unique ID reference of the detection.

        """
        super(DetectionElement, self).__init__()
        self._uuid = uuid

    __hash__ = None  # type: ignore

    def __eq__(self, other: Any) -> bool:
        """
        Equality of two detections is defined by their equal spatial overlap
        AND their equivalent classification.

        When one element does not contain detection information but the other
        does, the two elements are of course considered NOT equal.
        If *neither* elements contain detection information, they are defined
        as NOT equal (undefined).

        :param DetectionElement other: Other detection element.
        :return: True if the two detections are equal in spacial overlap and
            classification.
        """
        try:
            s_bb, s_ce = self.get_detection()
            o_bb, o_ce = other.get_detection()
            return s_bb == o_bb and s_ce == o_ce
        except NoDetectionError:
            return False

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __repr__(self) -> str:
        # using "{{...}}" to skip .format activation.
        return "{:s}{{uuid: {}}}".format(self.__class__.__name__, self._uuid)

    def __nonzero__(self) -> bool:
        """
        A DetectionElement is considered non-zero if ``has_detection`` returns
        True. See method documentation for details.

        :return: True if this instance is non-zero (see above), false
            otherwise.
        :rtype: bool
        """
        return self.has_detection()

    __bool__ = __nonzero__

    @property
    def uuid(self) -> Hashable:
        return self._uuid

    #
    # Abstract methods
    #

    @abc.abstractmethod
    def __getstate__(self) -> dict:
        return {
            '_uuid': self._uuid,
        }

    @abc.abstractmethod
    def __setstate__(self, state: dict) -> None:
        self._uuid = state['_uuid']

    @abc.abstractmethod
    def has_detection(self) -> bool:
        """
        :return: Whether or not this container currently contains a valid
            detection bounding box and classification element (must be
            non-zero).
        :rtype: bool
        """

    @abc.abstractmethod
    def get_bbox(self) -> Optional[AxisAlignedBoundingBox]:
        """
        :return: The spatial bounding box of this detection.
        :rtype: smqtk.representation.AxisAlignedBoundingBox

        :raises NoDetectionError: No detection AxisAlignedBoundingBox set yet.
        """

    @abc.abstractmethod
    def get_classification(self) -> Optional[ClassificationElement]:
        """
        :return: The classification element of this detection.
        :rtype: smqtk.representation.ClassificationElement

        :raises NoDetectionError: No detection ClassificationElement set yet or
            the element is empty.
        """

    @abc.abstractmethod
    def get_detection(self) -> Tuple[AxisAlignedBoundingBox, ClassificationElement]:
        """
        :return: The paired spatial bounding box and classification element of
            this detection.
        :rtype: (smqtk.representation.AxisAlignedBoundingBox,
                 smqtk.representation.ClassificationElement)

        :raises NoDetectionError: No detection AxisAlignedBoundingBox and
            ClassificationElement set yet.

        """

    @abc.abstractmethod
    def set_detection(self, bbox: AxisAlignedBoundingBox, classification_element: ClassificationElement) \
            -> "DetectionElement":
        """
        Set a bounding box and classification element to this detection
        element.

        :param smqtk.representation.AxisAlignedBoundingBox bbox:
            Spatial bounding box instance.

        :param smqtk.representation.ClassificationElement classification_element:
            The classification of this detection.

        :raises ValueError: No, or invalid, AxisAlignedBoundingBox or
            ClassificationElement was provided.

        :returns: Self
        :rtype: DetectionElement

        """
