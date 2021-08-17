import unittest.mock as mock
import pytest

from smqtk_detection.exceptions import NoDetectionError
from smqtk_detection.interfaces.detection_element import DetectionElement
from smqtk_detection.utils.bbox import AxisAlignedBoundingBox
from smqtk_classifier.interfaces.classification_element import ClassificationElement
from typing import Any, Dict, Tuple, Union


###############################################################################
# Helper classes and methods

class DummyDetectionElement (DetectionElement):
    """
    Dummy implementation for testing methods implemented in abstract parent
    class (no constructor override). Abstract methods are not implemented
    beyond declaration.
    """

    # Satisfy Pluggable ##################################

    @classmethod
    def is_usable(cls) -> bool:
        return True

    # Satisfy Configurable ###############################

    def get_config(self) -> Dict[str, Any]:
        raise NotImplementedError()

    # Satisfy DetectionElement ###########################

    def __getstate__(self) -> Dict[Any, Any]:
        raise NotImplementedError()

    def __setstate__(self, state: Dict[Any, Any]) -> None:
        raise NotImplementedError()

    def has_detection(  # type: ignore
            self) -> bool:
        raise NotImplementedError()

    def set_detection(self, bbox: AxisAlignedBoundingBox,
                      classification_element: ClassificationElement) -> DetectionElement:
        raise NotImplementedError()

    def get_bbox(self) -> None:
        raise NotImplementedError()

    def get_classification(self) -> None:
        raise NotImplementedError()

    def get_detection(  # type: ignore
            self) -> Tuple[AxisAlignedBoundingBox, ClassificationElement]:
        raise NotImplementedError()


###############################################################################
# Tests

def test_construction() -> None:
    """
    Test that normal construction sets the correct attributes
    """
    expected_uuid = 0
    m = mock.MagicMock(spec_set=DetectionElement)
    # noinspection PyCallByClass
    DetectionElement.__init__(m, expected_uuid)
    assert m._uuid == expected_uuid


def test_get_default_config_override() -> None:
    """
    Test override of get_default_config s.t. ``uuid`` is not present in the
    result dict.
    """
    default = DetectionElement.get_default_config()
    assert 'uuid' not in default


def test_from_config_override_mdFalse() -> None:
    """
    Test that ``from_config`` appropriately passes runtime-provided UUID value.
    """
    expected_ret_val = 'expected return value'
    with mock.patch('smqtk_core.configuration.Configurable.from_config') as m_confFromConfig:
        m_confFromConfig.return_value = expected_ret_val
        given_conf: dict = {}
        expected_uuid = 'test uuid'
        expected_conf = {
            'uuid': expected_uuid
        }

        DetectionElement.from_config(given_conf, expected_uuid,
                                     merge_default=False)
        m_confFromConfig.assert_called_once_with(expected_conf,
                                                 merge_default=False)


def test_from_config_override_mdTrue() -> None:
    """
    Test that ``from_config`` appropriately passes runtime-provided UUID value.
    """
    with mock.patch('smqtk_core.configuration.Configurable.from_config') as m_confFromConfig:
        given_conf: dict = {}
        expected_uuid = 'test uuid'
        expected_conf = {
            'uuid': expected_uuid
        }

        DetectionElement.from_config(given_conf, expected_uuid,
                                     merge_default=True)
        m_confFromConfig.assert_called_once_with(expected_conf,
                                                 merge_default=False)


def test_from_config_uuid_preseed_mdFalse() -> None:
    """
    Test that UUID provided at runtime prevails over any UUID provided
    through the config.
    """
    with mock.patch('smqtk_core.configuration.Configurable.from_config') as m_confFromConfig:
        given_conf = {
            "uuid": "should not get through",
        }
        expected_uuid = "actually expected UUID"
        expected_conf = {
            'uuid': expected_uuid
        }

        DetectionElement.from_config(given_conf, expected_uuid,
                                     merge_default=False)
        m_confFromConfig.assert_called_once_with(expected_conf,
                                                 merge_default=False)


def test_from_config_uuid_preseed_mdTrue() -> None:
    """
    Test that UUID provided at runtime prevails over any UUID provided
    through the config.
    """
    with mock.patch('smqtk_core.configuration.Configurable.from_config') as m_confFromConfig:
        given_conf = {
            "uuid": "should not get through",
        }
        expected_uuid = "actually expected UUID"
        expected_conf = {
            'uuid': expected_uuid
        }

        DetectionElement.from_config(given_conf, expected_uuid,
                                     merge_default=True)
        m_confFromConfig.assert_called_once_with(expected_conf,
                                                 merge_default=False)


def test_hash() -> None:
    """
    Test that a DetectionElement is hashable based on solely on UUID.
    """
    with pytest.raises(TypeError, match="unhashable type"):
        hash(DummyDetectionElement(0))


def test_eq_both_no_detections() -> None:
    """
    Test that two elements with no detection info set are considered not equal.
    """
    d1 = DummyDetectionElement(0)
    d2 = DummyDetectionElement(1)
    d1.get_detection = d2.get_detection = mock.MagicMock(side_effect=NoDetectionError)  # type: ignore
    assert (d1 == d2) is False
    assert (d2 == d1) is False
    assert (d1 != d2) is True
    assert (d2 != d1) is True


def test_eq_one_no_detection() -> None:
    """
    Test that when one element has no detection info then they are considered
    NOT equal.
    """
    d_without = DummyDetectionElement(0)
    d_without.get_detection = mock.MagicMock(side_effect=NoDetectionError)  # type: ignore
    d_with = DummyDetectionElement(1)
    d_with.get_detection = mock.MagicMock(return_value=(1, 2))  # type: ignore

    assert (d_with == d_without) is False
    assert (d_without == d_with) is False
    assert (d_with != d_without) is True
    assert (d_without != d_with) is True


def test_eq_unequal_detections() -> None:
    """
    Test that two detections, with valid, but different contents, test out not
    equal.
    """
    d1 = DummyDetectionElement(0)
    d2 = DummyDetectionElement(1)
    d1.get_detection = mock.Mock(return_value=('a', 1))  # type: ignore
    d2.get_detection = mock.Mock(return_value=('b', 2))  # type: ignore
    assert (d1 == d2) is False


def test_eq_unequal_just_one() -> None:
    """
    Test inequality where just one of the two sub-components of detections (bb,
    classification) are different.
    """
    d1 = DummyDetectionElement(0)
    d2 = DummyDetectionElement(1)

    d1.get_detection = mock.Mock(return_value=('a', 1))  # type: ignore
    d2.get_detection = mock.Mock(return_value=('a', 2))  # type: ignore
    assert (d1 == d2) is False

    d1.get_detection = mock.Mock(return_value=('a', 1))  # type: ignore
    d2.get_detection = mock.Mock(return_value=('b', 1))  # type: ignore
    assert (d1 == d2) is False


def test_eq_success() -> None:
    """
    Test when two different detection instances returns the same value pair
    from ``get_detection()``.
    """
    d1 = DummyDetectionElement(0)
    d2 = DummyDetectionElement(1)
    d1.get_detection = d2.get_detection = mock.MagicMock(return_value=('a', 0))  # type: ignore
    assert d1 == d2


def test_nonzero_has_detection() -> None:
    """
    Test that boolean cast of a DetectionElement occurs appropriately when the
    element has a detection.
    """
    expected_val = True
    inst = DummyDetectionElement(0)
    inst.has_detection = mock.MagicMock(return_value=expected_val)  # type: ignore
    assert bool(inst) is expected_val
    inst.has_detection.assert_called_once_with()


def test_nonzero_no_detection() -> None:
    """
    Test that boolean cast of a DetectionElement occurs appropriately when the
    element has a detection.
    """
    expected_val = False
    inst = DummyDetectionElement(0)
    inst.has_detection = mock.MagicMock(return_value=expected_val)  # type: ignore
    assert bool(inst) is expected_val
    inst.has_detection.assert_called_once_with()


def test_property_uuid() -> None:
    """
    Test that given UUID hashable is returned via `uuid` property.
    """
    expected_uuid: Union[str, int] = 0
    assert DummyDetectionElement(expected_uuid).uuid == expected_uuid

    expected_uuid = 'a hashable string'
    assert DummyDetectionElement(expected_uuid).uuid == expected_uuid


def test_getstate() -> None:
    """
    Test that expected "state" representation is returned from __getstate__.
    """
    expected_uuid = 'expected-uuid'
    expected_state = {
        '_uuid': expected_uuid
    }

    # Mock an instance of DetectionElement with expected uuid attribute set.
    m = mock.MagicMock(spec_set=DetectionElement)
    m._uuid = expected_uuid

    actual_state = DetectionElement.__getstate__(m)
    assert actual_state == expected_state


def test_setstate() -> None:
    """
    Test that __setstate__ base implementation sets the correct instance
    attributes.
    """
    expected_uuid = 'expected_uuid'
    expected_state = {
        '_uuid': expected_uuid
    }

    # Mock an instance of DetectionElement
    m = mock.MagicMock(spec_set=DetectionElement)

    # noinspection PyCallByClass
    # - for testing purposes.
    DetectionElement.__setstate__(m, expected_state)
    assert m._uuid == expected_uuid
