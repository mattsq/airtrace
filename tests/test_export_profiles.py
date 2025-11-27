import types

import pytest

from airtrace.export.profiles import ExportProfile, get_profile, get_profile_for_model, list_profiles


class DummyModel:
    """Model with a non-matching name to trigger default profile selection."""

    pass


class DummyInformerModel:
    """Model class whose name should map to the informer export profile."""

    __name__ = "InformerModel"

    # Using a real class ensures ``__class__.__name__`` contains the key we expect
    # without depending on the actual implementation of the model.
    def __init__(self):
        self.__class__.__name__ = "InformerModel"


def test_get_profile_for_model_matches_by_name_and_defaults():
    informer_profile = get_profile_for_model(DummyInformerModel())
    assert isinstance(informer_profile, ExportProfile)
    assert informer_profile.name == "informer"

    default_profile = get_profile_for_model(DummyModel())
    assert default_profile.name == "default"


def test_list_profiles_returns_copy():
    profiles = list_profiles()
    profiles["custom"] = ExportProfile(
        name="custom",
        opset_version=1,
        verification_tolerance=0.0,
    )

    # Mutating the returned mapping must not affect the internal registry
    assert "custom" not in list_profiles()


def test_get_profile_validates_presence():
    all_profiles = list_profiles()
    assert set(all_profiles.keys())

    with pytest.raises(KeyError, match="Profile 'missing' not found"):
        get_profile("missing")


def test_get_profile_returns_requested_profile():
    informer_profile = get_profile("informer")

    assert isinstance(informer_profile, ExportProfile)
    assert informer_profile.name == "informer"

