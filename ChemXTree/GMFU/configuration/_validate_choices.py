# GMFU validation choices
# Reconstructor: Star <Star@seas.hahaha.edu>
# For license information, see LICENSE.TXT
""" Validate choices"""
def _validate_choices(cls):
    for key in cls.__dataclass_fields__.keys():
        atr = cls.__dataclass_fields__[key]
        if atr.init:
            if "choices" in atr.metadata.keys():
                if getattr(cls, key) not in atr.metadata.get("choices"):
                    raise ValueError(
                        f"{getattr(cls, key)} is not a valid choice for {key}. Please choose from on of the following: {atr.metadata['choices']}"
                    )