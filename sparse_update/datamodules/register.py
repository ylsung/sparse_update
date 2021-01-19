DATAMODULE_REGISTER = {}


def register(class_name):
    """Register the class by its `name` in DATAMODULE_REGISTER"""
    DATAMODULE_REGISTER[class_name.name] = class_name

    return class_name
