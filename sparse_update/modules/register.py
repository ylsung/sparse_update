PLMODULE_REGISTER = {}


def register(class_name):
    """Register the class by its `name` in PLMODULE_REGISTER"""
    PLMODULE_REGISTER[class_name.name] = class_name

    return class_name
