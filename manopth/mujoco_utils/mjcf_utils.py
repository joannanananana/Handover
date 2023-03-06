import xml.etree.ElementTree as ET
import numpy as np
import mujoco_py

RED = [1, 0, 0, 1]
GREEN = [0, 1, 0, 1]
BLUE = [0, 0, 1, 1]


def read_standard_xml(xml_file):
    model = mujoco_py.load_model_from_path(xml_file)
    return ET.fromstring(model.get_xml())


def array_to_string(array):
    return " ".join(["{}".format(x) for x in array])


def string_to_array(string):
    return np.array([float(x) for x in string.split(" ")])


def set_alpha(node, alpha=0.1):
    for child_node in node.findall(".//*[@rgba]"):
        rgba_orig = string_to_array(child_node.get("rbga"))
        child_node.set("rgba", array_to_string(list(rgba_orig[0:3]) + [alpha]))


def new_joint(**kwargs):
    element = ET.Element("joint", attrib=kwargs)
    return element


def new_actuator(joint, act_type="actuator", **kwargs):
    element = ET.Element(act_type, attrib=kwargs)
    element.set("joint", joint)
    return element


def new_site(name, rgba=RED, pos=(0, 0, 0), size=(0.005,), **kwargs):
    kwargs["rgba"] = array_to_string(rgba)
    kwargs["pos"] = array_to_string(pos)
    kwargs["size"] = array_to_string(size)
    kwargs["name"] = name
    element = ET.Element("site", attrib=kwargs)
    return element


def new_geom(geom_type, size, pos=(0, 0, 0), rgba=RED, group=0, **kwargs):
    kwargs["type"] = str(geom_type)
    kwargs["size"] = array_to_string(size)
    kwargs["rgba"] = array_to_string(rgba)
    kwargs["group"] = str(group)
    kwargs["pos"] = array_to_string(pos)
    element = ET.Element("geom", attrib=kwargs)
    return element


def new_body(name=None, pos=None, quat=None, **kwargs):
    if name:
        kwargs["name"] = name
    if pos:
        kwargs["pos"] = array_to_string(pos)
    if quat:
        kwargs["quat"] = array_to_string(quat)
    element = ET.Element("body", attrib=kwargs)
    return element


def new_inertial(name=None, pos=(0, 0, 0), mass=None, **kwargs):
    if name:
        kwargs["name"] = name
    if mass:
        kwargs["mass"] = str(mass)
    kwargs["pos"] = array_to_string(pos)
    element = ET.Element("inertial", attrib=kwargs)
    return element
