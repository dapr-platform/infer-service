
from typing import List, Tuple, Dict, Any
import inspect, logging

def logFunction():
    frame_records = inspect.stack()[1]
    frame = frame_records[0]
    info = inspect.getframeinfo(frame)
    logging.info(f"Function: {info.function}, File: {os.path.basename(info.filename)}, Line: {info.lineno}")

def getPolygonFromConfiguration(configurations: List[Dict[str, any]]):
    polygon = None
    if configurations is not None:
        filtered_configs = [config for config in configurations if config.get("name") == "polygonPoints"]
    else:
        filtered_configs = []

    if len(filtered_configs) > 0:
        config = filtered_configs[0]
        polygon = [[point["x"], point["y"]] for point in config.get("value", [])]
    return polygon

def getItemValueFromConfiguration(configurations: List[Dict[str, any]], name: str, defaultValue: Any):
    if configurations is not None:
        filtered_configs = [config for config in configurations if config.get("name") == name]
    else:
        filtered_configs = []

    if len(filtered_configs) > 0:
        config = filtered_configs[0]
        if config.get("value") is None:
            return defaultValue
        return config.get("value")
    return defaultValue
def get_absolute_polygon(polygon, width, height):
    return [[int(point[0] * width), int(point[1] * height)] for point in polygon]