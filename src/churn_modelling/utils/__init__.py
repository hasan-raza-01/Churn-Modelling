from churn_modelling.exception import CustomException
import sys, yaml, os, json, pickle, importlib
from pathlib import Path
from box import ConfigBox
from typing import Any, Dict



def create_dirs(path:str)->None:
    """creates directory if path do not exists

    Args:
        path (str): directory path for creation
    """
    try:
        os.makedirs(Path(path), exist_ok=True)
    except Exception as e:
        raise CustomException(e, sys)
    

def load_yaml(path:str)->ConfigBox:
    """reads the yaml file available in path

    Args:
        path (str): path of the yaml file

    Returns:
        ConfigBox: dict["key"] = value --------->  dict.key = value
    """
    try:
        with open(Path(path), "r") as yaml_file_obj:
            return ConfigBox(yaml.safe_load(yaml_file_obj))
    except Exception as e:
        raise CustomException(e, sys)
    

def dump_yaml(content:any, file_path:str)->None:
    """saves the yaml file with provided content

    Args:
        content (any): content for the yaml file
        path (str): path to save the file
    """
    try:
        with open(Path(file_path), "w") as file:
            yaml.safe_dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)
    
def dump_json(data:dict, path:str)->None:
    """saves the dictoanary into json file

    Args:
        data (dict): dictionary data to save in form of json
        path (str): path to save the file
    """
    try:
        # Serializing json
        json_object = json.dumps(data, default=str, indent=4)

        # Writing to sample.json
        with open(Path(path), "w") as outfile:
            outfile.write(json_object)
    except Exception as e:
        raise CustomException(e, sys)

# Helper to convert stringified class to actual class object
def string_to_class(class_str: str) -> Any:
    """Converts string like "<class 'torch.optim.adam.Adam'>" to actual class."""
    if isinstance(class_str, str) and class_str.startswith("<class ") and ">" in class_str:
        clean_str = class_str.strip("<>").replace("class ", "").replace("'", "")
        module_path, class_name = clean_str.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    return class_str

# Recursive conversion in case of nested dicts/lists
def convert_classes_recursively(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_classes_recursively(string_to_class(v)) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_classes_recursively(string_to_class(i)) for i in obj]
    else:
        return string_to_class(obj)
    
def load_json(path:str) -> Dict:
    """reads the data present inside the file provided in \'path\' variable

    Args:
        path (str): path of the json file

    Returns:
        json: json of data inside file
    """
    try:
        # Opening JSON file
        with open(Path(path), 'r') as openfile:

            # Reading from json file
            json_object = json.load(openfile)
            return convert_classes_recursively(json_object)
    except Exception as e:
        raise CustomException(e, sys)
    
def save_pickle(path:str, object:object)-> None:
    """saves the object into .h5 file

    Args:
        path (str): path to save the object
        object (object): object to be saved
    """
    try:
        with open(Path(path), "wb") as file_obj:
            pickle.dump(object, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_pickle(path:str)-> object:
    """load the object present at path with pickle and return

    Args:
        path (str): path for the object

    Returns:
        object: object at path will be returned
    """
    try:
        with open(Path(path), "rb") as file_obj:
            obj = pickle.load(file_obj)
            return obj
    except Exception as e:
        raise CustomException(e, sys)

def create_dirs(path) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise CustomException(e, sys)
    
