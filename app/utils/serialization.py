from typing import Any, Dict, List


def ensure_serializable(data: Any) -> Any:
    """
    Ensure that the data structure is serializable by converting any unhashable keys
    to their string representation.

    Args:
        data: The data to be serialized

    Returns:
        Serializable version of the data
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Convert unhashable keys to strings
            if isinstance(key, (list, dict)):
                key = str(key)
            result[key] = ensure_serializable(value)
        return result
    elif isinstance(data, list):
        return [ensure_serializable(item) for item in data]
    elif hasattr(data, "dict") and callable(getattr(data, "dict")):
        # Handle Pydantic models
        return ensure_serializable(data.dict())
    else:
        return data


def clean_response(response_data: Any) -> Any:
    """
    Clean a response object to make it serializable.

    Args:
        response_data: The response data to clean

    Returns:
        Cleaned response data ready for serialization
    """
    return ensure_serializable(response_data)
