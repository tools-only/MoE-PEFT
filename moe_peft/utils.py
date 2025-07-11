import importlib.metadata
import importlib.util
import logging
from typing import Optional, Tuple, Union

import torch
from packaging import version
from typing import List

def preference_mapping(preference: List[float]):
    '''
    preference: 90 dimension
    '''
    CATEGORY_TO_INDICES = {
        0: list(range(2, 14)) + [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],  # 人格特质
        1: list(range(35, 47)),  # 动机与社会需求
        2: [15, 16, 17],  # 沟通与人际互动
        3: [18, 42, 43, 44, 45, 46, 47, 48],  # 工作倾向
        4: [11, 12, 13, 14],  # 风险安全
        5: [7, 8, 9] + [58, 59, 60, 61, 62, 63, 64, 65, 66, 67],  # 日常生活
        6: [21, 22] + [49, 50, 51, 52, 53, 54, 55, 56, 57],  # 学习知识
        7: [20] + list(range(68, 89)),  # 美学创意
        8: [0, 1],  # 人口统计
    }

    router_mask = []
    for cat_id in range(9):
        indices = CATEGORY_TO_INDICES[cat_id]
        activated = any(preference[i] in [0, 1] for i in indices)
        router_mask.append(int(activated))
    return router_mask


def copy_parameters(source: torch.nn.Module, dest: torch.nn.Module):
    dest.load_state_dict(source.state_dict())
    dest.requires_grad_(False)


def setup_logging(log_level: str = "WARN", log_file: str = None):
    # set the logger
    log_handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="[%(asctime)s] MoE-PEFT: %(message)s",
        level=log_level,
        handlers=log_handlers,
        force=True,
    )


def is_package_available(
    pkg_name: str, pkg_version: Optional[str] = None
) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logging.debug(f"Detected {pkg_name} version {package_version}")
    if pkg_version is not None:
        return package_exists and version.parse(package_version) >= version.parse(
            pkg_version
        )
    else:
        return package_exists


class Unsubscribable:
    def __init__(self) -> None:
        raise RuntimeError(f"Instant unsubscribable class {__class__}")


# Class Placeholder for Bitsandbytes
class Linear8bitLt(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class Linear4bit(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class BitsAndBytesConfig:
    def __init__(self, **kwargs) -> None:
        raise RuntimeError("Quantization not supported.")


class NoneContexts(object):
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass
