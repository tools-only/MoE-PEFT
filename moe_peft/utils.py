import importlib.metadata
import importlib.util
import logging
from typing import Optional, Tuple, Union, List

import torch
from packaging import version

features_persona = [
    ("Young", "Older"),
    ("Female", "Male"),
    ("High Neuroticism", "Low Neuroticism"),
    ("High Extraversion", "Low Extraversion"),
    ("High Openness", "Low Openness"),
    ("High Agreeableness", "Low Agreeableness"),
    ("High Conscientiousness", "Low Conscientiousness"),
    ("Likes a certain food", "Dislikes a certain food"),
    ("Likes a certain living environment", "Dislikes a certain living environment"),
    ("Likes sleep", "Dislikes sleep"),
    ("Aggressive investment", "Conservative investment"),
    ("Good at saving", "Bad at saving"),
    ("Concerned about physical safety", "Not concerned about physical safety"),
    ("Concerned about environmental safety", "Not concerned about environmental safety"),
    ("Prefers superficial interaction (casual, stress-free chat)", "Prefers deep interaction (discussing interests, emotional topics, etc.)"),
    ("Prefers direct communication to handle conflict", "Prefers avoidance, mediation, compromise to handle conflict"),
    ("Concise communication style", "Detailed communication style"),
    ("Strong need for a certain work environment", "Indifferent to work environment needs"),
    ("Strong need for recognition from others", "Indifferent to recognition from others"),
    ("Strong need for personal achievement", "Indifferent to personal achievement"),
    ("Likes a certain area of knowledge", "Dislikes a certain area of knowledge"),
    ("Likes a certain learning style", "Dislikes a certain learning style"),
    ("Likes a certain form of creative expression (e.g., art, writing, music)", "Dislikes a certain form of creative expression (e.g., art, writing, music)"),
    ("Strong need for Order (neatness, organization, avoiding chaos)", "Indifferent to orderliness"),
    ("Strong need for Retention (holding onto objects, unwilling to lose or change)", "Indifferent to retention (unconcerned about keeping objects)"),
    ("Strong need for Inviolacy (maintaining dignity and reputation)", "Indifferent to inviolacy (unconcerned with dignity or reputation)"),
    ("Strong need for Infavoidance (avoiding failure and embarrassment)", "Indifferent to Infavoidance (unconcerned with failure or embarrassment)"),
    ("Strong need for Counteraction (overcoming failure and obstacles)", "Indifferent to Counteraction (unconcerned with failure)"),
    ("Strong need for Seclusion (desire for isolation from others)", "Indifferent to Seclusion (does not care about isolation)"),
    ("Strong need for Dominance (controlling others through command or persuasion)", "Indifferent to Dominance (does not care about control)"),
    ("Strong need for Deference (following authority or rules)", "Indifferent to Deference (does not care about authority)"),
    ("Strong need for Autonomy (pursuing independence and self-reliance)", "Indifferent to Autonomy (does not care about independence)"),
    ("Strong need for Contrariance (pursuing uniqueness, opposing the norm)", "Indifferent to Contrariance (does not seek uniqueness)"),
    ("Strong need for Abasement (accepting blame, enjoying pain or misfortune)", "Indifferent to Abasement (does not accept blame or enjoy misfortune)"),
    ("Strong need for Aggression (controlling others through forceful means)", "Indifferent to Aggression (does not engage in aggression)"),
    ("Strong need for Affiliation (desiring close relationships)", "Indifferent to Affiliation (does not care about close relationships)"),
    ("Strong need for Rejection (isolating oneself from negatively evaluated people)", "Indifferent to Rejection (does not care about social exclusion)"),
    ("Strong need for Nurturance (caring for others, protecting them from danger)", "Indifferent to Nurturance (does not care about nurturing others)"),
    ("Strong need for Succorance (desiring help, love, and comfort from others)", "Indifferent to Succorance (does not rely on others for comfort)"),
    ("Strong need for Play (enjoying fun, relaxation, and laughter)", "Indifferent to Play (does not prioritize fun or relaxation)"),
    ("Concerned about harmlessness", "Indifferent about harmlessness"),
    ("Concerned about instruction-following", "Indifferent about instruction-following"),
    ("Concerned about honesty", "Indifferent about honesty"),
    ("Concerned about truthfulness", "Indifferent about truthfulness"),
    ("Concerned about helpfulness", "Indifferent about helpfulness"),
    ("Concerned about coherence", "Indifferent about coherence"),
    ("Concerned about complexity", "Indifferent about complexity"),
    ("Likes science", "Dislikes science"),
    ("Likes knowledge", "Dislikes knowledge"),
    ("Likes psychology", "Dislikes psychology"),
    ("Likes cinema", "Dislikes cinema"),
    ("Likes entertainment", "Dislikes entertainment"),
    ("Likes gaming", "Dislikes gaming"),
    ("Likes parenting", "Dislikes parenting"),
    ("Likes wild imagination", "Dislikes wild imagination"),
    ("Likes anime", "Dislikes anime"),
    ("Likes sports", "Dislikes sports"),
    ("Likes law", "Dislikes law"),
    ("Likes workplace", "Dislikes workplace"),
    ("Likes pets", "Dislikes pets"),
    ("Likes travel", "Dislikes travel"),
    ("Likes health", "Dislikes health"),
    ("Likes stories", "Dislikes stories"),
    ("Likes cars", "Dislikes cars"),
    ("Likes gourmet food", "Dislikes gourmet food"),
    ("Likes education", "Dislikes education"),
    ("Likes current events", "Dislikes current events"),
    ("Likes home decor", "Dislikes home decor"),
    ("Likes international", "Dislikes international"),
    ("Likes finance", "Dislikes finance"),
    ("Likes campus life", "Dislikes campus life"),
    ("Likes digital technology", "Dislikes digital technology"),
    ("Likes emotions", "Dislikes emotions"),
    ("Likes humor", "Dislikes humor"),
    ("Likes music", "Dislikes music"),
    ("Likes reading", "Dislikes reading"),
    ("Likes painting", "Dislikes painting"),
    ("Likes dance", "Dislikes dance"),
    ("Likes crafts", "Dislikes crafts"),
    ("Likes photography", "Dislikes photography"),
    ("Likes culture", "Dislikes culture"),
    ("Likes fitness", "Dislikes fitness"),
    ("Likes art", "Dislikes art"),
    ("Likes stationery and planners", "Dislikes stationery and planners"),
    ("Likes celebrities", "Dislikes celebrities"),
    ("Likes outdoors", "Dislikes outdoors"),
    ("Likes camping", "Dislikes camping"),
    ("Likes social sciences", "Dislikes social sciences"),
    ("Likes weddings", "Dislikes weddings"),
    ("Likes fashion", "Dislikes fashion")
]

def encode_persona_to_vector(persona_str):
    vector = []
    global features_persona
    for left, right in features_persona:
        if left in persona_str:
            vector.append(1.0)
        elif right in persona_str:
            vector.append(0.0)
        else:
            vector.append(0.5)
    return vector

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

def weighted_preference_mapping(preference: List[float]) -> List[float]:
    raw_scores = []
    for cat_id in range(9):
        indices = CATEGORY_TO_INDICES[cat_id]
        count = sum(1 for i in indices if preference[i] in [0, 1])
        raw_scores.append(count)

    total = sum(raw_scores)
    if total > 0:
        router_mask = [c / total for c in raw_scores]
    else:
        router_mask = [1.0 / len(raw_scores)] * len(raw_scores)

    return router_mask

def preference_mapping(preference: List[float]):
    '''
    preference: 90 dimension
    '''
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
