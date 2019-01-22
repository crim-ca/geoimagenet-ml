from src.typedefs import AnyStr, SettingDict


def get_base_url(settings):
    # type: (SettingDict) -> AnyStr
    return settings.get('src.api.url').rstrip('/').strip()
