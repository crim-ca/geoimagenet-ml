from geoimagenet_ml.typedefs import AnyStr, SettingDict


def get_base_url(settings):
    # type: (SettingDict) -> AnyStr
    return settings.get('geoimagenet_ml.api.url').rstrip('/').strip()
