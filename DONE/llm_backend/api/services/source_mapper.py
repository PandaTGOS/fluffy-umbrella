from typing import Iterable

BASE_DOC_URL = f"http://ourspaceaes.aerord.internal/Pages/EC/EC%20Arbetstid%20ledighet%20och%20franvaro"

def map_sources(filenames: Iterable[str]):
    sources = []

    for filename in filenames:
        name = filename.rsplit(".", 1)[0]
        label = name.replace("_", " ")
        url = f"{BASE_DOC_URL}/ec_{name}.aspx"

        sources.append({
            "id": name,
            "label": label,
            "url": url
        })

    return sources