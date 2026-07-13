"""FastAPI application: thin HTTP/presentation layer.

Routes:
  GET /                      homepage roster, grouped by family surname
  GET /person/{person_id}    every photo a person appears in, by age
  GET /person/{id}/workflow/{share}/{name}  pipeline walkthrough for one photo
  GET /media/portrait/{id}   the person's portrait image (cropped/curated)
  GET /media/photo/{share}/{name}  the cropped-print photo
  GET /media/raw/{share}/{name}     the original downloaded photo
  GET /media/rotated/{share}/{name} the orientation-corrected photo
  GET /media/facecrop/{share}/{name}/{index}  a single detected face crop

All domain work is delegated to the repository / resolver / presenter; the
handlers only translate between HTTP and those services.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pipeline.shared.paths import DATA_DIR, STEPS_DIR

from web.config import FAMILY_TREE_ENABLED, FAMILY_TREE_SPACING
from web.family_tree import load_family_tree
from web.i18n import Locale, Translator, get_translator
from web.kinship import RelationshipResolver
from web.portraits import PortraitService
from web.presenter import Presenter
from web.repository import Repository
from web.workflow import WorkflowService

DEFAULT_VIEWER = "I0004"  # default "Wer bist du?" viewer (Ève Curie in the quickstart tree)

_HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))

# Build the read-only index once at startup; the photo collection is static
# within a run and re-reading on every request would be wasteful.
_tree = load_family_tree()
_repo = Repository(_tree)
_resolver = RelationshipResolver(_tree)
_portraits = PortraitService()
_presenter = Presenter(_repo, _resolver, _portraits, _tree)
_workflow = WorkflowService(_tree)

app = FastAPI(title="Ancestry Photos")
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


def _viewer_id(request: Request) -> str:
    """Resolve the chosen 'who are you' person, falling back to the default."""
    candidate = request.query_params.get("you", DEFAULT_VIEWER)
    return candidate if _repo.profile(candidate) or candidate == DEFAULT_VIEWER else DEFAULT_VIEWER


def _lang_switch(request: Request, current: Locale) -> list[dict]:
    """One entry per locale for the header switcher: a link to the *current* page
    with `lang` swapped, so switching language keeps you where you are."""
    return [
        {
            "label": loc.value.upper(),
            "url": str(request.url.include_query_params(lang=loc.value)),
            "active": loc == current,
        }
        for loc in Locale
    ]


def _page(request: Request, t: Translator, viewer_id: str, active_view: str, **extra) -> dict:
    """Common template context: the translator (`_`), the active locale, and the
    chrome bits every page needs (viewer picker, language switcher, nav state).
    `nav_qs` is the persisted query string every internal link carries so `you`
    and `lang` survive navigation."""
    locale = t.locale.value
    return {
        "_": t,
        "locale": locale,
        "viewer_id": viewer_id,
        "viewer_options": _presenter.viewer_options(),
        "family_tree_enabled": FAMILY_TREE_ENABLED,
        "active_view": active_view,
        "nav_qs": f"you={viewer_id}&lang={locale}",
        "lang_switch": _lang_switch(request, t.locale),
        **extra,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request, t: Translator = Depends(get_translator)) -> HTMLResponse:
    viewer_id = _viewer_id(request)
    return templates.TemplateResponse(
        request,
        "index.html",
        _page(request, t, viewer_id, "roster", groups=_presenter.home_groups(viewer_id, t)),
    )


@app.get("/tree", response_class=HTMLResponse)
def family_tree(request: Request, t: Translator = Depends(get_translator)) -> HTMLResponse:
    if not FAMILY_TREE_ENABLED:
        raise HTTPException(status_code=404, detail=t("error.tree_disabled"))
    viewer_id = _viewer_id(request)
    return templates.TemplateResponse(
        request,
        "tree.html",
        _page(
            request,
            t,
            viewer_id,
            "tree",
            graph=_presenter.family_tree_graph(viewer_id, t.locale.value),
            spacing=FAMILY_TREE_SPACING,
        ),
    )


@app.get("/person/{person_id}", response_class=HTMLResponse)
def person(
    request: Request, person_id: str, t: Translator = Depends(get_translator)
) -> HTMLResponse:
    profile = _repo.profile(person_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=t("error.unknown_person"))
    viewer_id = _viewer_id(request)
    has_portrait = (
        _portraits.resolve(profile.person, _repo.portrait_source(person_id)) is not None
    )
    return templates.TemplateResponse(
        request,
        "person.html",
        _page(
            request,
            t,
            viewer_id,
            "roster",
            profile=profile,
            relationship=_presenter.relationship(viewer_id, person_id, t),
            has_portrait=has_portrait,
        ),
    )


@app.get("/person/{person_id}/workflow/{share}/{name}", response_class=HTMLResponse)
def person_workflow(
    request: Request,
    person_id: str,
    share: str,
    name: str,
    t: Translator = Depends(get_translator),
) -> HTMLResponse:
    """Pipeline walkthrough for one photo, rendered as a modal fragment."""
    if _repo.profile(person_id) is None:
        raise HTTPException(status_code=404, detail=t("error.unknown_person"))
    workflow = _workflow.build(f"{share}/{name}", person_id, t)
    return templates.TemplateResponse(
        request, "workflow.html", {"workflow": workflow, "_": t}
    )


@app.get("/media/portrait/{person_id}")
def portrait_media(
    person_id: str, t: Translator = Depends(get_translator)
) -> FileResponse:
    profile = _repo.profile(person_id)
    if profile is None:
        raise HTTPException(status_code=404)
    path = _portraits.resolve(profile.person, _repo.portrait_source(person_id))
    if path is None:
        raise HTTPException(status_code=404, detail=t("error.no_portrait"))
    return _serve(path, t("error.image_not_found"))


@app.get("/media/photo/{share}/{name}")
def photo_media(
    share: str, name: str, t: Translator = Depends(get_translator)
) -> FileResponse:
    return _serve(_repo.frame_image_path(f"{share}/{name}"), t("error.photo_not_found"))


@app.get("/media/raw/{share}/{name}")
def raw_media(
    share: str, name: str, t: Translator = Depends(get_translator)
) -> FileResponse:
    return _serve(DATA_DIR / "raw" / share / f"{name}.jpg", t("error.image_not_found"))


@app.get("/media/rotated/{share}/{name}")
def rotated_media(
    share: str, name: str, t: Translator = Depends(get_translator)
) -> FileResponse:
    return _serve(STEPS_DIR / "rotate" / share / f"{name}.jpg", t("error.image_not_found"))


@app.get("/media/facecrop/{share}/{name}/{index}")
def facecrop_media(
    share: str, name: str, index: int, t: Translator = Depends(get_translator)
) -> FileResponse:
    return _serve(
        STEPS_DIR / "face_crop" / share / name / f"face_{index:02d}.jpg",
        t("error.image_not_found"),
    )


def _serve(path: Path, not_found: str) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail=not_found)
    # no-store: media URLs like /media/portrait/I0004 collide across data roots
    # (person ids repeat between the real tree and the quickstart Curie tree), so
    # a browser cache keyed on URL would serve one dataset's image for another's.
    # This is a local viewer — caching buys nothing and correctness matters.
    return FileResponse(path, headers={"Cache-Control": "no-store"})
