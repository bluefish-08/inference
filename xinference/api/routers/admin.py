"""Admin / cluster / infrastructure route registration and handlers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import uuid
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

import aiohttp
from fastapi import Body, Depends, HTTPException, Query, Request, Security
from pydantic import BaseModel

from ... import __version__
from ...constants import XINFERENCE_TOKEN_ROUTER_ENABLED
from ...core.virtual_env_manager import VirtualEnvConflictError
from ...types import PeftModelConfig
from ..dependencies import get_api
from ..responses import JSONResponse

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI

logger = logging.getLogger(__name__)


# --- Handlers (top-level, inject dependencies via Depends) ---


async def get_status(api: "RESTfulAPI" = Depends(get_api)) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.get_status()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_address(api: "RESTfulAPI" = Depends(get_api)) -> JSONResponse:
    return JSONResponse(content=api._supervisor_address)


async def is_cluster_authenticated(
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    return JSONResponse(content={"auth": api.is_authenticated()})


async def get_cluster_device_info(
    api: "RESTfulAPI" = Depends(get_api),
    detailed: bool = Query(False),
    include_routers: bool = Query(False),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.get_cluster_device_info(
            detailed=detailed, include_routers=include_routers
        )
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_cluster_version() -> JSONResponse:
    try:
        # keep the response keys the versioneer-era API exposed; the build
        # backend records the full 40-character SHA in _commit.py, with the
        # setuptools-scm short node as a fallback for artifacts built without
        # git metadata
        try:
            from ..._commit import full_revisionid
        except ImportError:
            try:
                from ..._version import commit_id
            except ImportError:
                commit_id = None
            full_revisionid = commit_id.lstrip("g") if commit_id else None
        data = {
            "version": __version__,
            "full-revisionid": full_revisionid,
        }
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_devices_count(
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.get_devices_count()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_workers_info(
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        res = await supervisor_ref.get_workers_info()
        return JSONResponse(content=res)
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_supervisor_info(
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        res = await supervisor_ref.get_supervisor_info()
        return JSONResponse(content=res)
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def abort_cluster(
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        res = await supervisor_ref.abort_cluster()
        os.kill(os.getpid(), signal.SIGINT)
        return JSONResponse(content={"result": res})
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def list_cached_models(
    api: "RESTfulAPI" = Depends(get_api),
    model_name: str = Query(None),
    worker_ip: str = Query(None),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.list_cached_models(model_name, worker_ip)
        resp = {"list": data}
        return JSONResponse(content=resp)
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def cache_model(
    request: Request,
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    """Download model artifacts into one worker cache without deployment."""
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Invalid input. Expected a JSON object.")
        internal_keys = sorted(key for key in payload if key.startswith("_"))
        if internal_keys:
            raise ValueError(
                "Invalid input. Internal fields are not allowed: "
                + ", ".join(internal_keys)
            )

        cache_uid = payload.get("cache_uid") or str(uuid.uuid4())
        model_name = payload.get("model_name")
        model_type = payload.get("model_type", "LLM")
        model_engine = payload.get("model_engine")
        if not isinstance(cache_uid, str):
            raise ValueError("Invalid input. `cache_uid` must be a string.")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Invalid input. Please specify the `model_name` field.")
        if not isinstance(model_type, str) or not model_type:
            raise ValueError("Invalid input. `model_type` must be a string.")
        model_type = "LLM" if model_type.lower() == "llm" else model_type.lower()
        if not model_engine and model_type == "LLM":
            raise ValueError("Invalid input. Please specify the `model_engine` field.")

        peft_model_config = payload.get("peft_model_config")
        if peft_model_config is not None:
            peft_model_config = PeftModelConfig.from_dict(peft_model_config)

        explicit_keys = {
            "cache_uid",
            "model_name",
            "model_type",
            "model_engine",
            "model_size_in_billions",
            "model_format",
            "quantization",
            "peft_model_config",
            "worker_ip",
            "download_hub",
            "model_path",
            "enable_virtual_env",
            "virtual_env_packages",
        }
        # Keep deployment-only controls out of the model constructor while
        # forwarding model-specific artifact choices such as draft,
        # multimodal-projector, ControlNet, GGUF, and lightning options.
        deployment_only_keys = {
            "model_uid",
            "replica",
            "replica_config",
            "replica_placement_mode",
            "n_gpu",
            "gpu_idx",
            "n_gpu_layers",
            "n_worker",
            "request_limits",
            "enable_thinking",
            "reasoning_content",
            "cpu_offload",
            "quantization_config",
            "num_speculative_tokens",
            "envs",
            "virtual_env_find_links",
            "save_autostart",
        }
        kwargs = {
            key: value
            for key, value in payload.items()
            if key not in explicit_keys and key not in deployment_only_keys
        }

        supervisor_ref = await api._get_supervisor_ref()
        result = await supervisor_ref.cache_builtin_model(
            cache_uid=cache_uid,
            model_name=model_name,
            model_size_in_billions=payload.get("model_size_in_billions"),
            model_format=payload.get("model_format"),
            quantization=payload.get("quantization"),
            model_engine=model_engine,
            model_type=model_type,
            peft_model_config=peft_model_config,
            worker_ip=payload.get("worker_ip"),
            download_hub=payload.get("download_hub"),
            model_path=payload.get("model_path"),
            enable_virtual_env=payload.get("enable_virtual_env"),
            virtual_env_packages=payload.get("virtual_env_packages"),
            **kwargs,
        )
        return JSONResponse(content=result)
    except ValueError as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except asyncio.CancelledError:
        logger.info("Cache operation was cancelled")
        raise HTTPException(status_code=499, detail="Download cancelled")
    except RuntimeError as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_cache_model_progress(
    cache_uid: str,
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        result = await supervisor_ref.get_cache_builtin_model_progress_details(
            cache_uid
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def cancel_cache_model(
    cache_uid: str,
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        await supervisor_ref.cancel_cache_builtin_model(cache_uid)
        return JSONResponse(content=None)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def delete_cache_download(
    cache_uid: str,
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        result = await supervisor_ref.delete_cache_builtin_model(cache_uid)
        return JSONResponse(content=result)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def pause_cache_model(
    cache_uid: str,
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        result = await supervisor_ref.pause_cache_builtin_model(cache_uid)
        return JSONResponse(content=result)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def resume_cache_model(
    cache_uid: str,
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        result = await supervisor_ref.resume_cache_builtin_model(cache_uid)
        return JSONResponse(content=result)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def list_model_downloads(
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.list_model_downloads()
        return JSONResponse(content={"list": data})
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def list_model_files(
    api: "RESTfulAPI" = Depends(get_api),
    model_version: str = Query(None),
    worker_ip: str = Query(None),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.list_deletable_models(model_version, worker_ip)
        response = {
            "model_version": model_version,
            "worker_ip": worker_ip,
            "paths": data,
        }
        return JSONResponse(content=response)
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def confirm_and_remove_model(
    api: "RESTfulAPI" = Depends(get_api),
    model_version: str = Query(None),
    worker_ip: str = Query(None),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        res = await supervisor_ref.confirm_and_remove_model(
            model_version=model_version, worker_ip=worker_ip
        )
        return JSONResponse(content={"result": res})
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def list_virtual_envs(
    api: "RESTfulAPI" = Depends(get_api),
    model_name: str = Query(None),
    model_engine: str = Query(None),
    worker_ip: str = Query(None),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.list_virtual_envs(
            model_name, model_engine, worker_ip
        )
        resp = {"list": data}
        return JSONResponse(content=resp)
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def remove_virtual_env(
    api: "RESTfulAPI" = Depends(get_api),
    model_name: str = Query(None),
    model_engine: str = Query(None),
    python_version: str = Query(None),
    worker_ip: str = Query(None),
) -> JSONResponse:
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name parameter is required")
    try:
        supervisor_ref = await api._get_supervisor_ref()
        res = await supervisor_ref.remove_virtual_env(
            model_name=model_name,
            model_engine=model_engine,
            python_version=python_version,
            worker_ip=worker_ip,
        )
        return JSONResponse(content={"result": res})
    except VirtualEnvConflictError as e:
        logger.warning(e)
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_progress(
    request_id: str,
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        result = {"progress": await supervisor_ref.get_progress(request_id)}
        return JSONResponse(content=result)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_ui_config(request: Request) -> JSONResponse:
    store = request.app.state.monitor_config_store
    mon = store.get_all()
    dashboards = store.get_dashboards()

    return JSONResponse(
        content={
            "grafana_url": mon["grafana_url"],
            "grafana_datasource": mon["grafana_datasource"],
            "grafana_alert_datasource": mon["grafana_alert_datasource"],
            "grafana_dashboard_uid": dashboards.get("overview", "xinference-overview"),
            "grafana_dashboards": dashboards,
            "grafana_dashboards_configured": store.get_configured_dashboard_keys(),
            "cluster_name": mon["cluster_name"],
            "es_enabled": bool(os.environ.get("XINFERENCE_ES_URL", "")),
            "auth_advanced": os.environ.get("XINFERENCE_AUTH_ADVANCED", "true").lower()
            not in ("0", "false", "no"),
            "oidc_enabled": os.environ.get("XINFERENCE_OIDC_ENABLED", "").lower()
            in ("1", "true", "yes"),
            "token_router_enabled": XINFERENCE_TOKEN_ROUTER_ENABLED,
        }
    )


class MonitorConfigUpdate(BaseModel):
    grafana_url: Optional[str] = None
    grafana_datasource: Optional[str] = None
    grafana_alert_datasource: Optional[str] = None
    cluster_name: Optional[str] = None
    grafana_dashboards: Optional[Dict[str, str]] = None


class CheckGrafanaRequest(BaseModel):
    grafana_url: str


async def get_monitor_config(request: Request) -> JSONResponse:
    store = request.app.state.monitor_config_store
    all_cfg = store.get_all()
    sources = store.get_sources()
    dashboards = store.get_dashboards()

    return JSONResponse(
        content={
            "grafana_url": all_cfg["grafana_url"],
            "grafana_datasource": all_cfg["grafana_datasource"],
            "grafana_alert_datasource": all_cfg["grafana_alert_datasource"],
            "cluster_name": all_cfg["cluster_name"],
            "grafana_dashboards": dashboards,
            "grafana_dashboards_configured": store.get_configured_dashboard_keys(),
            "sources": sources,
        }
    )


async def update_monitor_config(
    request: Request,
    body: MonitorConfigUpdate = Body(...),
) -> JSONResponse:
    store = request.app.state.monitor_config_store

    username = ""
    advanced_auth = getattr(request.app.state, "advanced_auth", None)
    if advanced_auth:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                payload = advanced_auth.verify_access_token(auth_header[7:])
                username = payload.get("sub", "")
            except Exception:
                pass

    data = body.model_dump(exclude_none=True)
    updates = {}
    for field in (
        "grafana_url",
        "grafana_datasource",
        "grafana_alert_datasource",
        "cluster_name",
    ):
        if field in data:
            updates[field] = data[field]
    if "grafana_dashboards" in data:
        for dashboard_key, uid in data["grafana_dashboards"].items():
            updates[f"dashboard_{dashboard_key}"] = uid

    store.update(updates, username=username)
    return JSONResponse(content={"status": "ok"})


async def check_grafana(
    request: Request,
    body: CheckGrafanaRequest = Body(...),
) -> JSONResponse:
    grafana_url = body.grafana_url.rstrip("/")
    if not grafana_url:
        return JSONResponse(
            content={"ok": False, "error": "Grafana URL is empty"},
            status_code=400,
        )

    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{grafana_url}/api/health")
            resp.raise_for_status()
            return JSONResponse(content={"ok": True, "body": resp.json()})
    except Exception as e:
        return JSONResponse(
            content={"ok": False, "error": str(e)},
            status_code=200,
        )


async def reset_monitor_config(request: Request) -> JSONResponse:
    store = request.app.state.monitor_config_store
    store.reset()
    return JSONResponse(content={"status": "ok"})


_FIELD_NAME_RE = re.compile(r"^[a-zA-Z0-9_.@]+$")
_TEXT_FIELDS = {"message"}
_ES_PIT_KEEP_ALIVE = "1m"
_ES_SEARCH_AFTER_BATCH_SIZE = 5000
_ES_MAX_SEARCH_AFTER_REQUESTS = 10


def _get_es_total(data: dict[str, Any]) -> int:
    total_value = data.get("hits", {}).get("total", 0)
    return (
        int(total_value.get("value", 0))
        if isinstance(total_value, dict)
        else int(total_value)
    )


async def _search_es_page(
    session: aiohttp.ClientSession,
    *,
    es_url: str,
    es_index: str,
    headers: dict[str, str],
    query: dict[str, Any],
    page_from: int,
    size: int,
    source: Optional[dict[str, Any]] = None,
    error_context: str = "ES",
) -> tuple[list[dict[str, Any]], int]:
    """Fetch an arbitrary page with PIT-backed ``search_after`` queries.

    Elasticsearch limits ``from + size`` pagination to 10,000 hits by
    default.  A PIT supplies the stable ``_shard_doc`` tiebreaker required to
    walk past that boundary safely.  For pages closer to the end of the result
    set, walking in ascending order bounds the work by the nearer edge.
    """
    base_url = es_url.rstrip("/")
    pit_id = ""

    async def _read_response(
        resp: aiohttp.ClientResponse, operation: str
    ) -> dict[str, Any]:
        if resp.status != 200:
            text = await resp.text()
            logger.error(
                "%s %s failed: status=%d body=%s",
                error_context,
                operation,
                resp.status,
                text[:500],
            )
            raise HTTPException(status_code=502, detail="Elasticsearch query failed")
        return await resp.json()

    async def _search(
        body: dict[str, Any], *, include_source: bool = True
    ) -> dict[str, Any]:
        nonlocal pit_id
        request_body = {
            **body,
            "pit": {"id": pit_id, "keep_alive": _ES_PIT_KEEP_ALIVE},
        }
        if include_source and source is not None:
            request_body["_source"] = source
        async with session.post(
            f"{base_url}/_search", json=request_body, headers=headers
        ) as resp:
            data = await _read_response(resp, "search")
        pit_id = data.get("pit_id") or pit_id
        return data

    try:
        async with session.post(
            f"{base_url}/{es_index}/_pit",
            params={"keep_alive": _ES_PIT_KEEP_ALIVE},
            headers=headers,
        ) as resp:
            pit_data = await _read_response(resp, "PIT open")
        pit_id = pit_data.get("id", "")
        if not pit_id:
            logger.error("%s PIT open returned no id", error_context)
            raise HTTPException(status_code=502, detail="Elasticsearch query failed")

        count_data = await _search(
            {"query": query, "size": 0, "track_total_hits": True, "_source": False},
            include_source=False,
        )
        total = _get_es_total(count_data)
        if page_from >= total:
            return [], total

        result_size = min(size, total - page_from)
        reverse_offset = total - (page_from + result_size)
        ascending = reverse_offset < page_from
        remaining = reverse_offset if ascending else page_from
        max_traversal_hits = _ES_SEARCH_AFTER_BATCH_SIZE * _ES_MAX_SEARCH_AFTER_REQUESTS
        if remaining > max_traversal_hits:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Requested page is too far from either end of the result set; "
                    "narrow the filters or time range"
                ),
            )
        order = "asc" if ascending else "desc"
        sort = [
            {
                "@timestamp": {
                    "order": order,
                    "format": "strict_date_optional_time_nanos",
                    "numeric_type": "date_nanos",
                }
            },
            {"_shard_doc": order},
        ]
        search_after: Optional[list[Any]] = None

        while remaining > 0:
            batch_size = min(remaining, _ES_SEARCH_AFTER_BATCH_SIZE)
            batch_body: dict[str, Any] = {
                "query": query,
                "size": batch_size,
                "sort": sort,
                "track_total_hits": False,
                "_source": False,
            }
            if search_after is not None:
                batch_body["search_after"] = search_after
            batch_data = await _search(batch_body, include_source=False)
            batch_hits = batch_data.get("hits", {}).get("hits", [])
            if not batch_hits:
                return [], total
            search_after = batch_hits[-1].get("sort")
            if not search_after:
                logger.error("%s search hit returned no sort values", error_context)
                raise HTTPException(
                    status_code=502, detail="Elasticsearch query failed"
                )
            remaining -= len(batch_hits)
            if len(batch_hits) < batch_size:
                return [], total

        page_body: dict[str, Any] = {
            "query": query,
            "size": result_size,
            "sort": sort,
            "track_total_hits": False,
        }
        if search_after is not None:
            page_body["search_after"] = search_after
        page_data = await _search(page_body)
        page_hits = page_data.get("hits", {}).get("hits", [])
        if ascending:
            page_hits.reverse()
        return [hit["_source"] for hit in page_hits], total
    finally:
        if pit_id:
            try:
                async with session.delete(
                    f"{base_url}/_pit", json={"id": pit_id}, headers=headers
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(
                            "%s PIT close failed: status=%d body=%s",
                            error_context,
                            resp.status,
                            text[:500],
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning("%s PIT close failed: %s", error_context, e)


async def search_logs(
    q: str = "",
    level: str = "",
    module: str = "",
    node: str = "",
    log_type: str = "",
    filters: list[str] = Query(
        [], description="Field filters, e.g. filters=+node:val1&filters=-level:val2"
    ),
    time_from: str = "now-1h",
    time_to: str = "now",
    size: int = 200,
    page_from: int = 0,
    node_field: str = "node",
) -> JSONResponse:
    es_url = os.environ.get("XINFERENCE_ES_URL", "")
    if not es_url:
        raise HTTPException(status_code=503, detail="Elasticsearch is not configured")

    es_index = os.environ.get("XINFERENCE_ES_INDEX", "xinference-logs-*")
    es_auth = os.environ.get("XINFERENCE_ES_AUTH", "")

    size = max(1, min(size, 500))
    page_from = max(0, page_from)
    time_from, time_to = _freeze_es_time_bounds(time_from, time_to)
    if node_field not in ("node", "node.keyword"):
        node_field = "node"

    must = []
    filter_clauses: list[dict[str, Any]] = [
        {"range": {"@timestamp": {"gte": time_from, "lte": time_to}}}
    ]

    if q:
        must.append(
            {
                "simple_query_string": {
                    "query": q,
                    "fields": ["message"],
                    "default_operator": "AND",
                }
            }
        )

    for field, value in [
        ("level", level),
        ("module", module),
        (node_field, node),
        ("log_type", log_type),
    ]:
        if value:
            terms = [v.strip() for v in value.split(",") if v.strip()]
            if terms:
                filter_clauses.append({"terms": {field: terms}})

    must_not: list[dict[str, Any]] = []
    plus_filters: dict[str, list[str]] = {}
    for token in filters:
        token = token.strip()
        if len(token) < 3 or token[0] not in ("+", "-"):
            continue
        sep = token.find(":", 1)
        if sep < 0:
            continue
        op = token[0]
        field_name = token[1:sep]
        field_value = token[sep + 1 :]
        if field_name == "node":
            field_name = node_field
        if not _FIELD_NAME_RE.match(field_name) or not field_value:
            continue
        if op == "+":
            plus_filters.setdefault(field_name, []).append(field_value)
        else:
            if field_name in _TEXT_FIELDS:
                must_not.append({"match_phrase": {field_name: field_value}})
            else:
                must_not.append({"term": {field_name: field_value}})

    for field_name, values in plus_filters.items():
        if field_name in _TEXT_FIELDS:
            filter_clauses.append(
                {
                    "bool": {
                        "should": [{"match_phrase": {field_name: v}} for v in values]
                    }
                }
            )
        else:
            filter_clauses.append({"terms": {field_name: values}})

    query: dict[str, Any] = {
        "bool": {"must": must, "filter": filter_clauses, "must_not": must_not}
    }

    headers = {"Content-Type": "application/json"}
    auth = None
    if es_auth:
        if es_auth.startswith("ApiKey "):
            headers["Authorization"] = es_auth
        else:
            parts = es_auth.split(":", 1)
            if len(parts) == 2:
                auth = aiohttp.BasicAuth(parts[0], parts[1])

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, auth=auth) as session:
            hits, total = await _search_es_page(
                session,
                es_url=es_url,
                es_index=es_index,
                headers=headers,
                query=query,
                page_from=page_from,
                size=size,
                source={"excludes": ["@version"]},
                error_context="ES log query",
            )
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error("ES connection error or timeout: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to Elasticsearch or query timed out",
        )

    return JSONResponse(content={"hits": hits, "total": total})


async def list_log_nodes() -> JSONResponse:
    es_url = os.environ.get("XINFERENCE_ES_URL", "")
    if not es_url:
        raise HTTPException(status_code=503, detail="Elasticsearch is not configured")

    es_index = os.environ.get("XINFERENCE_ES_INDEX", "xinference-logs-*")
    es_auth = os.environ.get("XINFERENCE_ES_AUTH", "")

    headers = {"Content-Type": "application/json"}
    auth = None
    if es_auth:
        if es_auth.startswith("ApiKey "):
            headers["Authorization"] = es_auth
        else:
            parts = es_auth.split(":", 1)
            if len(parts) == 2:
                auth = aiohttp.BasicAuth(parts[0], parts[1])

    url = f"{es_url.rstrip('/')}/{es_index}/_search"

    async def _aggregate(field: str) -> Optional[list[dict[str, Any]]]:
        body = {"size": 0, "aggs": {"nodes": {"terms": {"field": field, "size": 200}}}}
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, auth=auth) as session:
            async with session.post(url, json=body, headers=headers) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
        return data.get("aggregations", {}).get("nodes", {}).get("buckets", [])

    try:
        # "node" is usually a keyword field; fall back to "node.keyword" if the
        # mapping is text (terms aggregation requires a keyword/fielddata field).
        buckets = await _aggregate("node")
        node_field = "node"
        if buckets is None:
            buckets = await _aggregate("node.keyword")
            node_field = "node.keyword"
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error("ES connection error or timeout: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to Elasticsearch or query timed out",
        )

    if buckets is None:
        logger.error("ES node aggregation failed for both 'node' and 'node.keyword'")
        raise HTTPException(status_code=502, detail="Elasticsearch query failed")

    nodes = [b["key"] for b in buckets if b.get("key")]
    return JSONResponse(content={"nodes": nodes, "node_field": node_field})


async def search_logs_context(
    timestamp: str = "",
    size: int = 5,
    node: str = "",
    node_field: str = "node",
) -> JSONResponse:
    if not timestamp:
        raise HTTPException(status_code=400, detail="timestamp is required")

    es_url = os.environ.get("XINFERENCE_ES_URL", "")
    if not es_url:
        raise HTTPException(status_code=503, detail="Elasticsearch is not configured")

    es_index = os.environ.get("XINFERENCE_ES_INDEX", "xinference-logs-*")
    es_auth = os.environ.get("XINFERENCE_ES_AUTH", "")

    size = max(1, min(size, 50))
    if node_field not in ("node", "node.keyword"):
        node_field = "node"

    node_filter: list[dict[str, Any]] = []
    if node:
        node_filter = [{"term": {node_field: node}}]

    older_body: dict[str, Any] = {
        "query": {
            "bool": {
                "filter": [
                    {"range": {"@timestamp": {"lt": timestamp}}},
                    *node_filter,
                ]
            }
        },
        "sort": [{"@timestamp": "desc"}],
        "size": size + 1,
        "_source": {"excludes": ["@version"]},
    }

    newer_body: dict[str, Any] = {
        "query": {
            "bool": {
                "filter": [
                    {"range": {"@timestamp": {"gt": timestamp}}},
                    *node_filter,
                ]
            }
        },
        "sort": [{"@timestamp": "asc"}],
        "size": size + 1,
        "_source": {"excludes": ["@version"]},
    }

    headers = {"Content-Type": "application/json"}
    auth = None
    if es_auth:
        if es_auth.startswith("ApiKey "):
            headers["Authorization"] = es_auth
        else:
            parts = es_auth.split(":", 1)
            if len(parts) == 2:
                auth = aiohttp.BasicAuth(parts[0], parts[1])

    url = f"{es_url.rstrip('/')}/{es_index}/_search"

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, auth=auth) as session:

            async def fetch(body: dict, name: str) -> dict:
                async with session.post(url, json=body, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(
                            "ES context %s query failed: status=%d body=%s",
                            name,
                            resp.status,
                            text[:500],
                        )
                        raise HTTPException(
                            status_code=502,
                            detail="Elasticsearch query failed",
                        )
                    return await resp.json()

            older_data, newer_data = await asyncio.gather(
                fetch(older_body, "older"),
                fetch(newer_body, "newer"),
            )
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error("ES connection error or timeout: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to Elasticsearch or query timed out",
        )

    older_hits_raw = [
        hit["_source"] for hit in older_data.get("hits", {}).get("hits", [])
    ]
    newer_hits_raw = [
        hit["_source"] for hit in newer_data.get("hits", {}).get("hits", [])
    ]

    has_more_older = len(older_hits_raw) > size
    has_more_newer = len(newer_hits_raw) > size

    older_hits = older_hits_raw[:size]
    newer_hits = newer_hits_raw[:size]

    return JSONResponse(
        content={
            "older": older_hits,
            "newer": newer_hits,
            "anchor_timestamp": timestamp,
            "has_more_older": has_more_older,
            "has_more_newer": has_more_newer,
        }
    )


# --- Route registration ---


def _parse_relative_time(
    expr: str, *, now: Optional[datetime] = None
) -> Optional[datetime]:
    """Parse ES-style relative time, epoch milliseconds, or ISO timestamp."""
    reference_time = now or datetime.now(timezone.utc)

    if expr == "now":
        return reference_time
    m = re.match(r"now-(\d+)([mhdw])", expr)
    if m:
        val, unit = int(m.group(1)), m.group(2)
        delta = {
            "m": timedelta(minutes=val),
            "h": timedelta(hours=val),
            "d": timedelta(days=val),
            "w": timedelta(weeks=val),
        }.get(unit)
        if delta is None:
            return None
        return reference_time - delta
    # Epoch milliseconds (numeric string like "1716854400000")
    if expr.isdigit():
        return datetime.fromtimestamp(int(expr) / 1000, tz=timezone.utc)
    # ISO 8601 timestamp (e.g. "2026-05-28T03:00:00.000Z")
    try:
        return datetime.fromisoformat(expr.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass
    return None


def _freeze_es_time_bounds(
    time_from: str,
    time_to: str,
    *,
    now: Optional[datetime] = None,
) -> tuple[str, str]:
    """Resolve supported relative date math against one shared instant."""
    reference_time = now or datetime.now(timezone.utc)

    def _freeze(value: str) -> str:
        if value != "now" and re.fullmatch(r"now-\d+[mhdw]", value) is None:
            return value
        parsed = _parse_relative_time(value, now=reference_time)
        return parsed.isoformat().replace("+00:00", "Z") if parsed else value

    return _freeze(time_from), _freeze(time_to)


def _escape_es_wildcard(value: str) -> str:
    """Escape characters that are special to an Elasticsearch wildcard query."""
    return value.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?")


def _es_substring_clause(field_name: str, value: str) -> dict[str, Any]:
    """Build a substring query compatible with common ES string mappings."""
    pattern = f"*{_escape_es_wildcard(value)}*"
    return {
        "bool": {
            "should": [
                {"wildcard": {field: {"value": pattern, "case_insensitive": True}}}
                for field in (field_name, f"{field_name}.keyword")
            ],
            "minimum_should_match": 1,
        }
    }


_AUDIT_TEXT_FILTER_FIELDS = (
    "user",
    "api_key_name",
    "model_id",
    "model_name",
    "client_ip",
)
_AUDIT_FILTER_OPTION_LIMIT = 500


def _add_bounded_audit_filter_option(
    options: list[tuple[str, str]], seen: set[str], value: Any
) -> None:
    if value is None:
        return
    text = str(value)
    if not text or text in seen:
        return

    item = (text.casefold(), text)
    position = bisect_left(options, item)
    if len(options) >= _AUDIT_FILTER_OPTION_LIMIT and position >= len(options):
        return

    options.insert(position, item)
    seen.add(text)
    if len(options) > _AUDIT_FILTER_OPTION_LIMIT:
        _, removed = options.pop()
        seen.remove(removed)


def _audit_filter_aggregation_body(
    time_from: str,
    time_to: str,
    fields: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    fields = fields or {
        field_name: field_name for field_name in _AUDIT_TEXT_FILTER_FIELDS
    }
    return {
        "size": 0,
        "query": {"range": {"@timestamp": {"gte": time_from, "lte": time_to}}},
        "aggs": {
            response_field: {
                "terms": {
                    "field": es_field,
                    "size": _AUDIT_FILTER_OPTION_LIMIT,
                }
            }
            for response_field, es_field in fields.items()
        },
    }


def _aggregatable_field_indices(
    field_caps: dict[str, Any], field_name: str, all_indices: set[str]
) -> set[str]:
    result: set[str] = set()
    capabilities = field_caps.get("fields", {}).get(field_name, {})
    for capability in capabilities.values():
        capability_indices = set(capability.get("indices") or all_indices)
        non_aggregatable = set(capability.get("non_aggregatable_indices") or [])
        if capability.get("aggregatable") or non_aggregatable:
            result.update(capability_indices - non_aggregatable)
    return result


def _audit_filter_field_groups(
    field_caps: dict[str, Any],
) -> list[tuple[list[str], dict[str, str]]]:
    all_indices = {str(index_name) for index_name in field_caps.get("indices", [])}
    index_fields: dict[str, dict[str, str]] = {
        index_name: {} for index_name in all_indices
    }

    for field_name in _AUDIT_TEXT_FILTER_FIELDS:
        direct_indices = _aggregatable_field_indices(
            field_caps, field_name, all_indices
        )
        keyword_field = f"{field_name}.keyword"
        keyword_indices = _aggregatable_field_indices(
            field_caps, keyword_field, all_indices
        )
        for index_name in all_indices:
            if index_name in direct_indices:
                index_fields[index_name][field_name] = field_name
            elif index_name in keyword_indices:
                index_fields[index_name][field_name] = keyword_field

    groups: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for index_name, fields in index_fields.items():
        if not fields:
            continue
        signature = tuple(sorted(fields.items()))
        groups.setdefault(signature, []).append(index_name)

    return [
        (sorted(indices), dict(signature))
        for signature, indices in sorted(groups.items())
    ]


def _audit_entry_in_time_range(
    entry: dict[str, Any], t_from: Optional[datetime], t_to: Optional[datetime]
) -> bool:
    if not t_from and not t_to:
        return True
    try:
        timestamp = datetime.fromisoformat(
            str(entry.get("@timestamp", "")).replace("Z", "+00:00")
        )
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return not ((t_from and timestamp < t_from) or (t_to and timestamp > t_to))
    except (ValueError, TypeError):
        return False


def _match_substring(stored: Any, needle: str) -> bool:
    """Case-insensitive substring match used by audit free-text filters.

    An empty ``needle`` means the filter is not active and matches everything.
    """
    if not needle:
        return True
    if not isinstance(stored, str):
        return False
    return needle.lower() in stored.lower()


def _match_enum(stored: Any, allowed: set) -> bool:
    """Case-insensitive exact match against a set of allowed values.

    An empty ``allowed`` set means the filter is not active.
    """
    if not allowed:
        return True
    if not isinstance(stored, str):
        return False
    return stored.lower() in allowed


def _audit_log_paths() -> list[str]:
    """Current audit.log plus its rotated siblings, oldest first."""
    import glob

    from ...constants import XINFERENCE_LOG_DIR

    base = os.path.join(XINFERENCE_LOG_DIR, "audit.log")
    return sorted(path for path in glob.glob(base + "*") if os.path.isfile(path))


def _iter_audit_entries(
    *,
    time_from: str,
    time_to: str,
    user: str = "",
    api_key_name: str = "",
    model_id: str = "",
    model_name: str = "",
    model: str = "",
    model_type: str = "",
    category: str = "",
    auth_type: str = "",
    status: str = "",
    client_ip: str = "",
) -> list[dict]:
    """Read audit.log and its rotated files, returning the matching entries."""
    t_from = _parse_relative_time(time_from)
    t_to = _parse_relative_time(time_to)

    def _enum_set(value: str) -> set:
        return {v.strip().lower() for v in value.split(",") if v.strip()}

    enum_filters = [
        ("status", _enum_set(status)),
        ("category", _enum_set(category)),
        ("model_type", _enum_set(model_type)),
        ("auth_type", _enum_set(auth_type)),
    ]
    text_filters = [
        ("user", user),
        ("api_key_name", api_key_name),
        ("model_id", model_id),
        ("model_name", model_name),
        ("client_ip", client_ip),
    ]

    results: list[dict] = []
    for path in _audit_log_paths():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    # A line may be valid JSON yet not an object (e.g. `null`,
                    # `[1,2]`); `entry.get(...)` below would raise AttributeError.
                    if not isinstance(entry, dict):
                        continue
                    if not _audit_entry_in_time_range(entry, t_from, t_to):
                        continue
                    if not all(
                        _match_substring(entry.get(field), needle)
                        for field, needle in text_filters
                        if needle
                    ):
                        continue
                    if not all(
                        _match_enum(entry.get(field), allowed)
                        for field, allowed in enum_filters
                        if allowed
                    ):
                        continue
                    # One box for the model: entries written before a model
                    # finished loading carry only the uid, so matching just
                    # model_name would silently hide them.
                    if model and not (
                        _match_substring(entry.get("model_name"), model)
                        or _match_substring(entry.get("model_id"), model)
                    ):
                        continue
                    results.append(entry)
        except OSError:
            continue
    return results


async def _search_audit_from_file(
    *,
    time_from: str,
    time_to: str,
    user: str,
    api_key_name: str,
    model_id: str,
    model_name: str,
    model: str,
    model_type: str,
    category: str,
    auth_type: str,
    status: str,
    client_ip: str,
    page_from: int,
    size: int,
) -> JSONResponse:
    """Fallback: search audit events from local audit.log file."""
    results = await asyncio.to_thread(
        _iter_audit_entries,
        time_from=time_from,
        time_to=time_to,
        user=user,
        api_key_name=api_key_name,
        model_id=model_id,
        model_name=model_name,
        model=model,
        model_type=model_type,
        category=category,
        auth_type=auth_type,
        status=status,
        client_ip=client_ip,
    )
    results.sort(key=lambda x: x.get("@timestamp", ""), reverse=True)
    total = len(results)
    hits = results[page_from : page_from + size]
    return JSONResponse(content={"hits": hits, "total": total})


async def _list_audit_filter_options_from_file(
    *, time_from: str, time_to: str
) -> JSONResponse:
    t_from = _parse_relative_time(time_from)
    t_to = _parse_relative_time(time_to)

    def _read_options() -> dict[str, list[str]]:
        options: dict[str, list[tuple[str, str]]] = {
            field_name: [] for field_name in _AUDIT_TEXT_FILTER_FIELDS
        }
        seen: dict[str, set[str]] = {
            field_name: set() for field_name in _AUDIT_TEXT_FILTER_FIELDS
        }
        for audit_path in _audit_log_paths():
            try:
                with open(audit_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        if not isinstance(entry, dict):
                            continue
                        if not _audit_entry_in_time_range(entry, t_from, t_to):
                            continue
                        for field_name in _AUDIT_TEXT_FILTER_FIELDS:
                            _add_bounded_audit_filter_option(
                                options[field_name],
                                seen[field_name],
                                entry.get(field_name),
                            )
            except OSError:
                continue

        return {key: [value for _, value in values] for key, values in options.items()}

    content = await asyncio.to_thread(_read_options)
    return JSONResponse(content=content)


async def list_audit_filter_options(
    time_from: str = "now-1h", time_to: str = "now"
) -> JSONResponse:
    es_url = os.environ.get("XINFERENCE_ES_URL", "")
    if not es_url:
        return await _list_audit_filter_options_from_file(
            time_from=time_from, time_to=time_to
        )

    from ...constants import XINFERENCE_AUDIT_ES_INDEX

    headers = {"Content-Type": "application/json"}
    auth = None
    es_auth = os.environ.get("XINFERENCE_ES_AUTH", "")
    if es_auth:
        if es_auth.startswith("ApiKey "):
            headers["Authorization"] = es_auth
        else:
            parts = es_auth.split(":", 1)
            if len(parts) == 2:
                auth = aiohttp.BasicAuth(parts[0], parts[1])

    es_base_url = es_url.rstrip("/")
    field_names = ",".join(
        field_name
        for response_field in _AUDIT_TEXT_FILTER_FIELDS
        for field_name in (response_field, f"{response_field}.keyword")
    )
    field_caps_url = (
        f"{es_base_url}/{XINFERENCE_AUDIT_ES_INDEX}/_field_caps"
        f"?fields={field_names}&include_unmapped=true"
    )
    option_values: dict[str, list[tuple[str, str]]] = {
        field_name: [] for field_name in _AUDIT_TEXT_FILTER_FIELDS
    }
    seen_values: dict[str, set[str]] = {
        field_name: set() for field_name in _AUDIT_TEXT_FILTER_FIELDS
    }
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, auth=auth) as session:

            async def fetch(
                url: str, request_body: Optional[dict[str, Any]] = None
            ) -> tuple[int, str, dict[str, Any]]:
                request_kwargs: dict[str, Any] = {"headers": headers}
                if request_body is not None:
                    request_kwargs["json"] = request_body
                async with session.post(url, **request_kwargs) as resp:
                    if resp.status != 200:
                        return resp.status, await resp.text(), {}
                    return resp.status, "", await resp.json()

            status, response_text, field_caps = await fetch(field_caps_url)
            if status != 200:
                logger.error(
                    "ES audit field capabilities query failed: status=%d body=%s",
                    status,
                    response_text[:500],
                )
                raise HTTPException(
                    status_code=502, detail="Elasticsearch query failed"
                )

            for indices, fields in _audit_filter_field_groups(field_caps):
                search_url = f"{es_base_url}/{','.join(indices)}/_search"
                body = _audit_filter_aggregation_body(time_from, time_to, fields=fields)
                status, response_text, data = await fetch(search_url, body)
                if status != 200:
                    logger.error(
                        "ES audit filter aggregation failed: status=%d body=%s",
                        status,
                        response_text[:500],
                    )
                    raise HTTPException(
                        status_code=502, detail="Elasticsearch query failed"
                    )

                aggregations = data.get("aggregations", {})
                for field_name in _AUDIT_TEXT_FILTER_FIELDS:
                    for bucket in aggregations.get(field_name, {}).get("buckets", []):
                        _add_bounded_audit_filter_option(
                            option_values[field_name],
                            seen_values[field_name],
                            bucket.get("key"),
                        )
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
        logger.error("ES connection error, timeout, or invalid response: %s", e)
        raise HTTPException(status_code=502, detail="Audit service unavailable")

    return JSONResponse(
        content={
            field_name: [value for _, value in option_values[field_name]]
            for field_name in option_values
        }
    )


async def search_audit_logs(
    time_from: str = "now-1h",
    time_to: str = "now",
    user: str = "",
    api_key_name: str = "",
    model_id: str = "",
    model_name: str = "",
    model: str = "",
    model_type: str = "",
    category: str = "",
    auth_type: str = "",
    status: str = "",
    client_ip: str = "",
    page_from: int = 0,
    size: int = 50,
) -> JSONResponse:
    es_url = os.environ.get("XINFERENCE_ES_URL", "")
    if not es_url:
        return await _search_audit_from_file(
            time_from=time_from,
            time_to=time_to,
            user=user,
            api_key_name=api_key_name,
            model_id=model_id,
            model_name=model_name,
            model=model,
            model_type=model_type,
            category=category,
            auth_type=auth_type,
            status=status,
            client_ip=client_ip,
            page_from=page_from,
            size=size,
        )

    from ...constants import XINFERENCE_AUDIT_ES_INDEX

    es_index = XINFERENCE_AUDIT_ES_INDEX
    es_auth = os.environ.get("XINFERENCE_ES_AUTH", "")

    size = max(1, min(size, 500))
    page_from = max(0, page_from)
    time_from, time_to = _freeze_es_time_bounds(time_from, time_to)

    must: list[dict[str, Any]] = []
    filter_clauses: list[dict[str, Any]] = [
        {"range": {"@timestamp": {"gte": time_from, "lte": time_to}}}
    ]

    for field_name, value in [
        ("user", user),
        ("api_key_name", api_key_name),
        ("model_id", model_id),
        ("model_name", model_name),
        ("client_ip", client_ip),
    ]:
        if value:
            # Substring, case-insensitive, to match the file-mode semantics.
            # Dynamic mapping commonly creates `text` + `.keyword`, while an
            # index template may map the field directly as `keyword` or
            # `wildcard`. Query both names so either mapping works. An unmapped
            # alternative simply contributes no match.
            #
            # NOTE: a leading wildcard cannot use the index and forces a scan of
            # all terms in the segment. That is acceptable for typical audit
            # volumes, but on a large index deployments should install an index
            # template mapping these fields directly to the `wildcard` type
            # (ES >= 7.9), which is covered by the first alternative.
            filter_clauses.append(_es_substring_clause(field_name, value))

    if model:
        filter_clauses.append(
            {
                "bool": {
                    "should": [
                        _es_substring_clause("model_name", model),
                        _es_substring_clause("model_id", model),
                    ],
                    "minimum_should_match": 1,
                }
            }
        )

    for field_name, value in [
        ("model_type", model_type),
        ("category", category),
        ("auth_type", auth_type),
        ("status", status),
    ]:
        if value:
            terms = [v.strip().lower() for v in value.split(",") if v.strip()]
            if len(terms) == 1:
                filter_clauses.append({"term": {field_name: terms[0]}})
            else:
                filter_clauses.append({"terms": {field_name: terms}})

    query: dict[str, Any] = {"bool": {"must": must, "filter": filter_clauses}}

    headers = {"Content-Type": "application/json"}
    auth = None
    if es_auth:
        if es_auth.startswith("ApiKey "):
            headers["Authorization"] = es_auth
        else:
            parts = es_auth.split(":", 1)
            if len(parts) == 2:
                auth = aiohttp.BasicAuth(parts[0], parts[1])

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, auth=auth) as session:
            hits, total = await _search_es_page(
                session,
                es_url=es_url,
                es_index=es_index,
                headers=headers,
                query=query,
                page_from=page_from,
                size=size,
                error_context="ES audit query",
            )
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error("ES connection error or timeout: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Audit service unavailable",
        )

    return JSONResponse(content={"hits": hits, "total": total})


_AUDIT_STATS_INTERVALS = (
    (timedelta(hours=3), timedelta(minutes=5), "5m"),
    (timedelta(hours=12), timedelta(minutes=15), "15m"),
    (timedelta(days=2), timedelta(hours=1), "1h"),
    (timedelta(days=14), timedelta(hours=6), "6h"),
)


def _pick_stats_interval(span: timedelta) -> tuple[timedelta, str]:
    for limit, step, label in _AUDIT_STATS_INTERVALS:
        if span <= limit:
            return step, label
    return timedelta(days=1), "1d"


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * q))
    return round(sorted_values[min(index, len(sorted_values) - 1)], 1)


def _latency_summary(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {
        "avg": round(sum(values) / len(values), 1) if values else 0.0,
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": round(values[-1], 1) if values else 0.0,
    }


def _group_slot(entry: dict, step_seconds: int) -> Optional[int]:
    try:
        timestamp = datetime.fromisoformat(
            str(entry.get("@timestamp", "")).replace("Z", "+00:00")
        )
    except (ValueError, TypeError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return int(timestamp.timestamp()) // step_seconds * step_seconds


class _Group:
    """Running totals for one model or one API key."""

    def __init__(self, key: str):
        self.key = key
        self.count = 0
        self.errors = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latencies: list[float] = []
        self.ttfts: list[float] = []
        self.speeds: list[float] = []
        self.members: set[str] = set()
        self.buckets: dict[int, int] = {}
        self.labels: dict[str, str] = {}

    def add(self, entry: dict, *, ok: bool, slot: Optional[int]) -> None:
        self.count += 1
        if not ok:
            self.errors += 1
        for field, bucket in (
            ("latency_ms", self.latencies),
            ("ttft_ms", self.ttfts),
            ("output_tps", self.speeds),
        ):
            value = entry.get(field)
            # Total duration only means something for a streamed answer here;
            # a non-streaming call's duration is dominated by output length.
            if field == "latency_ms" and entry.get("stream") is not True:
                continue
            if isinstance(value, (int, float)):
                bucket.append(float(value))
        for field in ("prompt_tokens", "completion_tokens"):
            value = entry.get(field)
            if isinstance(value, int):
                setattr(self, field, getattr(self, field) + value)
        if slot is not None:
            self.buckets[slot] = self.buckets.get(slot, 0) + 1

    def as_row(self, *, total: int, member_field: str, slots: list[int]) -> dict:
        return {
            "key": self.key,
            "count": self.count,
            "share": round(self.count / total * 100, 1) if total else 0.0,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            member_field: len(self.members),
            "latency": _latency_summary(self.latencies),
            "ttft": _latency_summary(self.ttfts),
            "tps": _latency_summary(self.speeds),
            "series": [self.buckets.get(slot, 0) for slot in slots],
            **self.labels,
        }


def _compute_audit_stats(
    entries: list[dict], *, t_from: datetime, t_to: datetime, top_n: int
) -> dict[str, Any]:
    step, step_label = _pick_stats_interval(t_to - t_from)
    step_seconds = int(step.total_seconds())

    latencies: list[float] = []
    ttfts: list[float] = []
    speeds: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    users: set[str] = set()
    api_keys: set[str] = set()
    models: set[str] = set()
    errors = 0
    buckets: dict[int, list[int]] = {}
    by_model: dict[str, _Group] = {}
    by_api_key: dict[str, _Group] = {}

    for entry in entries:
        ok = entry.get("status") == "success"
        if not ok:
            errors += 1

        latency = entry.get("latency_ms")
        if isinstance(latency, (int, float)) and entry.get("stream") is True:
            latencies.append(float(latency))
        for field, samples in (("ttft_ms", ttfts), ("output_tps", speeds)):
            value = entry.get(field)
            if isinstance(value, (int, float)):
                samples.append(float(value))
        for field, bucket_name in (
            ("prompt_tokens", "prompt"),
            ("completion_tokens", "completion"),
        ):
            value = entry.get(field)
            if isinstance(value, int):
                if bucket_name == "prompt":
                    prompt_tokens += value
                else:
                    completion_tokens += value

        user = str(entry.get("user") or "")
        model = str(entry.get("model_name") or entry.get("model_id") or "")
        api_key = str(entry.get("api_key_name") or user or "")
        if user:
            users.add(user)
        if model:
            models.add(model)
        if api_key:
            api_keys.add(api_key)

        slot = _group_slot(entry, step_seconds)
        if slot is not None:
            bucket = buckets.setdefault(slot, [0, 0])
            bucket[0] += 1
            if not ok:
                bucket[1] += 1

        if model:
            group = by_model.setdefault(model, _Group(model))
            group.add(entry, ok=ok, slot=slot)
            if api_key:
                group.members.add(api_key)
            group.labels.setdefault("model_type", str(entry.get("model_type") or ""))
        if api_key:
            group = by_api_key.setdefault(api_key, _Group(api_key))
            group.add(entry, ok=ok, slot=slot)
            if model:
                group.members.add(model)
            group.labels["user"] = user
            group.labels["client_ip"] = str(entry.get("client_ip") or "")

    start_slot = int(t_from.timestamp()) // step_seconds * step_seconds
    end_slot = int(t_to.timestamp()) // step_seconds * step_seconds
    slots = list(range(start_slot, end_slot + step_seconds, step_seconds))
    series = [
        {
            "ts": datetime.fromtimestamp(slot, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "count": buckets.get(slot, (0, 0))[0],
            "errors": buckets.get(slot, (0, 0))[1],
        }
        for slot in slots
    ]

    def _rows(groups: dict[str, _Group], member_field: str) -> list[dict]:
        rows = [
            group.as_row(total=len(entries), member_field=member_field, slots=slots)
            for group in groups.values()
        ]
        rows.sort(key=lambda row: row["count"], reverse=True)
        return rows[:top_n]

    return {
        "total": len(entries),
        "errors": errors,
        "unique_users": len(users),
        "unique_api_keys": len(api_keys),
        "unique_models": len(models),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency": _latency_summary(latencies),
        "ttft": _latency_summary(ttfts),
        "tps": _latency_summary(speeds),
        "interval": step_label,
        "series": series,
        "by_model": _rows(by_model, "api_keys"),
        "by_api_key": _rows(by_api_key, "models"),
    }


async def get_audit_stats(
    time_from: str = "now-24h",
    time_to: str = "now",
    user: str = "",
    api_key_name: str = "",
    model_name: str = "",
    model: str = "",
    model_type: str = "",
    category: str = "inference",
    status: str = "",
    client_ip: str = "",
    top_n: int = 10,
) -> JSONResponse:
    """Aggregate audit.log into call-volume / user / model / latency stats.

    Reads the local audit log even when ES is configured; the ES path has no
    aggregation implementation yet.
    """
    t_from = _parse_relative_time(time_from) or (
        datetime.now(timezone.utc) - timedelta(days=1)
    )
    t_to = _parse_relative_time(time_to) or datetime.now(timezone.utc)
    top_n = max(1, min(top_n, 100))

    entries = await asyncio.to_thread(
        _iter_audit_entries,
        time_from=time_from,
        time_to=time_to,
        user=user,
        api_key_name=api_key_name,
        model_name=model_name,
        model=model,
        model_type=model_type,
        category=category,
        status=status,
        client_ip=client_ip,
    )
    content = await asyncio.to_thread(
        _compute_audit_stats, entries, t_from=t_from, t_to=t_to, top_n=top_n
    )
    return JSONResponse(content=content)


# --- Route registration (original) ---


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()

    # Leaks worker addresses, host memory and per-GPU usage; the compose
    # healthcheck must not target it once this is gated.
    router.add_api_route(
        "/status",
        get_status,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route("/v1/address", get_address, methods=["GET"])
    router.add_api_route("/v1/cluster/auth", is_cluster_authenticated, methods=["GET"])
    router.add_api_route("/v1/cluster/ui_config", get_ui_config, methods=["GET"])

    router.add_api_route(
        "/v1/cluster/monitor_config",
        get_monitor_config,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cluster/monitor_config",
        update_monitor_config,
        methods=["PUT"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cluster/monitor_config/check-grafana",
        check_grafana,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cluster/monitor_config/reset",
        reset_monitor_config,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/cluster/info",
        get_cluster_device_info,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cluster/version",
        get_cluster_version,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cluster/devices",
        get_devices_count,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/workers",
        get_workers_info,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/supervisor",
        get_supervisor_info,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/clusters",
        abort_cluster,
        methods=["DELETE"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/cache/models",
        list_cached_models,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["cache:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cache/models",
        cache_model,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:write"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cache/models/files",
        list_model_files,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["cache:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cache/models",
        confirm_and_remove_model,
        methods=["DELETE"],
        dependencies=([Security(auth, scopes=["cache:delete"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cache/models/{cache_uid}/progress",
        get_cache_model_progress,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cache/models/{cache_uid}/cancel",
        cancel_cache_model,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:write"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/downloads",
        list_model_downloads,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/downloads/{cache_uid}/pause",
        pause_cache_model,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:write"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/downloads/{cache_uid}/resume",
        resume_cache_model,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:write"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/downloads/{cache_uid}",
        delete_cache_download,
        methods=["DELETE"],
        dependencies=([Security(auth, scopes=["cache:delete"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/virtualenvs",
        list_virtual_envs,
        methods=["GET"],
        dependencies=(
            [Security(auth, scopes=["virtualenv:list"])] if is_auth else None
        ),
    )
    router.add_api_route(
        "/v1/virtualenvs",
        remove_virtual_env,
        methods=["DELETE"],
        dependencies=(
            [Security(auth, scopes=["virtualenv:delete"])] if is_auth else None
        ),
    )

    router.add_api_route(
        "/v1/requests/{request_id}/progress",
        get_progress,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/cluster/logs",
        search_logs,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["logs:list"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/cluster/logs/context",
        search_logs_context,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["logs:list"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/cluster/logs/nodes",
        list_log_nodes,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["logs:list"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/audit/filter-options",
        list_audit_filter_options,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/audit/search",
        search_audit_logs,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/audit/stats",
        get_audit_stats,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
