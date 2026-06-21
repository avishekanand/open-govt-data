#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CBS StatLine OData v4 client.

Talks to the public CBS open-data API:

    https://datasets.cbs.nl/odata/v1/CBS/{TABLE_ID}

A table is a small semantic dataset, not a flat CSV. The base endpoint lists
entity sets; the important ones are:

    Properties      - table-level metadata (single object)
    Dimensions      - the axes of the table (e.g. WijkenEnBuurten, Perioden)
    MeasureCodes    - the measures/topics with titles, units, group ids
    MeasureGroups   - measure hierarchy
    {Dim}Codes      - code list (lookup) for each dimension
    {Dim}Groups     - hierarchy for each dimension
    Observations    - the statistical cells

This module only touches PUBLIC aggregate data. It never accesses confidential
microdata.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import requests

CBS_BASE = "https://datasets.cbs.nl/odata/v1/CBS"


class CbsODataClient:
    """Minimal, polite OData v4 client for CBS StatLine tables."""

    def __init__(
        self,
        base: str = CBS_BASE,
        timeout: int = 60,
        retries: int = 3,
        backoff: float = 1.5,
        delay: float = 0.0,
        session: Optional[requests.Session] = None,
        user_agent: str = "open-govt-data/cbs (research prototype)",
    ) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.delay = delay
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})

    # ------------------------------------------------------------------ core
    def get_json(self, url: str) -> Dict[str, Any]:
        """GET a single URL with retry/backoff, returning parsed JSON."""
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                if self.delay:
                    time.sleep(self.delay)
                return resp.json()
            except Exception as exc:  # noqa: BLE001 - we retry then re-raise
                last_err = exc
                if attempt < self.retries:
                    time.sleep(self.backoff ** attempt)
        raise RuntimeError(f"GET failed after {self.retries} attempts: {url}\n{last_err}")

    def get_odata(self, url: str) -> List[Dict[str, Any]]:
        """Fetch an OData entity set, following @odata.nextLink pagination.

        Returns the concatenated `value` arrays. If the endpoint returns a bare
        object (e.g. Properties), it is wrapped in a single-element list.
        """
        rows: List[Dict[str, Any]] = []
        next_url: Optional[str] = url
        while next_url:
            payload = self.get_json(next_url)
            value = payload.get("value")
            if value is None:
                # Bare entity (e.g. /Properties) — strip OData annotations.
                rows.append({k: v for k, v in payload.items() if not k.startswith("@odata")})
                break
            rows.extend(value)
            next_url = payload.get("@odata.nextLink")
        return rows

    # -------------------------------------------------------------- discovery
    def table_url(self, table_id: str) -> str:
        return f"{self.base}/{table_id}"

    def list_entities(self, table_id: str) -> List[str]:
        """Return the entity-set names exposed by a table's base endpoint."""
        payload = self.get_json(self.table_url(table_id))
        return [e.get("name") for e in payload.get("value", []) if e.get("name")]

    def fetch_properties(self, table_id: str) -> Dict[str, Any]:
        rows = self.get_odata(f"{self.table_url(table_id)}/Properties")
        return rows[0] if rows else {}

    def fetch_dimensions(self, table_id: str) -> List[Dict[str, Any]]:
        return self.get_odata(f"{self.table_url(table_id)}/Dimensions")

    def fetch_measure_codes(self, table_id: str) -> List[Dict[str, Any]]:
        return self.get_odata(f"{self.table_url(table_id)}/MeasureCodes")

    def fetch_measure_groups(self, table_id: str) -> List[Dict[str, Any]]:
        return self.get_odata(f"{self.table_url(table_id)}/MeasureGroups")

    def fetch_dimension_codes(self, table_id: str, dim: str) -> List[Dict[str, Any]]:
        """Code list for a dimension, e.g. WijkenEnBuurtenCodes."""
        return self.get_odata(f"{self.table_url(table_id)}/{dim}Codes")

    def fetch_dimension_groups(self, table_id: str, dim: str) -> List[Dict[str, Any]]:
        """Hierarchy for a dimension, e.g. WijkenEnBuurtenGroups."""
        return self.get_odata(f"{self.table_url(table_id)}/{dim}Groups")

    def fetch_table_metadata(self, table_id: str) -> Dict[str, Any]:
        """Fetch the full semantic layer for a table.

        Returns a dict with: entities, properties, dimensions, measure_codes,
        measure_groups, and per-dimension `codes`/`groups` (only for dimensions
        that advertise ContainsCodes / ContainsGroups).
        """
        entities = self.list_entities(table_id)
        dimensions = self.fetch_dimensions(table_id)
        codes: Dict[str, List[Dict[str, Any]]] = {}
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for dim in dimensions:
            key = dim.get("Identifier")
            if not key:
                continue
            if dim.get("ContainsCodes") and f"{key}Codes" in entities:
                codes[key] = self.fetch_dimension_codes(table_id, key)
            if dim.get("ContainsGroups") and f"{key}Groups" in entities:
                groups[key] = self.fetch_dimension_groups(table_id, key)
        return {
            "table_id": table_id,
            "entities": entities,
            "properties": self.fetch_properties(table_id),
            "dimensions": dimensions,
            "measure_codes": self.fetch_measure_codes(table_id),
            "measure_groups": self.fetch_measure_groups(table_id),
            "codes": codes,
            "groups": groups,
        }

    # ----------------------------------------------------------- observations
    @staticmethod
    def _odata_filter(field: str, values: Iterable[str]) -> str:
        clauses = [f"{field} eq '{v}'" for v in values]
        return " or ".join(clauses)

    def fetch_observations(
        self,
        table_id: str,
        filters: Optional[str] = None,
        select: Optional[List[str]] = None,
        top: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch observations, optionally filtered/selected.

        `filters` is a raw OData $filter expression. Use `where()` helpers or
        pass e.g. "WijkenEnBuurten eq 'GM0363'". Pagination is followed unless
        `top` caps the result.
        """
        params: List[str] = []
        if filters:
            params.append("$filter=" + quote(filters, safe="()' ="))
        if select:
            params.append("$select=" + ",".join(select))
        if top is not None:
            params.append(f"$top={top}")
        url = f"{self.table_url(table_id)}/Observations"
        if params:
            url += "?" + "&".join(params)
        return self.get_odata(url)

    def fetch_observations_for(
        self,
        table_id: str,
        dim: str,
        codes: Iterable[str],
        top: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience: observations where `dim` is in `codes`."""
        flt = self._odata_filter(dim, list(codes))
        return self.fetch_observations(table_id, filters=flt, top=top)
