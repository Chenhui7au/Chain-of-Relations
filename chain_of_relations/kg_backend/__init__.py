#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from __future__ import annotations

import os
from typing import Optional

from chain_of_relations.kg_backend.base import KGBackend
from chain_of_relations.kg_backend.freebase_backend import FreebaseBackend
from chain_of_relations.kg_backend.wikidata_backend import WikidataBackend


_DEFAULT_BACKEND: KGBackend = FreebaseBackend()
_WIKIDATA_BACKEND: KGBackend = WikidataBackend()


def get_backend(name: Optional[str] = None) -> KGBackend:
	backend_name = (name or os.getenv("KG_BACKEND", "freebase")).strip().lower()
	if backend_name in ("", "freebase"):
		return _DEFAULT_BACKEND
	if backend_name == "wikidata":
		return _WIKIDATA_BACKEND
	raise ValueError(f"Unsupported KG backend: {backend_name}")


def get_default_backend() -> KGBackend:
	return _DEFAULT_BACKEND


__all__ = ["KGBackend", "FreebaseBackend", "WikidataBackend", "get_backend", "get_default_backend"]
