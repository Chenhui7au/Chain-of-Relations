#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   dfs_stack.py
@Time    :   2026/03/08 11:23:09
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


from dataclasses import dataclass, field
from typing import Any, List, Optional

from chain_of_relations.schema import Entity, Relation


@dataclass
class DFSState:
	node: Any
	path: List[Any]
	depth: int


@dataclass
class DFSStack:
	items: List[DFSState] = field(default_factory=list)

	def is_empty(self) -> bool:
		return len(self.items) == 0

	def push(self, state: DFSState) -> None:
		self.items.append(state)

	def pop(self) -> Optional[DFSState]:
		if self.is_empty():
			return None
		return self.items.pop()


@dataclass
class DFSStackInitInput:
	root_entity: Entity


@dataclass
class DFSStackInitOutput:
	stack: DFSStack


def create_dfs_stack(inp: DFSStackInitInput) -> DFSStackInitOutput:
	stack = DFSStack(items=[DFSState(node=inp.root_entity, path=[inp.root_entity], depth=0)])
	return DFSStackInitOutput(stack=stack)


@dataclass
class PushRelationChildrenInput:
	stack: DFSStack
	current_state: DFSState
	selected_relations: List[Relation]
	max_depth: int


@dataclass
class PushRelationChildrenOutput:
	stack: DFSStack
	inserted_count: int


def push_relation_children(
	inp: PushRelationChildrenInput,
) -> PushRelationChildrenOutput:
	if inp.current_state.depth >= inp.max_depth:
		return PushRelationChildrenOutput(stack=inp.stack, inserted_count=0)

	inserted = 0
	next_depth = inp.current_state.depth + 1

	while inp.selected_relations:
		relation = inp.selected_relations.pop()
		next_path = list(inp.current_state.path)
		next_path.append(relation)
		inp.stack.push(DFSState(node=relation, path=next_path, depth=next_depth))
		inserted += 1

	return PushRelationChildrenOutput(stack=inp.stack, inserted_count=inserted)

